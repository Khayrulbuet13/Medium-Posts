from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Union
import time
import uuid
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.exceptions import RequestValidationError
import json
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    LogitsProcessorList,
    LogitsProcessor,
    StoppingCriteriaList,
    StoppingCriteria,
    MinLengthLogitsProcessor,
)

import logging, os
from logging.handlers import RotatingFileHandler

hf_token = os.environ.get('HUGGINGFACE_HUB_TOKEN')
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Consider specifying allowed origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



# Configure logging
# Updated logging configuration
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create handlers
console_handler = logging.StreamHandler()

# Use RotatingFileHandler for log rotation; create a new file if the size exceeds 5MB and keep the last 5 logs.
file_handler = RotatingFileHandler('app.log', maxBytes=5*1024*1024, backupCount=5)

# Set levels for handlers
console_handler.setLevel(logging.INFO)
file_handler.setLevel(logging.INFO)

# Create a formatter and set it for both handlers
log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
formatter = logging.Formatter(log_format)

console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

# Add the handlers to the logger
logger.addHandler(console_handler)
logger.addHandler(file_handler)


# Exception handler for validation errors
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    logger.error(f"Validation Error: {exc}")
    return JSONResponse(
        status_code=400,
        content={
            "error": {
                "message": "Invalid request.",
                "type": "invalid_request_error",
                "param": None,
                "code": None,
            }
        },
    )

# Define the message structure
class Message(BaseModel):
    role: str
    content: Union[str, List[dict], None] = None
    name: Optional[str] = None
    function_call: Optional[dict] = None

    class Config:
        extra = "allow"

# Define the request body structure
class ChatCompletionRequest(BaseModel):
    model: Optional[str] = "meta-llama/Llama-3.2-3B-Instruct"
    messages: List[Message]
    temperature: Optional[float] = Field(default=1.0, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(default=1.0, ge=0.0, le=1.0)
    n: Optional[int] = Field(default=1, ge=1)
    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None
    max_tokens: Optional[int] = Field(default=100, ge=1)
    presence_penalty: Optional[float] = Field(default=0.0, ge=-2.0, le=2.0)
    frequency_penalty: Optional[float] = Field(default=0.0, ge=-2.0, le=2.0)
    logit_bias: Optional[dict] = None
    user: Optional[str] = None
    functions: Optional[List[dict]] = None  # Not implemented
    function_call: Optional[Union[str, dict]] = None  # Not implemented

# Load model and tokenizer
model_name = "meta-llama/Llama-3.2-3B-Instruct"
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

# tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, token=hf_token)




# Determine torch_dtype based on GPU capabilities
if torch.cuda.is_available():
    if torch.cuda.get_device_capability(0)[0] >= 8:
        torch_dtype = torch.bfloat16
    else:
        torch_dtype = torch.float16
else:
    torch_dtype = torch.float32


# Load the model with the determined torch_dtype
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quantization_config,
    device_map="auto",
    torch_dtype=torch_dtype,
    trust_remote_code=True,  # In case the model requires it
    token=hf_token
)

# Log the dtype for verification
logger.info(f"Using torch_dtype: {torch_dtype}")

# Set the pad_token_id to the eos_token_id
model.config.pad_token_id = tokenizer.eos_token_id

device = next(model.parameters()).device
if device.type == "cuda":
    logger.info(f"Model loaded on GPU: {torch.cuda.get_device_name(device.index)}")
else:
    logger.info("Model loaded on CPU")

# Helper functions
def adjust_logits_processor(logit_bias):
    logits_processor = LogitsProcessorList()
    if logit_bias:
        logits_processor.append(LogitBiasProcessor(logit_bias))
    return logits_processor

class LogitBiasProcessor(LogitsProcessor):
    def __init__(self, logit_bias):
        self.logit_bias = {
            int(k): float(v) for k, v in logit_bias.items()
        }  # Convert keys to integers

    def __call__(self, input_ids, scores):
        vocab_size = scores.shape[-1]
        for token_id, bias in self.logit_bias.items():
            if 0 <= token_id < vocab_size:
                scores[:, token_id] += bias
        return scores

def create_stopping_criteria(stop_sequences):
    if not stop_sequences:
        return None
    stop_token_ids = [tokenizer.encode(seq, add_special_tokens=False) for seq in stop_sequences]
    return StoppingCriteriaList([StopOnTokens(stop_token_ids)])

class StopOnTokens(StoppingCriteria):
    def __init__(self, stop_ids_list):
        self.stop_ids_list = stop_ids_list

    def __call__(self, input_ids, scores, **kwargs):
        for stop_ids in self.stop_ids_list:
            if len(stop_ids) > input_ids.shape[1]:
                continue
            if torch.equal(input_ids[0, -len(stop_ids):], torch.tensor(stop_ids, device=input_ids.device)):
                return True
        return False

@app.post("/v1/chat/completions")
async def create_chat_completion(request: Request):
    try:
        data = await request.json()

        # Parse the request data
        chat_request = ChatCompletionRequest(**data)

        # Build the prompt
        prompt = ""
        for message in chat_request.messages:
            content = message.content
            if isinstance(content, list):
                content = ''.join(item.get('text', '') for item in content)
            if not isinstance(content, str):
                content = str(content)
            prompt += f"{content}\n"

        # Log input prompt
        logger.info(f"Input Prompt: {prompt}")

        # Tokenize input
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        input_ids = inputs["input_ids"]

        # Set do_sample based on temperature
        do_sample = chat_request.temperature > 0.0


        # Adjust logits processors
        logits_processor = adjust_logits_processor(chat_request.logit_bias)

        # Set up stopping criteria
        if chat_request.stop:
            if isinstance(chat_request.stop, str):
                stop_sequences = [chat_request.stop]
            elif isinstance(chat_request.stop, list):
                stop_sequences = chat_request.stop
            else:
                stop_sequences = []
            stopping_criteria = create_stopping_criteria(stop_sequences)
        else:
            stopping_criteria = None

        # Generate outputs
        try:
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens = chat_request.max_tokens or 1024,
                    temperature=chat_request.temperature,
                    top_p=chat_request.top_p,
                    do_sample=do_sample,
                    num_return_sequences=chat_request.n,
                    repetition_penalty=1.0 + chat_request.frequency_penalty,
                    logits_processor=logits_processor,
                    stopping_criteria=stopping_criteria,
                )
        except torch.cuda.OutOfMemoryError:
            logger.error("CUDA out of memory error during model generation.")
            return JSONResponse(status_code=500, content={"error": "CUDA out of memory"})
        except Exception as e:
            logger.error(f"Error during model generation: {str(e)}")
            return JSONResponse(status_code=500, content={"error": "Error during model generation"})

        # Process outputs
        responses = []
        for i in range(chat_request.n):
            output_ids = outputs[i]
            generated_text = tokenizer.decode(output_ids, skip_special_tokens=True)
            assistant_response = generated_text[len(prompt):].strip()
            responses.append(assistant_response)

        # Log output responses
        for response_text in responses:
            logger.info(f"Generated Response: {response_text}")

        request_id = str(uuid.uuid4())

        if chat_request.stream:
            def generate():
                for i in range(chat_request.n):
                    output_ids = outputs[i]
                    generated_tokens = output_ids[len(input_ids[0]):]  # Only new tokens
                    
                    for token_id in generated_tokens:
                        token = token_id.unsqueeze(0).unsqueeze(0)
                        text = tokenizer.decode(token, skip_special_tokens=True)
                        response = {
                            "id": request_id,
                            "object": "chat.completion.chunk",
                            "created": int(time.time()),
                            "model": chat_request.model,
                            "choices": [
                                {
                                    "index": i,
                                    "delta": {"content": text},
                                    "finish_reason": None,
                                }
                            ],
                        }
                        yield f"data: {json.dumps(response)}\n\n"
                # Send the final [DONE] message
                yield "data: [DONE]\n\n"

            return StreamingResponse(
                generate(),
                media_type="text/event-stream",
            )
        else:
            choices = []
            for i, assistant_response in enumerate(responses):
                choice = {
                    "index": i,
                    "message": {"role": "assistant", "content": assistant_response},
                    "finish_reason": "stop",
                }
                choices.append(choice)
            response = {
                "id": request_id,
                "object": "chat.completion",
                "created": int(time.time()),
                "model": chat_request.model,
                "choices": choices,
                "usage": {
                    "prompt_tokens": input_ids.shape[1],
                    "completion_tokens": sum(
                        [len(outputs[i]) - input_ids.shape[1] for i in range(chat_request.n)]
                    ),
                    "total_tokens": sum([len(outputs[i]) for i in range(chat_request.n)]),
                },
            }
            return JSONResponse(content=response)

    except Exception as e:
        logger.error(f"Error: {str(e)}")
        # Return error in OpenAI's error format
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "message": "An error occurred during processing.",
                    "type": "internal_server_error",
                    "param": None,
                    "code": None,
                }
            },
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

