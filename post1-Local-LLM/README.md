# Local LLM API with Docker

This repository provides a local API endpoint for a Large Language Model (LLM) using Docker, enabling easy deployment and access to a model hosted locally. The setup leverages FastAPI for building the API and Docker Compose for container orchestration, including GPU acceleration support and optional ngrok tunneling.

## Directory Structure

The repository is organized as follows:

```plaintext
post1-Local-LLM/
├── app
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── main.py
├── docker-compose.yml
└── .env

```

## Prerequisites

- **Docker** and **Docker Compose** installed on your system.
- **NVIDIA Docker** (if using GPU support).
- Tokens for Hugging Face Hub and ngrok for API access and tunneling.

## Setup and Configuration

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Medium-Posts/post1-Local-LLM.git
   cd post1-Local-LLM
   ```

2. **Configure Environment Variables**:
   Create a `.env` file in the root directory with the following content:
   ```plaintext
   HUGGINGFACE_HUB_TOKEN=<Your Hugging Face Token>
   NGROK_AUTHTOKEN=<Your ngrok Auth Token>
   ```

3. **Install Dependencies**:
   Dependencies are listed in `app/requirements.txt` and include:
   - `fastapi`, `pydantic`, `torch`, `transformers`, `uvicorn`, etc.
   They will be automatically installed within the Docker container.

## Running the Application

1. **Build and Start the Docker Containers**:
   Use Docker Compose to build and run the application.
   ```bash
   docker-compose up --build
   ```
   This will:
   - Start the FastAPI server on port `8000`.
   - Expose the ngrok tunnel on port `4040` for external access.

2. **Access the API**:
   - Local access: `http://localhost:8000`
   - External access (via ngrok): Use the ngrok-provided URL (check ngrock dashboard for the exact address).

## API Endpoints

### Chat Completion

- **Endpoint**: `POST /v1/chat/completions`
- **Description**: Generates responses based on the provided chat messages.

#### Request Example
```
curl -X POST https://YOUR_NGROK_URL/v1/chat/completions \
     -H "Content-Type: application/json" \
     -d '{
           "messages": [
               {"role": "assistant", "content": "what is the capital of US? Be short and precise."}
           ],
           "max_tokens": 50,
           "temperature": 0.1
         }'

```

#### Response Example
```json
{
    "id": "53d6da8b-ba8e-4d74-be21-197ab7b24176",
    "object": "chat.completion",
    "created": 1730839239,
    "model": "meta-llama/Llama-3.2-3B-Instruct",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "The capital of the United States is Washington, D.C. (District of Columbia)."
            },
            "finish_reason": "stop"
        }
    ],
    "usage": {
        "prompt_tokens": 13,
        "completion_tokens": 18,
        "total_tokens": 31
    }
}
```

## Logging and Error Handling

The application logs incoming requests, responses, and errors. Logs are saved to `app.log` and rotate automatically when exceeding 5MB (up to 5 log files).

## GPU Support

The configuration enables GPU usage with NVIDIA Docker. Ensure your Docker environment has access to the GPU.

## Troubleshooting

- **CUDA Out of Memory**: If GPU memory is insufficient, adjust model configurations or use a smaller model.
- **Environment Variables Not Found**: Check if `.env` is correctly set up and included in `docker-compose.yml`.

## Contribution

Contributions are welcome! Please fork the repository, create a feature branch, and submit a pull request.

---

**Note**: This project requires valid Hugging Face and ngrok tokens for accessing models and external access.
