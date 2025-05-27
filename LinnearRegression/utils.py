"""
Linear Regression Utilities

This module provides utility functions for creating synthetic datasets and interactive
visualizations for linear regression analysis.
"""

import numpy as np
import pandas as pd
import plotly.graph_objs as go
from sklearn.metrics import mean_squared_error, r2_score
import ipywidgets as widgets
from IPython.display import display
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

def create_simple_student_score_dataset(output_path="simple_student_scores.csv"):
    """
    Create a synthetic simple linear regression dataset with student study hours and exam scores.
    
    Args:
        output_path (str, optional): Path to save the CSV file. Defaults to "simple_student_scores.csv".
    
    Returns:
        pandas.DataFrame: The student scores dataset with Study_Hours and Exam_Score columns
    
    Example:
        >>> df = create_simple_student_score_dataset()
        >>> print(df.shape)
        (100, 2)
    """
    np.random.seed(42)
    n_students = 100
    
    # Generate realistic student data
    study_hours = np.random.uniform(1, 8, n_students)
    # Scores are roughly: 10 * hours + some noise + base score
    scores = 10 * study_hours + np.random.normal(0, 10, n_students) + 2
    scores = np.clip(scores, 0, 100)  # Keep scores between 0-100
    
    # Create DataFrame
    df = pd.DataFrame({
        'Study_Hours': study_hours,
        'Exam_Score': scores
    })
    
    # Save to CSV
    try:
        df.to_csv(output_path, index=False)
        print(f"Simple student scores dataset saved successfully to {output_path}!")
    except Exception as e:
        print(f"Warning: Could not save dataset to {output_path}: {str(e)}")
    
    return df

def create_student_score_dataset(output_path="student_scores.csv"):
    """
    Create a synthetic multiple linear regression dataset with student metrics and exam scores.
    
    Args:
        output_path (str, optional): Path to save the CSV file. Defaults to "student_scores.csv".
    
    Returns:
        pandas.DataFrame: The multiple regression dataset with Study_Hours, Sleep_Hours, 
                          Previous_Score, Attendance_Percent, and Exam_Score columns
    
    Example:
        >>> df = create_student_score_dataset()
        >>> print(df.shape)
        (200, 5)
    """
    np.random.seed(42)
    n_students = 200
    
    # Generate multiple features
    study_hours = np.random.uniform(1, 10, n_students)
    sleep_hours = np.random.uniform(4, 10, n_students)
    previous_score = np.random.uniform(40, 95, n_students)
    attendance = np.random.uniform(60, 100, n_students)
    
    # Create target variable with multiple influences
    exam_score = (
        8 * study_hours +           # Study hours have strong positive effect
        2 * sleep_hours +           # Sleep has moderate positive effect  
        0.3 * previous_score +      # Previous performance matters
        0.2 * attendance +          # Attendance has small positive effect
        np.random.normal(0, 5, n_students) + 10  # Noise + base score
    )
    
    exam_score = np.clip(exam_score, 0, 100)  # Keep scores between 0-100
    
    # Create DataFrame
    df = pd.DataFrame({
        'Study_Hours': study_hours,
        'Sleep_Hours': sleep_hours,
        'Previous_Score': previous_score,
        'Attendance_Percent': attendance,
        'Exam_Score': exam_score
    })
    
    # Save to CSV
    try:
        df.to_csv(output_path, index=False)
        print(f"Student scores dataset saved successfully to {output_path}!")
    except Exception as e:
        print(f"Warning: Could not save dataset to {output_path}: {str(e)}")
    
    return df

def create_simple_house_price_dataset(output_path="simple_house_price_dataset.csv"):
    """
    Create a synthetic house price dataset for practice exercises with basic features.
    
    Args:
        output_path (str, optional): Path to save the CSV file. 
                                     Defaults to "simple_house_price_dataset.csv".
    
    Returns:
        pandas.DataFrame: The house price dataset with Size_SqFt, Location_Score, 
                          Age_Years, and Price_USD columns
    
    Example:
        >>> df = create_simple_house_price_dataset()
        >>> print(df.shape)
        (150, 4)
    """
    np.random.seed(123)
    n_houses = 150
    
    # Features
    house_size = np.random.uniform(800, 3000, n_houses)
    location_score = np.random.uniform(1, 10, n_houses)
    age = np.random.uniform(0, 50, n_houses)
    
    # Price calculation
    price = (
        150 * house_size +
        20000 * location_score +
        -2000 * age +
        np.random.normal(0, 30000, n_houses) + 50000
    )
    price = np.maximum(price, 50000)
    
    # Create DataFrame
    df = pd.DataFrame({
        'Size_SqFt': house_size,
        'Location_Score': location_score,
        'Age_Years': age,
        'Price_USD': price
    })
    
    # Save to CSV
    try:
        df.to_csv(output_path, index=False)
        print(f"Simple house price dataset saved successfully to {output_path}!")
    except Exception as e:
        print(f"Warning: Could not save dataset to {output_path}: {str(e)}")
    
    return df

def create_house_price_dataset(output_path="house_price_dataset.csv"):
    """
    Create a synthetic house price dataset with complex, nonlinear relationships
    that make real-world sense.
    
    Args:
        output_path (str, optional): Path to save the CSV file. 
                                     Defaults to "house_price_dataset.csv".
    
    Returns:
        pandas.DataFrame: The house price dataset with Size_SqFt, Location_Score, Age_Years,
                          Ceiling_Height_Ft, Garage_Size_Cars, Distance_to_Metro_km, 
                          and Price_USD columns
    
    Example:
        >>> df = create_house_price_dataset()
        >>> print(df.shape)
        (150, 7)
    """
    np.random.seed(123)
    n_houses = 150

    # Base features
    house_size = np.random.uniform(800, 3000, n_houses)
    location_score = np.random.uniform(1, 10, n_houses)
    age = np.random.uniform(0, 50, n_houses)
    
    # New realistic features
    ceiling_height = np.random.uniform(8, 15, n_houses)              # in feet
    garage_size = np.random.uniform(0, 3, n_houses)                  # car capacity
    distance_to_metro = np.random.uniform(0.2, 20, n_houses)        # in km

    # Price with mixed relationships
    price = (
        150 * house_size +                          # linear
        20000 * location_score +                    # linear
        -2000 * age +                               
        3000 * (ceiling_height ** 2) +              # squared
        1000 * (garage_size ** 3) +                 # cubic
        -25000 * np.log(distance_to_metro) +        # log
        np.random.normal(0, 30000, n_houses) + 50000
    )
    price = np.maximum(price, 50000)

    # Create DataFrame
    df = pd.DataFrame({
        'Size_SqFt': house_size,
        'Location_Score': location_score,
        'Age_Years': age,
        'Ceiling_Height_Ft': ceiling_height,
        'Garage_Size_Cars': garage_size,
        'Distance_to_Metro_km': distance_to_metro,
        'Price_USD': price
    })

    # Save to CSV
    try:
        df.to_csv(output_path, index=False)
        print(f"House price dataset saved successfully to {output_path}!")
    except Exception as e:
        print(f"Warning: Could not save dataset to {output_path}: {str(e)}")
    
    return df

def create_interactive_regression_demo(df):
    """
    Create an interactive linear regression demo with sliders for slope and intercept.
    
    Args:
        df (pandas.DataFrame): DataFrame with 'Study_Hours' and 'Exam_Score' columns
        
    Returns:
        tuple: (slope_slider, intercept_slider) - The slider widgets for slope and intercept
        
    Raises:
        ValueError: If the DataFrame doesn't contain the required columns
        
    Example:
        >>> df = create_simple_student_score_dataset()
        >>> slope_slider, intercept_slider = create_interactive_regression_demo(df)
    """
    # Validate input DataFrame
    required_columns = ['Study_Hours', 'Exam_Score']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"DataFrame is missing required columns: {missing_columns}")
    
    X_demo = df['Study_Hours'].values
    y_demo = df['Exam_Score'].values
    
    # Estimate best-fit line
    best_fit = np.polyfit(X_demo, y_demo, 1)
    best_slope = best_fit[0]
    best_intercept = best_fit[1]
    slope_range = abs(best_slope) * 2
    intercept_range = abs(best_intercept) * 20
    
    # Calculate fixed axis ranges to maintain consistent graph size
    x_min = 0
    x_max = max(X_demo) * 1.1  # Add 10% margin
    y_min = 0
    y_max = max(y_demo) * 1.2  # Add 20% margin to accommodate different lines
    
    # Create a title widget
    title_widget = widgets.HTML(
        value='<h1 style="text-align: center; font-family: Arial, sans-serif; color: #2c3e50; margin-bottom: 20px;">📈 Interactive Linear Regression</h1>',
        layout=widgets.Layout(width='100%')
    )
    
    # Create output widgets
    output = widgets.Output()
    
    # Define consistent box dimensions
    metrics_width = '450px'
    slider_width = '450px'
    box_height = '220px'
    
    # Create metrics box with fixed width but flexible height
    metrics_box = widgets.HTML(
        value="",
        placeholder="MSE and R² will appear here.",
        layout=widgets.Layout(
            width=metrics_width,
            height='180px',
        )
    )
    
    # Create separate sliders for slope and intercept
    slope_slider = widgets.FloatSlider(
        value=best_slope,
        min=best_slope - slope_range/2,
        max=best_slope + slope_range/2,
        step=0.1,
        description='',
        layout=widgets.Layout(width='300px', margin='0px 10px'),
        continuous_update=False
    )
    
    intercept_slider = widgets.FloatSlider(
        value=best_intercept,
        min=best_intercept - intercept_range/2,
        max=best_intercept + intercept_range/2,
        step=0.1,
        description='',
        layout=widgets.Layout(width='300px', margin='0px 10px'),
        continuous_update=False
    )
    
    # Create custom labels for the sliders
    slope_label = widgets.HTML(
        value='<div style="font-family: Arial, sans-serif; font-size: 16px; font-weight: bold; padding-top: 2px;">Slope (m):</div>',
        layout=widgets.Layout(width='100px')
    )
    
    intercept_label = widgets.HTML(
        value='<div style="font-family: Arial, sans-serif; font-size: 16px; font-weight: bold; padding-top: 2px;">Intercept (b):</div>',
        layout=widgets.Layout(width='100px')
    )
    
    # Add a title for the sliders box
    slider_title = widgets.HTML(
        value='<h2 style="font-family: Arial, sans-serif; color: #2c3e50; margin-top: 0; margin-bottom: 15px; font-size: 22px; text-align: center;">Adjust Parameters</h2>'
    )
    
    # Function to update the plot
    def update_plot(change):
        # Clear previous output
        output.clear_output(wait=True)
        
        # Get current values from sliders
        slope = slope_slider.value
        intercept = intercept_slider.value
        
        # Predicted line
        x_line = np.linspace(x_min, x_max, 100)
        y_line = slope * x_line + intercept
    
        # Predictions and error calculation
        y_pred = slope * X_demo + intercept
        mse = mean_squared_error(y_demo, y_pred)
        r2 = r2_score(y_demo, y_pred)
    
        # Update metrics display with improved typography and bold values
        metrics_box.value = f"""
        <div style="font-family: Arial, sans-serif; line-height: 1.5; font-size: 16px; background-color: rgba(255, 255, 255, 0.9); padding: 15px; border-radius: 5px; border: 1px solid #cccccc; box-shadow: 2px 2px 5px rgba(0, 0, 0, 0.1); height: 100%;">
            <h2 style="color: #2c3e50; margin-top: 0; margin-bottom: 10px; font-size: 22px; text-align: center;">📊 Model Performance</h2>
            <div style="margin-left: 10px;">
                <div style="font-size: 16px; margin-bottom: 5px;"><strong>Slope (m):</strong> <strong>{slope:.2f}</strong></div>
                <div style="font-size: 16px; margin-bottom: 5px;"><strong>Intercept (b):</strong> <strong>{intercept:.2f}</strong></div>
                <div style="font-size: 16px; margin-bottom: 5px; margin-top: 10px;"><span style="color:#e67e22;"><strong>MSE:</strong></span> <strong>{mse:.2f}</strong></div>
                <div style="font-size: 16px;"><span style="color:#27ae60;"><strong>R²:</strong></span> <strong>{r2:.3f}</strong></div>
            </div>
        </div>
        """
    
        # Plotly elements
        trace_data = go.Scatter(
            x=X_demo, y=y_demo, mode='markers',
            name='Data Points',
            marker=dict(size=15, color="#e9a001")
        )
    
        trace_line = go.Scatter(
            x=x_line, y=y_line, mode='lines',
            name=f'Fit Line: y = {slope:.2f}x + {intercept:.2f}',
            line=dict(color='#2ba77b', width=3)
        )
    
        layout = go.Layout(
            xaxis=dict(
                title=dict(text='Hours Studied', font=dict(family="Arial, sans-serif", size=20)),
                range=[x_min, x_max],
                constrain='domain',
                tickfont=dict(family="Arial, sans-serif", size=20)
            ),
            yaxis=dict(
                title=dict(text='Score', font=dict(family="Arial, sans-serif", size=20)),
                range=[y_min, y_max],
                constrain='domain',
                tickfont=dict(family="Arial, sans-serif", size=20)
            ),
            height=550,
            width=900,
            margin=dict(t=40, l=80, r=80, b=60),
            showlegend=True,
            legend=dict(
                font=dict(family="Arial, sans-serif", size=20),
                bgcolor='rgba(255,255,255,0.8)'
            ),
            plot_bgcolor='#f8f9fa',
            paper_bgcolor='white'
        )
    
        with output:
            fig = go.Figure(data=[trace_data, trace_line], layout=layout)
            fig.show()
    
    # Connect the sliders to the update function
    slope_slider.observe(update_plot, names='value')
    intercept_slider.observe(update_plot, names='value')
    
    # Create slider groups with labels on the same line
    slope_group = widgets.HBox([slope_label, slope_slider], 
                              layout=widgets.Layout(margin='10px 0px', align_items='center'))
    intercept_group = widgets.HBox([intercept_label, intercept_slider], 
                                  layout=widgets.Layout(margin='10px 0px', align_items='center'))
    
    # Create a styled box for the sliders
    slider_box = widgets.VBox(
        [slider_title, slope_group, intercept_group],
        layout=widgets.Layout(
            border='1px solid #cccccc',
            padding='15px',
            margin='0px 0px 0px 10px',
            border_radius='5px',
            background_color='rgba(255, 255, 255, 0.9)',
            box_shadow='2px 2px 5px rgba(0, 0, 0, 0.1)',
            width=slider_width,
            height=box_height,
            overflow='hidden'
        )
    )
    
    # Add more space between the boxes and the plot
    spacer = widgets.HTML(
        value='<div style="height: 30px;"></div>'
    )
    
    # Create the UI with title at the top and swapped positions
    ui = widgets.VBox([
        title_widget,
        widgets.HBox([metrics_box, slider_box]),
        spacer,
        output
    ])
    
    # Display the UI and initial plot
    display(ui)
    update_plot(None)  # Initial plot
    
    return slope_slider, intercept_slider
