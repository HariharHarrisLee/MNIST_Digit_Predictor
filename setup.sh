#!/bin/bash

# MNIST Digit Predictor Setup Script

echo "Setting up MNIST Digit Predictor..."

# Check if running in Heroku environment
if [ "$PORT" ]; then
    echo "Heroku deployment detected..."
    
    # Create Streamlit config for Heroku
    mkdir -p ~/.streamlit/
    
    echo "\
[general]\n\
email = \"mnist-app@example.com\"\n\
" > ~/.streamlit/credentials.toml
    
    echo "\
[server]\n\
headless = true\n\
enableCORS = false\n\
port = $PORT\n\
" > ~/.streamlit/config.toml
    
else
    # Local development setup
    
    # Create virtual environment if it doesn't exist
    if [ ! -d "venv" ]; then
        echo "Creating virtual environment..."
        python3.9 -m venv venv
    fi
    
    # Activate virtual environment
    echo "Activating virtual environment..."
    source venv/bin/activate
    
    # Install dependencies
    echo "Installing dependencies..."
    pip install -r requirements.txt
    
    echo "Setup complete! To run the app:"
    echo "1. Activate virtual environment: source venv/bin/activate"
    echo "2. Run Streamlit app: streamlit run app.py"
fi