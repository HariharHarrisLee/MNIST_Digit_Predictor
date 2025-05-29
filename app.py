# Main entry point for Streamlit Cloud deployment
# This file imports and runs the main application

import sys
import os

# Add the streamlit_app directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'streamlit_app'))

# Import and run the main application
from streamlit_app import main

if __name__ == "__main__":
    main()