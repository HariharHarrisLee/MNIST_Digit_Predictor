# MNIST Handwritten Digit Predictor

A comprehensive machine learning application that compares MLP (Multi-Layer Perceptron) and CNN (Convolutional Neural Network) models for handwritten digit recognition using the MNIST dataset.

## 🎯 Features

- **Interactive EDA**: Explore the MNIST dataset with visualizations and statistics
- **Model Training & Comparison**: Train both MLP and CNN models and compare their performance
- **Real-time Predictions**: Upload images or test with random samples to see both models' predictions
- **Performance Metrics**: Detailed comparison including accuracy, F1 score, and latency measurements
- **Model Persistence**: Save and load trained models automatically

## 🏗️ Architecture

### MLP Model
```python
Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
])
```

### CNN Model
```python
Sequential([
    Conv2D(32, (3, 3), padding='same'),
    BatchNormalization(),
    Activation('relu'),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), padding='same'),
    BatchNormalization(),
    Activation('relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
])
```

## 📁 Project Structure

```
MNIST_Digit_Predictor/
├── data/                     # Dataset storage (auto-downloaded)
├── models/                   # Saved trained models
│   ├── mlp_baseline.h5      # MLP model
│   └── mnist_cnn.h5         # CNN model
├── notebooks/               # Jupyter notebooks (placeholder)
│   ├── EDA.ipynb
│   └── Modeling.ipynb
├── streamlit_app/           # Main application
│   └── streamlit_app.py
├── requirements.txt         # Dependencies
├── setup.sh                # Setup script
└── README.md               # This file
```

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- pip package manager

### Installation

1. **Clone and Navigate**
   ```bash
   cd MNIST_Digit_Predictor
   ```

2. **Run Setup Script**
   ```bash
   chmod +x setup.sh
   ./setup.sh
   ```

3. **Manual Installation (Alternative)**
   ```bash
   # Create virtual environment
   python3.9 -m venv venv
   source venv/bin/activate
   
   # Install dependencies
   pip install -r requirements.txt
   ```

### Running the Application

```bash
# Make sure virtual environment is activated
source venv/bin/activate

# Run Streamlit app
streamlit run streamlit_app/streamlit_app.py
```

The application will open in your browser at `http://localhost:8501`

## 🖥️ Usage Guide

### 1. Exploratory Data Analysis (EDA)
- View dataset statistics and distribution
- Explore sample images from each digit class
- Understand the data characteristics

### 2. Train & Compare Models
- Set training parameters (epochs, batch size)
- Train both MLP and CNN models simultaneously
- Compare performance metrics:
  - Test Accuracy (4 decimal places)
  - F1 Score (4 decimal places)
  - Average Latency (milliseconds)
  - Training Time

### 3. Make Predictions
- **Upload Images**: Upload your own handwritten digit images
- **Random Samples**: Test with random samples from the test set
- **Model Comparison**: See predictions from both models side-by-side
- **Performance Tracking**: Monitor prediction latency for each model

## 📊 Performance Metrics

The application tracks and compares:

- **Accuracy**: Model performance on test dataset
- **F1 Score**: Weighted average F1 score across all classes
- **Latency**: Per-prediction latency in milliseconds
- **Training Time**: Total time to train each model

## 🔧 Technical Details

### Dependencies
- **TensorFlow 2.12.0**: Deep learning framework
- **Streamlit 1.22.0**: Web application framework
- **NumPy 1.23.5**: Numerical computing
- **Pandas 2.0.0**: Data manipulation
- **Matplotlib 3.7.1**: Data visualization
- **Seaborn 0.12.2**: Statistical plotting
- **Scikit-learn 1.2.2**: ML utilities
- **Pillow 9.5.0**: Image processing

### Model Details
- **Input**: 28x28 grayscale images
- **Output**: 10 classes (digits 0-9)
- **Preprocessing**: Normalization to [0,1] range
- **Loss Function**: Categorical crossentropy
- **Optimizer**: Adam
- **Metrics**: Accuracy

### Image Processing
- Automatic resizing to 28x28 pixels
- Grayscale conversion
- Color inversion detection for MNIST compatibility
- Normalization to match training data

## 🎮 Interactive Features

- **Real-time Training**: Watch models train with live progress
- **Live Predictions**: Instant predictions with latency measurement
- **Visual Comparisons**: Side-by-side model performance
- **Random Testing**: Quick evaluation with test samples
- **Model Persistence**: Automatic saving and loading of trained models

## 🐛 Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   # Ensure all dependencies are installed
   pip install -r requirements.txt
   ```

2. **Memory Issues**
   - Reduce batch size in training parameters
   - Use fewer epochs for initial testing

3. **Model Loading Issues**
   - Retrain models using the "Train & Compare" tab
   - Check that models/ directory has write permissions

## 🚀 Deployment

### Local Deployment
The application runs locally using Streamlit's development server.

### Streamlit Cloud Deployment
1. Push code to GitHub repository
2. Connect repository to Streamlit Cloud
3. Deploy with automatic dependency installation

## 📈 Expected Results

Typical performance ranges:
- **MLP Accuracy**: 96-98%
- **CNN Accuracy**: 98-99%
- **MLP Latency**: 5-15ms per prediction
- **CNN Latency**: 10-30ms per prediction

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- MNIST dataset by Yann LeCun
- TensorFlow team for the deep learning framework
- Streamlit team for the web app framework