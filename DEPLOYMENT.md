# MNIST Digit Predictor - Deployment Guide

This guide covers multiple deployment options for the MNIST Digit Predictor application.

## 🌐 Streamlit Cloud Deployment (Recommended)

### Prerequisites
- GitHub account
- Streamlit Cloud account (free at share.streamlit.io)

### Step-by-Step Instructions

1. **Prepare Repository**
   ```bash
   # Initialize git repository (if not already done)
   git init
   git add .
   git commit -m "Initial commit: MNIST Digit Predictor"
   ```

2. **Push to GitHub**
   ```bash
   # Create repository on GitHub first, then:
   git remote add origin https://github.com/YOUR_USERNAME/mnist-digit-predictor.git
   git branch -M main
   git push -u origin main
   ```

3. **Deploy on Streamlit Cloud**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Click "New app"
   - Connect your GitHub repository
   - Set the main file path: `app.py`
   - Click "Deploy"

4. **Configuration**
   - The app will automatically install dependencies from `requirements.txt`
   - Initial deployment may take 5-10 minutes
   - Models will be downloaded and trained on first use

### Important Notes for Streamlit Cloud

- **Memory Limits**: Streamlit Cloud has memory constraints. The app is optimized to handle this.
- **Model Storage**: Models are saved in the session and will persist during the session.
- **Data Loading**: MNIST dataset is automatically downloaded on first run.
- **Performance**: First model training may take longer due to resource constraints.

## 🐳 Docker Deployment

### Dockerfile
```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy application files
COPY . .

# Expose port
EXPOSE 8501

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Run the application
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### Build and Run
```bash
# Build Docker image
docker build -t mnist-predictor .

# Run container
docker run -p 8501:8501 mnist-predictor
```

## ☁️ Heroku Deployment

### Required Files

1. **Procfile**
   ```
   web: sh setup.sh && streamlit run app.py --server.port=$PORT --server.address=0.0.0.0
   ```

2. **setup.sh**
   ```bash
   mkdir -p ~/.streamlit/
   echo "\
   [general]\n\
   email = \"your-email@domain.com\"\n\
   " > ~/.streamlit/credentials.toml
   echo "\
   [server]\n\
   headless = true\n\
   enableCORS=false\n\
   port = $PORT\n\
   " > ~/.streamlit/config.toml
   ```

3. **runtime.txt**
   ```
   python-3.9.16
   ```

### Deployment Steps
```bash
# Install Heroku CLI and login
heroku login

# Create Heroku app
heroku create your-mnist-app

# Deploy
git push heroku main
```

## 🖥️ Local Production Deployment

### Using Gunicorn (Not applicable for Streamlit)
Streamlit has its own server, so use the built-in server:

```bash
# Production-like local deployment
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
```

### Using Reverse Proxy (Nginx)
```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://localhost:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
    }
}
```

## 📊 Performance Optimization for Deployment

### Memory Management
- Models are cached using `@st.cache_resource`
- Data loading is cached using `@st.cache_data`
- Efficient batch processing for predictions

### Resource Limits
- Training epochs can be reduced for cloud deployment
- Batch sizes are optimized for memory constraints
- Models are saved/loaded efficiently

### Monitoring
- Built-in error handling and user feedback
- Performance metrics tracking
- Resource usage optimization

## 🔧 Environment Variables

For production deployment, consider these environment variables:

```bash
# Optional: Control training parameters
MNIST_EPOCHS=5
MNIST_BATCH_SIZE=128
MNIST_CACHE_DIR=/tmp/mnist_cache

# Optional: Model storage
MODEL_STORAGE_PATH=/app/models
```

## 🚀 CI/CD Pipeline (GitHub Actions)

Create `.github/workflows/deploy.yml`:

```yaml
name: Deploy to Streamlit Cloud

on:
  push:
    branches: [ main ]

jobs:
  deploy:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v2
    
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: 3.9
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
    
    - name: Test application
      run: |
        python -c "from streamlit_app.streamlit_app import *; print('Import test passed')"
    
    - name: Deploy notification
      run: echo "Application tested successfully"
```

## 📝 Deployment Checklist

### Pre-Deployment
- [ ] All dependencies in requirements.txt
- [ ] Code tested locally
- [ ] Memory usage optimized
- [ ] Error handling implemented
- [ ] Documentation updated

### Post-Deployment
- [ ] Application loads successfully
- [ ] All tabs functional
- [ ] Model training works
- [ ] Predictions working
- [ ] Performance acceptable

## 🐛 Troubleshooting

### Common Issues

1. **Memory Errors**
   - Reduce training epochs
   - Use smaller batch sizes
   - Clear cache between runs

2. **Slow Loading**
   - Models need to download MNIST on first run
   - Training takes time on first use
   - Subsequent runs are faster due to caching

3. **Import Errors**
   - Ensure all dependencies in requirements.txt
   - Check Python version compatibility
   - Verify file paths are correct

### Performance Tips

1. **First Run Optimization**
   - Initial MNIST download takes time
   - First model training is slower
   - Pre-trained models can be included in repo

2. **User Experience**
   - Clear loading indicators
   - Progress bars for training
   - Error messages for guidance

## 📈 Scaling Considerations

### For High Traffic
- Consider using model serving frameworks (TensorFlow Serving)
- Implement model caching strategies
- Use load balancers for multiple instances

### For Enterprise
- Container orchestration (Kubernetes)
- Model versioning and management
- Monitoring and logging solutions

---

Choose the deployment method that best fits your needs:
- **Streamlit Cloud**: Easiest, free tier available
- **Docker**: Most flexible, works anywhere
- **Heroku**: Good for small to medium apps
- **Local**: Best for development and testing