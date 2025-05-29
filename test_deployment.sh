#!/bin/bash

# 🚀 MNIST Digit Predictor - One-Click Deployment Test
# This script tests all deployment configurations

echo "🔢 MNIST Digit Predictor - Deployment Test"
echo "=========================================="

# Test 1: Dependencies
echo "✅ Testing dependencies..."
python -c "
import sys
sys.path.append('streamlit_app')
try:
    from streamlit_app import *
    print('✅ All imports successful!')
except Exception as e:
    print(f'❌ Import error: {e}')
    exit(1)
"

# Test 2: Core functionality
echo "✅ Testing core functionality..."
python -c "
import sys
sys.path.append('streamlit_app')
from streamlit_app import load_data, create_mlp_model, create_cnn_model
try:
    print('  📊 Loading data...')
    data = load_data()
    print('  🧠 Creating MLP model...')
    mlp = create_mlp_model()
    print('  🖼️ Creating CNN model...')
    cnn = create_cnn_model()
    print('✅ Core functionality working!')
except Exception as e:
    print(f'❌ Functionality error: {e}')
    exit(1)
"

# Test 3: App entry point
echo "✅ Testing app entry point..."
python -c "
import sys, os
sys.path.append('streamlit_app')
try:
    from app import *
    print('✅ App entry point working!')
except Exception as e:
    print(f'❌ Entry point error: {e}')
    exit(1)
"

echo ""
echo "🎉 All tests passed! Ready for deployment:"
echo ""
echo "📱 Local deployment:"
echo "   streamlit run app.py"
echo ""
echo "🐳 Docker deployment:"
echo "   docker-compose up --build"
echo ""
echo "🌐 Streamlit Cloud:"
echo "   1. Push to GitHub"
echo "   2. Deploy on share.streamlit.io"
echo "   3. Set main file: app.py"
echo ""
echo "📚 For detailed instructions, see:"
echo "   - QUICK_DEPLOY.md (Quick start)"
echo "   - DEPLOYMENT.md (Comprehensive guide)"
echo ""