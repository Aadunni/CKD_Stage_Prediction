#!/bin/bash
# Setup script for CKD API with LLM Assessment

echo "🏥 CKD Prediction API with LLM Medical Assessment Setup"
echo "=================================================="

# Check if Gemini API key is set
if [ -z "$GEMINI_API_KEY" ]; then
    echo "❌ GEMINI_API_KEY environment variable not set"
    echo "📝 To enable LLM medical assessment, set your Gemini API key:"
    echo "   export GEMINI_API_KEY='your-gemini-api-key-here'"
    echo ""
    echo "🔑 Get your API key from: https://makersuite.google.com/app/apikey"
    echo ""
else
    echo "✅ GEMINI_API_KEY is configured"
fi

# Install additional dependencies for LLM features
echo "📦 Installing additional dependencies for LLM features..."
pip install aiohttp

echo ""
echo "🚀 Starting CKD API with LLM Assessment..."
echo "🌐 API Documentation: http://localhost:8000/docs"
echo "📊 Sample Data: http://localhost:8000/sample-data"
echo "🤖 LLM Assessment: Available if GEMINI_API_KEY is set"
echo ""

# Start the API
uvicorn ckd_api:app --reload --host 0.0.0.0 --port 8000