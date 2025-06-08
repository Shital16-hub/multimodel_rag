#!/bin/bash

echo "🚀 Starting Simplified Multimodal RAG System..."

# Set environment
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Create directories
mkdir -p documents static/images static/temp .cache chroma_db

echo "✨ Innovations:"
echo "   🔗 LangChain MultiVector Retriever"
echo "   🚀 vLLM V1 Multimodal Generation"
echo "   🎯 Integrated Text-Image Responses"
echo "   ⚡ Simple, Reliable Architecture"
echo ""

# Start application
echo "🔥 Starting FastAPI application..."
python app.py