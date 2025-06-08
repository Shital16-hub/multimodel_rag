import os
from pathlib import Path
from typing import Dict, Any
import torch

class Config:
    # Base paths
    BASE_DIR = Path(__file__).parent
    DOCUMENTS_DIR = BASE_DIR / "documents"
    STATIC_DIR = BASE_DIR / "static"
    CACHE_DIR = BASE_DIR / ".cache"
    
    # vLLM Configuration (Core Innovation)
    VLLM_HOST = "localhost"
    VLLM_PORT = 8000
    VLLM_MODEL = "llava-hf/llava-v1.6-mistral-7b-hf"
    VLLM_MAX_TOKENS = 2048
    VLLM_TEMPERATURE = 0.1
    
    # Supported multimodal models
    MODELS = {
        "llava_next": "llava-hf/llava-v1.6-mistral-7b-hf",
        "qwen2_vl": "Qwen/Qwen2-VL-7B-Instruct", 
        "pixtral": "mistralai/Pixtral-12B-2409",
        "llava_onevision": "llava-hf/llava-onevision-qwen2-7b-ov-hf"
    }
    
    # Retrieval settings
    SIMILARITY_TOP_K = 5
    IMAGE_TOP_K = 3
    
    # Embeddings
    TEXT_EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"
    IMAGE_EMBEDDING_MODEL = "clip-ViT-B-32"
    
    # Vector store
    CHROMA_PERSIST_DIR = str(BASE_DIR / "chroma_db")
    
    # Performance settings
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    USE_FP16 = torch.cuda.is_available()
    MAX_CONCURRENT_REQUESTS = 10
    
    # Image processing
    MAX_IMAGE_SIZE = (1024, 1024)
    IMAGE_QUALITY = 85
    PDF_DPI = 150  # For PDF to image conversion
    
    # File upload settings (missing from your original config)
    MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
    ALLOWED_EXTENSIONS = ['.pdf']
    
    def __init__(self):  # Fixed: was def **init**(self):
        # Create directories
        for directory in [self.DOCUMENTS_DIR, self.STATIC_DIR, self.CACHE_DIR]:
            directory.mkdir(exist_ok=True)
        
        # Create static subdirs
        (self.STATIC_DIR / "images").mkdir(exist_ok=True)
        (self.STATIC_DIR / "temp").mkdir(exist_ok=True)
        
        print(f"📁 Created directories:")
        print(f"   Documents: {self.DOCUMENTS_DIR}")
        print(f"   Static: {self.STATIC_DIR}")
        print(f"   Cache: {self.CACHE_DIR}")

# Create the global config instance
config = Config()