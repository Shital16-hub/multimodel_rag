import asyncio
import logging
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from typing import List, Dict, Any
from pathlib import Path
import json
import aiofiles
from asyncio_throttle import Throttler

from config import config
from multimodal_retriever import multimodal_retriever

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(
    title="Universal Multimodal RAG - Works with ANY PDF",
    description="Universal RAG system that processes ANY PDF content - documents, reports, research papers, books, manuals, etc.",
    version="2.0.0",
    docs_url="/docs"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files
app.mount("/static", StaticFiles(directory=str(config.STATIC_DIR)), name="static")

# Request throttling
throttler = Throttler(rate_limit=config.MAX_CONCURRENT_REQUESTS)

@app.on_event("startup")
async def startup_event():
    """Initialize system components"""
    try:
        logger.info("🚀 Starting Universal Multimodal RAG System...")
        await multimodal_retriever.initialize()
        logger.info("✅ System ready!")
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        raise

@app.get("/", response_class=HTMLResponse)
async def root():
    """Universal web interface - works with ANY document content"""
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Universal Multimodal RAG - Any PDF Content</title>
        <script src="https://cdn.tailwindcss.com"></script>
        <script>
            tailwind.config = {
                theme: {
                    extend: {
                        animation: {
                            'gradient': 'gradient 8s ease infinite',
                        }
                    }
                }
            }
        </script>
        <style>
            @keyframes gradient {
                0%, 100% { background-position: 0% 50%; }
                50% { background-position: 100% 50%; }
            }
            .gradient-bg {
                background: linear-gradient(-45deg, #667eea, #764ba2, #f093fb, #f5576c);
                background-size: 400% 400%;
                animation: gradient 8s ease infinite;
            }
            .response-content {
                line-height: 1.8;
            }
            .response-content img {
                margin: 20px auto;
                border-radius: 12px;
                box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
                max-width: 100%;
                height: auto;
            }
        </style>
    </head>
    <body class="gradient-bg min-h-screen">
        <div class="container mx-auto px-4 py-8">
            <!-- Header -->
            <div class="text-center mb-12">
                <h1 class="text-6xl font-bold text-white mb-4 drop-shadow-lg">
                    📚 Universal Multimodal RAG
                </h1>
                <p class="text-2xl text-white/90 mb-6 drop-shadow">
                    Upload ANY PDF • Ask ANY Question • Get Intelligent Answers
                </p>
                <div class="flex justify-center space-x-4 text-sm text-white/80">
                    <span class="bg-white/20 backdrop-blur px-4 py-2 rounded-full">📄 Any Document Type</span>
                    <span class="bg-white/20 backdrop-blur px-4 py-2 rounded-full">🔍 Intelligent Processing</span>
                    <span class="bg-white/20 backdrop-blur px-4 py-2 rounded-full">🖼️ Visual Integration</span>
                    <span class="bg-white/20 backdrop-blur px-4 py-2 rounded-full">⚡ Fast Responses</span>
                </div>
            </div>

            <!-- Features Section -->
            <div class="grid grid-cols-1 md:grid-cols-3 gap-6 mb-12">
                <div class="bg-white/10 backdrop-blur-xl rounded-2xl p-6 border border-white/20">
                    <div class="text-4xl mb-4">📊</div>
                    <h3 class="text-xl font-bold text-white mb-3">Reports & Analytics</h3>
                    <p class="text-white/70">Business reports, financial documents, research papers with charts and graphs</p>
                </div>
                <div class="bg-white/10 backdrop-blur-xl rounded-2xl p-6 border border-white/20">
                    <div class="text-4xl mb-4">📖</div>
                    <h3 class="text-xl font-bold text-white mb-3">Books & Manuals</h3>
                    <p class="text-white/70">Technical manuals, textbooks, documentation with diagrams and illustrations</p>
                </div>
                <div class="bg-white/10 backdrop-blur-xl rounded-2xl p-6 border border-white/20">
                    <div class="text-4xl mb-4">🔬</div>
                    <h3 class="text-xl font-bold text-white mb-3">Academic Papers</h3>
                    <p class="text-white/70">Research papers, studies, presentations with tables and scientific diagrams</p>
                </div>
            </div>

            <!-- Upload Section -->
            <div class="bg-white/15 backdrop-blur-xl rounded-3xl p-8 mb-8 border border-white/30 shadow-2xl">
                <h2 class="text-3xl font-bold text-white mb-6">📁 Upload Your Documents</h2>
                <div class="border-2 border-dashed border-white/50 rounded-2xl p-12 text-center hover:border-white/70 transition-all">
                    <input type="file" id="fileInput" multiple accept=".pdf" 
                           class="hidden" onchange="handleFileSelect()">
                    <label for="fileInput" class="cursor-pointer">
                        <div class="text-8xl text-white/80 mb-6">📄</div>
                        <p class="text-white text-xl mb-4">Click to upload ANY PDF documents</p>
                        <p class="text-white/70 text-lg">Business reports • Research papers • Technical manuals • Books • Any PDF content</p>
                    </label>
                </div>
                <button onclick="uploadFiles()" 
                        class="mt-8 w-full bg-gradient-to-r from-blue-500 to-purple-600 text-white py-5 px-8 rounded-2xl font-bold text-xl hover:from-blue-600 hover:to-purple-700 transition-all transform hover:scale-[1.02] shadow-xl">
                    🚀 Process Documents
                </button>
            </div>

            <!-- Query Section -->
            <div class="bg-white/15 backdrop-blur-xl rounded-3xl p-8 mb-8 border border-white/30 shadow-2xl">
                <h2 class="text-3xl font-bold text-white mb-6">💬 Ask Anything</h2>
                <div class="space-y-6">
                    <textarea id="queryInput" 
                              placeholder="Ask any question about your uploaded documents. I can analyze text, explain concepts, describe images, summarize content, and more..."
                              class="w-full h-36 bg-white/10 border border-white/30 rounded-2xl px-6 py-4 text-white text-lg placeholder-white/60 focus:outline-none focus:border-white/70 resize-none backdrop-blur"></textarea>
                    <div class="flex items-center justify-between">
                        <div class="flex items-center space-x-6">
                            <label class="flex items-center text-white text-lg">
                                <input type="checkbox" id="useImages" checked class="mr-3 scale-125">
                                Include Visual Content Analysis
                            </label>
                        </div>
                        <button onclick="submitQuery()" 
                                class="bg-gradient-to-r from-green-500 to-emerald-600 text-white py-4 px-10 rounded-2xl font-bold text-lg hover:from-green-600 hover:to-emerald-700 transition-all shadow-xl">
                            ✨ Get Answer
                        </button>
                    </div>
                </div>
                
                <!-- Example queries for different document types -->
                <div class="mt-6 text-white/70">
                    <p class="text-sm mb-3">💡 Example questions for different document types:</p>
                    <div class="grid grid-cols-1 md:grid-cols-2 gap-3 text-xs">
                        <button onclick="setQuery('Summarize the main findings of this document.')" 
                                class="text-left p-3 bg-white/10 rounded-xl hover:bg-white/20 transition-all">
                            "Summarize the main findings of this document."
                        </button>
                        <button onclick="setQuery('What are the key insights from the data shown?')" 
                                class="text-left p-3 bg-white/10 rounded-xl hover:bg-white/20 transition-all">
                            "What are the key insights from the data shown?"
                        </button>
                        <button onclick="setQuery('Explain the methodology described in this research.')" 
                                class="text-left p-3 bg-white/10 rounded-xl hover:bg-white/20 transition-all">
                            "Explain the methodology described in this research."
                        </button>
                        <button onclick="setQuery('What are the recommendations or conclusions?')" 
                                class="text-left p-3 bg-white/10 rounded-xl hover:bg-white/20 transition-all">
                            "What are the recommendations or conclusions?"
                        </button>
                    </div>
                </div>
            </div>

            <!-- Status -->
            <div id="status" class="mb-6"></div>

            <!-- Response -->
            <div id="response" class="hidden bg-white/15 backdrop-blur-xl rounded-3xl p-8 border border-white/30 shadow-2xl">
                <div id="responseContent"></div>
            </div>
        </div>

        <script>
            let selectedFiles = [];

            function setQuery(query) {
                document.getElementById('queryInput').value = query;
            }

            function handleFileSelect() {
                selectedFiles = Array.from(document.getElementById('fileInput').files);
                updateFileDisplay();
            }

            function updateFileDisplay() {
                const fileInput = document.getElementById('fileInput');
                const label = fileInput.nextElementSibling;
                if (selectedFiles.length > 0) {
                    label.innerHTML = `
                        <div class="text-6xl text-green-300 mb-6">✅</div>
                        <p class="text-white text-xl mb-4">${selectedFiles.length} file(s) selected</p>
                        <p class="text-green-200 text-lg">Ready for universal processing</p>
                    `;
                }
            }

            async function uploadFiles() {
                if (selectedFiles.length === 0) {
                    showStatus('Please select files to upload', 'error');
                    return;
                }

                showStatus('🔄 Processing your documents with universal content extraction...', 'loading');

                const formData = new FormData();
                selectedFiles.forEach(file => formData.append('files', file));

                try {
                    const response = await fetch('/upload', {
                        method: 'POST',
                        body: formData
                    });
                    const result = await response.json();

                    if (result.status === 'success') {
                        let contentInfo = '';
                        if (result.content_types_detected) {
                            const contentTypes = Object.entries(result.content_types_detected)
                                .map(([type, count]) => `${count} ${type.replace('_', ' ')}`)
                                .join(', ');
                            contentInfo = ` • Content detected: ${contentTypes}`;
                        }
                        showStatus(`✅ Success! Processed ${result.files_processed} files • ${result.text_documents} text sections • ${result.visual_elements} visual elements${contentInfo}`, 'success');
                    } else {
                        showStatus(`❌ ${result.message}`, 'error');
                    }
                } catch (error) {
                    showStatus(`❌ Upload failed: ${error}`, 'error');
                }
            }

            async function submitQuery() {
                const query = document.getElementById('queryInput').value;
                const useImages = document.getElementById('useImages').checked;

                if (!query.trim()) {
                    showStatus('Please enter a question', 'error');
                    return;
                }

                showStatus('🧠 Analyzing your documents and generating intelligent response...', 'loading');

                try {
                    const startTime = Date.now();
                    const response = await fetch('/query', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({
                            question: query,
                            use_images: useImages
                        })
                    });
                    
                    const result = await response.json();
                    const endTime = Date.now();
                    const clientLatency = (endTime - startTime) / 1000;

                    if (result.status === 'success') {
                        displayResponse(result, clientLatency);
                        document.getElementById('status').innerHTML = '';
                    } else {
                        showStatus(`❌ ${result.message}`, 'error');
                    }
                } catch (error) {
                    showStatus(`❌ Query failed: ${error}`, 'error');
                }
            }

            function displayResponse(result, clientLatency) {
                const responseDiv = document.getElementById('response');
                const content = document.getElementById('responseContent');
                
                responseDiv.classList.remove('hidden');
                
                // Process response which may contain inline HTML images
                let responseHtml = result.response.replace(/\n/g, '<br>');
                
                content.innerHTML = `
                    <div class="space-y-8">
                        <div class="flex items-center justify-between mb-8">
                            <h3 class="text-3xl font-bold text-white">🤖 Intelligent Response</h3>
                            <div class="flex space-x-3 text-sm">
                                <span class="bg-green-400/20 text-green-200 px-4 py-2 rounded-full font-semibold">
                                    ⚡ ${result.latency_seconds}s server + ${clientLatency.toFixed(2)}s client
                                </span>
                                <span class="bg-blue-400/20 text-blue-200 px-4 py-2 rounded-full font-semibold">
                                    🧠 ${result.model_used.split('/').pop() || 'vLLM'}
                                </span>
                            </div>
                        </div>
                        
                        <div class="bg-white/10 rounded-2xl p-8 border border-white/20 backdrop-blur">
                            <div class="text-white prose prose-lg prose-invert max-w-none leading-relaxed response-content">
                                ${responseHtml}
                            </div>
                        </div>
                        
                        <div class="grid grid-cols-1 md:grid-cols-4 gap-6 text-sm">
                            <div class="bg-purple-400/20 rounded-xl p-6 text-center">
                                <div class="text-purple-200 font-bold text-lg">📄 Text Sources</div>
                                <div class="text-white text-2xl font-bold">${result.text_sources}</div>
                            </div>
                            <div class="bg-blue-400/20 rounded-xl p-6 text-center">
                                <div class="text-blue-200 font-bold text-lg">🖼️ Visual Elements</div>
                                <div class="text-white text-2xl font-bold">${result.image_sources}</div>
                            </div>
                            <div class="bg-green-400/20 rounded-xl p-6 text-center">
                                <div class="text-green-200 font-bold text-lg">🔤 Tokens</div>
                                <div class="text-white text-2xl font-bold">${result.tokens_generated}</div>
                            </div>
                            <div class="bg-orange-400/20 rounded-xl p-6 text-center">
                                <div class="text-orange-200 font-bold text-lg">🎯 Integration</div>
                                <div class="text-white text-lg font-bold">${result.integration_type}</div>
                            </div>
                        </div>
                        
                        <div class="bg-white/5 rounded-2xl p-6 border border-white/10">
                            <h4 class="text-xl font-bold text-white mb-4">🚀 Universal Features:</h4>
                            <div class="flex flex-wrap gap-3">
                                ${result.innovation_used.map(innovation => `
                                    <span class="bg-indigo-400/20 text-indigo-200 px-4 py-2 rounded-full text-sm font-semibold">
                                        ${innovation}
                                    </span>
                                `).join('')}
                            </div>
                            <p class="text-white/70 mt-4">Method: ${result.method}</p>
                        </div>
                        
                        ${result.sources && result.sources.visual_types && result.sources.visual_types.length > 0 ? `
                        <div class="bg-white/5 rounded-2xl p-6 border border-white/10">
                            <h4 class="text-xl font-bold text-white mb-4">📊 Visual Content Types:</h4>
                            <div class="flex flex-wrap gap-3">
                                ${[...new Set(result.sources.visual_types)].map(type => `
                                    <span class="bg-purple-400/20 text-purple-200 px-4 py-2 rounded-full text-sm">
                                        ${type.replace('_', ' ').toUpperCase()}
                                    </span>
                                `).join('')}
                            </div>
                        </div>
                        ` : ''}
                    </div>
                `;
            }

            function showStatus(message, type) {
                const statusDiv = document.getElementById('status');
                const colors = {
                    'loading': 'bg-blue-400/20 text-blue-200 border-blue-400/30',
                    'success': 'bg-green-400/20 text-green-200 border-green-400/30',
                    'error': 'bg-red-400/20 text-red-200 border-red-400/30'
                };
                
                statusDiv.innerHTML = `
                    <div class="p-6 rounded-2xl border ${colors[type]} backdrop-blur-xl text-lg font-semibold">
                        ${message}
                    </div>
                `;
            }

            // Auto-focus and keyboard shortcuts
            document.addEventListener('DOMContentLoaded', function() {
                document.getElementById('queryInput').focus();
            });

            document.getElementById('queryInput').addEventListener('keypress', function(e) {
                if (e.key === 'Enter' && e.ctrlKey) {
                    submitQuery();
                }
            });
        </script>
    </body>
    </html>
    """

@app.post("/upload")
async def upload_files(files: List[UploadFile] = File(...)):
    """Upload and process documents - FIXED VERSION using your working logic"""
    try:
        async with throttler:
            file_paths = []
            
            # Use the SAME logic that was working in your original code
            for file in files:
                if not file.filename.lower().endswith('.pdf'):
                    continue
                
                # Save file (using your working approach)
                file_path = config.DOCUMENTS_DIR / file.filename
                async with aiofiles.open(file_path, 'wb') as f:
                    content = await file.read()
                    await f.write(content)
                
                file_paths.append(str(file_path))
            
            if not file_paths:
                raise HTTPException(status_code=400, detail="No valid PDF files")
            
            # Process with universal pipeline (this should work)
            result = await multimodal_retriever.process_documents(file_paths)
            return JSONResponse(content=result)
            
    except Exception as e:
        logger.error(f"Upload error: {e}")
        return JSONResponse(
            content={"status": "error", "message": str(e)},
            status_code=500
        )

@app.post("/query")
async def query_documents(request: Dict[str, Any]):
    """Universal query processing - works with any question about any content"""
    try:
        async with throttler:
            question = request.get("question")
            use_images = request.get("use_images", True)
            
            if not question:
                raise HTTPException(status_code=400, detail="Question required")
            
            # Generate universal response
            result = await multimodal_retriever.retrieve_and_generate(
                question=question,
                use_images=use_images
            )
            
            return JSONResponse(content=result)
            
    except Exception as e:
        logger.error(f"Query error: {e}")
        return JSONResponse(
            content={"status": "error", "message": str(e)},
            status_code=500
        )

@app.get("/health")
async def health_check():
    """System health check"""
    try:
        return {
            "status": "healthy",
            "system": "Universal Multimodal RAG",
            "components": {
                "multimodal_retriever": multimodal_retriever.retriever is not None,
                "vllm_server": True,
                "vector_store": multimodal_retriever.vectorstore is not None,
                "text_embeddings": multimodal_retriever.text_embeddings is not None,
                "image_embeddings": multimodal_retriever.image_embeddings is not None
            },
            "capabilities": [
                "Universal PDF Processing",
                "Any Document Type Support", 
                "Intelligent Content Analysis",
                "Visual Element Integration",
                "Semantic Search & Retrieval",
                "Natural Language Responses"
            ],
            "supported_content": [
                "Business reports and presentations",
                "Academic papers and research",
                "Technical manuals and documentation", 
                "Books and educational materials",
                "Financial documents and statements",
                "Any PDF with text and/or visual content"
            ]
        }
        
    except Exception as e:
        return JSONResponse(
            content={"status": "unhealthy", "error": str(e)},
            status_code=500
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=False,
        workers=1,
        log_level="info"
    )