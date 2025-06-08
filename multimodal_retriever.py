import asyncio
import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import uuid
import warnings

# Suppress deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# LangChain core (Updated imports for latest version)
from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.retrievers import MultiVectorRetriever
from langchain.storage import InMemoryStore
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Simple, reliable document processing
import fitz  # PyMuPDF
from PIL import Image
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from config import config
from vllm_server import vllm_server

logger = logging.getLogger(__name__)

class SimplifiedMultimodalRetriever:
    """LangChain MultiVector + vLLM 0.9.0+ = Integrated Multimodal RAG"""
    
    def __init__(self):
        # LangChain components (Core Innovation)
        self.text_embeddings = None
        self.image_embeddings = None
        self.vectorstore = None
        self.retriever = None
        self.docstore = InMemoryStore()
        
        # Document tracking
        self.processed_docs: Dict[str, Any] = {}
        self.image_paths: List[str] = []
        self.image_embeddings_cache: Dict[str, np.ndarray] = {}
        
        # Text processing
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""]
        )
        
    async def initialize(self):
        """Initialize simplified system with vLLM 0.9.0+"""
        try:
            logger.info("🚀 Initializing Simplified Multimodal RAG with vLLM 0.9.0+...")
            
            # Text embeddings (proven approach)
            self.text_embeddings = HuggingFaceEmbeddings(
                model_name=config.TEXT_EMBEDDING_MODEL,
                model_kwargs={'device': config.DEVICE},
                encode_kwargs={'normalize_embeddings': True}
            )
            
            # Image embeddings (simple CLIP)
            self.image_embeddings = SentenceTransformer(
                config.IMAGE_EMBEDDING_MODEL,
                device=config.DEVICE
            )
            
            # Vector store
            self.vectorstore = Chroma(
                persist_directory=config.CHROMA_PERSIST_DIR,
                embedding_function=self.text_embeddings
            )
            
            # Initialize vLLM 0.9.0+ (The performance engine)
            await vllm_server.initialize()
            
            logger.info("✅ Simplified system ready (LangChain + vLLM 0.9.0+)")
            
        except Exception as e:
            logger.error(f"❌ Initialization failed: {e}")
            raise

    # ... rest of the methods remain the same as before ...
    # (The PDF extraction, document processing, and retrieval methods don't need changes)

    async def retrieve_and_generate(
        self, 
        question: str, 
        use_images: bool = True
    ) -> Dict[str, Any]:
        """
        THE CORE INNOVATION: LangChain MultiVector + vLLM 0.9.0+ Integrated Generation
        """
        
        try:
            start_time = asyncio.get_event_loop().time()
            
            # Step 1: Text retrieval using LangChain MultiVector (Innovation #1)
            text_docs = []
            if self.retriever:
                logger.info("🔍 Retrieving with LangChain MultiVector...")
                retrieved_docs = await asyncio.to_thread(
                    self.retriever.get_relevant_documents, 
                    question
                )
                text_docs = [doc.page_content for doc in retrieved_docs[:config.SIMILARITY_TOP_K]]
                logger.info(f"✅ Retrieved {len(text_docs)} text documents")
            
            # Step 2: Image retrieval using simple CLIP similarity
            relevant_images = []
            if use_images:
                logger.info("🖼️ Finding relevant images...")
                relevant_images = self._find_relevant_images(question)
                logger.info(f"✅ Found {len(relevant_images)} relevant images")
            
            # Step 3: Integrated generation using vLLM 0.9.0+ (Innovation #2)
            logger.info("🧠 Generating integrated response with vLLM 0.9.0+...")
            generation_result = await vllm_server.generate_integrated_response(
                question=question,
                text_context=text_docs,
                image_paths=relevant_images
            )
            
            # Calculate performance metrics
            end_time = asyncio.get_event_loop().time()
            latency = end_time - start_time
            
            # Return comprehensive result
            return {
                "status": "success",
                "response": generation_result.get("response", ""),
                "text_sources": len(text_docs),
                "image_sources": len(relevant_images),
                "latency_seconds": round(latency, 3),
                "model_used": generation_result.get("model", ""),
                "tokens_generated": generation_result.get("tokens_used", 0),
                "integration_type": generation_result.get("integration_type", "unknown"),
                "method": "LangChain MultiVector + vLLM 0.9.0+ Integrated Generation",
                "innovation_used": [
                    "LangChain MultiVector Retriever",
                    "vLLM 0.9.0+ Multimodal Generation",
                    "Integrated Text-Image Responses"
                ],
                "sources": {
                    "texts": text_docs[:2],  # Show samples
                    "images": [Path(img).name for img in relevant_images]
                }
            }
            
        except Exception as e:
            logger.error(f"Retrieve and generate error: {e}")
            return {
                "status": "error",
                "message": str(e),
                "latency_seconds": 0
            }

    def _extract_pdf_content(self, pdf_path: str) -> Tuple[List[str], List[str]]:
        """Simple, reliable PDF extraction"""
        texts = []
        image_paths = []
        
        try:
            doc = fitz.open(pdf_path)
            base_name = Path(pdf_path).stem
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                
                # Extract text (simple and reliable)
                text = page.get_text().strip()
                if text:
                    texts.append(text)
                
                # Convert page to high-quality image
                mat = fitz.Matrix(1.5, 1.5)  # Good quality
                pix = page.get_pixmap(matrix=mat)
                
                # Save page image
                img_filename = f"{base_name}_page_{page_num}_{uuid.uuid4().hex[:8]}.png"
                img_path = config.STATIC_DIR / "images" / img_filename
                pix.save(str(img_path))
                
                image_paths.append(str(img_path))
                
                # Precompute image embedding for fast retrieval
                try:
                    img_embedding = self.image_embeddings.encode(
                        Image.open(img_path), 
                        convert_to_tensor=False
                    )
                    self.image_embeddings_cache[str(img_path)] = img_embedding
                except Exception as e:
                    logger.warning(f"Image embedding failed for {img_path}: {e}")
            
            doc.close()
            logger.info(f"✅ Extracted {len(texts)} text chunks, {len(image_paths)} images from {Path(pdf_path).name}")
            
            return texts, image_paths
            
        except Exception as e:
            logger.error(f"PDF extraction error: {e}")
            return [], []

    async def process_documents(self, file_paths: List[str]) -> Dict[str, Any]:
        """Process documents with LangChain MultiVector approach"""
        try:
            all_text_docs = []
            all_image_paths = []
            doc_summaries = []
            
            for file_path in file_paths:
                file_path = Path(file_path)
                
                if file_path.suffix.lower() == '.pdf':
                    # Extract content
                    texts, images = self._extract_pdf_content(str(file_path))
                    
                    all_image_paths.extend(images)
                    self.image_paths.extend(images)
                    
                    # Process text with LangChain MultiVector pattern
                    for i, text in enumerate(texts):
                        # Split large texts into chunks
                        text_chunks = self.text_splitter.split_text(text)
                        
                        for j, chunk in enumerate(text_chunks):
                            doc_id = str(uuid.uuid4())
                            
                            # Store full content in docstore
                            full_doc = Document(
                                page_content=chunk,
                                metadata={
                                    "source": str(file_path),
                                    "page": i,
                                    "chunk": j,
                                    "doc_id": doc_id,
                                    "type": "text",
                                    "file_name": file_path.name
                                }
                            )
                            self.docstore.mset([(doc_id, full_doc)])
                            all_text_docs.append(full_doc)
                            
                            # Create summary for vector search (MultiVector key insight)
                            summary = f"""File: {file_path.name}
Page: {i+1}, Chunk: {j+1}
Content Preview: {chunk[:300]}...
Full Length: {len(chunk)} characters"""
                            
                            summary_doc = Document(
                                page_content=summary,
                                metadata={
                                    "doc_id": doc_id,
                                    "source": str(file_path),
                                    "type": "summary"
                                }
                            )
                            doc_summaries.append(summary_doc)
            
            # Create LangChain MultiVector retriever (THE INNOVATION)
            if doc_summaries:
                self.retriever = MultiVectorRetriever(
                    vectorstore=self.vectorstore,
                    docstore=self.docstore,
                    id_key="doc_id",
                    search_kwargs={"k": config.SIMILARITY_TOP_K}
                )
                
                # Add summary documents to vector store
                await asyncio.to_thread(
                    self.retriever.vectorstore.add_documents, 
                    doc_summaries
                )
                
                logger.info("✅ LangChain MultiVector retriever created")
            
            return {
                "status": "success",
                "files_processed": len(file_paths),
                "text_documents": len(all_text_docs),
                "images_extracted": len(all_image_paths),
                "method": "LangChain MultiVector + vLLM 0.9.0+",
                "innovation": "Integrated text-image responses"
            }
            
        except Exception as e:
            logger.error(f"Document processing error: {e}")
            return {"status": "error", "message": str(e)}

    def _find_relevant_images(self, question: str, top_k: int = None) -> List[str]:
        """Find relevant images using CLIP similarity"""
        top_k = top_k or config.IMAGE_TOP_K
        
        try:
            if not self.image_paths or not self.image_embeddings_cache:
                return []
            
            # Encode question
            question_embedding = self.image_embeddings.encode([question])
            
            # Calculate similarities
            similarities = []
            for img_path in self.image_paths:
                if img_path in self.image_embeddings_cache:
                    img_embedding = self.image_embeddings_cache[img_path]
                    similarity = cosine_similarity(
                        question_embedding.reshape(1, -1), 
                        img_embedding.reshape(1, -1)
                    )[0][0]
                    similarities.append((img_path, similarity))
            
            # Sort by similarity and return top_k
            similarities.sort(key=lambda x: x[1], reverse=True)
            relevant_images = [img_path for img_path, _ in similarities[:top_k]]
            
            logger.info(f"🎯 Found {len(relevant_images)} relevant images")
            return relevant_images
            
        except Exception as e:
            logger.warning(f"Image relevance search failed: {e}")
            # Fallback: return first few images
            return self.image_paths[:top_k] if self.image_paths else []

# Global retriever instance
multimodal_retriever = SimplifiedMultimodalRetriever()