import asyncio
import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import uuid
import warnings
import base64
from io import BytesIO

# Suppress deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Updated LangChain imports (fixing deprecations)
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings

try:
    from langchain_chroma import Chroma
except ImportError:
    from langchain_community.vectorstores import Chroma

from langchain.schema import Document
from langchain.retrievers import MultiVectorRetriever
from langchain.storage import InMemoryStore
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Document processing
import fitz  # PyMuPDF
from PIL import Image
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Optional: OpenCV for advanced image processing (fallback if not available)
try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False
    logging.warning("OpenCV not available. Using basic image processing.")

from config import config
from vllm_server import vllm_server

logger = logging.getLogger(__name__)

class UniversalMultimodalRetriever:
    """Universal RAG system that works with ANY PDF document - text, images, charts, tables, diagrams"""
    
    def __init__(self):
        # LangChain components
        self.text_embeddings = None
        self.image_embeddings = None
        self.vectorstore = None
        self.retriever = None
        self.docstore = InMemoryStore()
        
        # Document tracking
        self.processed_docs: Dict[str, Any] = {}
        self.image_paths: List[str] = []
        self.visual_elements: Dict[str, Dict] = {}  # Store any visual element metadata
        self.image_embeddings_cache: Dict[str, np.ndarray] = {}
        
        # Text processing
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""]
        )
        
    async def initialize(self):
        """Initialize universal system"""
        try:
            logger.info("🚀 Initializing Universal Multimodal RAG System...")
            
            # Text embeddings with updated import
            self.text_embeddings = HuggingFaceEmbeddings(
                model_name=config.TEXT_EMBEDDING_MODEL,
                model_kwargs={'device': config.DEVICE},
                encode_kwargs={'normalize_embeddings': True}
            )
            
            # Image embeddings
            self.image_embeddings = SentenceTransformer(
                config.IMAGE_EMBEDDING_MODEL,
                device=config.DEVICE
            )
            
            # Vector store with updated import
            self.vectorstore = Chroma(
                persist_directory=config.CHROMA_PERSIST_DIR,
                embedding_function=self.text_embeddings
            )
            
            # Initialize vLLM
            await vllm_server.initialize()
            
            logger.info("✅ Universal system ready - works with ANY PDF content!")
            
        except Exception as e:
            logger.error(f"❌ Initialization failed: {e}")
            raise

    def _extract_visual_elements(self, page, page_num: int, base_name: str) -> List[Dict]:
        """Extract ALL visual elements from PDF page - works with any content"""
        extracted_elements = []
        
        try:
            # Convert page to high-quality image
            mat = fitz.Matrix(2.0, 2.0)  # High resolution
            pix = page.get_pixmap(matrix=mat)
            img_data = pix.tobytes("png")
            
            # Always save the full page image as fallback
            page_img = Image.open(BytesIO(img_data))
            page_filename = f"{base_name}_page_{page_num}_{uuid.uuid4().hex[:8]}.png"
            page_path = config.STATIC_DIR / "images" / page_filename
            page_img.save(str(page_path))
            
            page_data = {
                "path": str(page_path),
                "filename": page_filename,
                "type": "page_content",
                "page": page_num,
                "description": f"Page {page_num + 1} content",
                "area": page_img.width * page_img.height,
                "dimensions": (page_img.width, page_img.height)
            }
            
            extracted_elements.append(page_data)
            
            # Try advanced extraction if OpenCV is available
            if OPENCV_AVAILABLE:
                try:
                    # Convert to OpenCV format
                    nparr = np.frombuffer(img_data, np.uint8)
                    cv_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
                    
                    # Detect potential visual elements (tables, charts, diagrams)
                    edges = cv2.Canny(gray, 50, 150)
                    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    height, width = gray.shape
                    min_area = (width * height) * 0.03  # At least 3% of page
                    max_area = (width * height) * 0.7   # At most 70% of page
                    
                    visual_candidates = []
                    for contour in contours:
                        area = cv2.contourArea(contour)
                        if min_area < area < max_area:
                            x, y, w, h = cv2.boundingRect(contour)
                            aspect_ratio = w / h
                            if 0.2 < aspect_ratio < 5.0 and w > 80 and h > 60:
                                visual_candidates.append((x, y, w, h, area))
                    
                    # Extract top visual elements
                    visual_candidates.sort(key=lambda x: x[4], reverse=True)
                    for i, (x, y, w, h, area) in enumerate(visual_candidates[:2]):  # Max 2 per page
                        try:
                            element_region = cv_img[y:y+h, x:x+w]
                            element_pil = Image.fromarray(cv2.cvtColor(element_region, cv2.COLOR_BGR2RGB))
                            
                            element_filename = f"{base_name}_page_{page_num}_element_{i}_{uuid.uuid4().hex[:8]}.png"
                            element_path = config.STATIC_DIR / "images" / element_filename
                            element_pil.save(str(element_path))
                            
                            element_type = self._classify_visual_element(element_pil)
                            
                            element_data = {
                                "path": str(element_path),
                                "filename": element_filename,
                                "type": element_type,
                                "page": page_num,
                                "description": f"{element_type.replace('_', ' ').title()} from page {page_num + 1}",
                                "area": area,
                                "dimensions": (w, h),
                                "position": (x, y)
                            }
                            
                            extracted_elements.append(element_data)
                            
                        except Exception as e:
                            logger.debug(f"Failed to extract visual element {i}: {e}")
                            continue
                            
                except Exception as e:
                    logger.debug(f"Advanced visual extraction failed: {e}")
            
            # Store visual element metadata
            for element in extracted_elements:
                self.visual_elements[element["path"]] = element
                
                # Generate embedding
                try:
                    img = Image.open(element["path"])
                    embedding = self.image_embeddings.encode(img, convert_to_tensor=False)
                    self.image_embeddings_cache[element["path"]] = embedding
                except Exception as e:
                    logger.warning(f"Embedding generation failed for {element['path']}: {e}")
            
            return extracted_elements
            
        except Exception as e:
            logger.error(f"Visual element extraction failed for page {page_num}: {e}")
            return []

    def _classify_visual_element(self, image: Image.Image) -> str:
        """Classify visual elements - works for any content type"""
        try:
            if not OPENCV_AVAILABLE:
                return "visual_content"
                
            # Convert to grayscale for analysis
            gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
            
            # Basic shape detection
            edges = cv2.Canny(gray, 50, 150)
            
            # Check for circular shapes (pie charts, diagrams)
            circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20,
                                     param1=50, param2=30, minRadius=15, maxRadius=100)
            
            if circles is not None and len(circles[0]) > 0:
                return "chart_or_diagram"
            
            # Check for rectangular structures (tables, bar charts)
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=30, minLineLength=20, maxLineGap=5)
            
            if lines is not None and len(lines) > 10:
                horizontal_lines = 0
                vertical_lines = 0
                
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
                    
                    if angle < 10 or angle > 170:
                        horizontal_lines += 1
                    elif 80 < angle < 100:
                        vertical_lines += 1
                
                if horizontal_lines > 5 and vertical_lines > 3:
                    return "table_or_grid"
                elif vertical_lines > horizontal_lines:
                    return "chart_or_graph"
                elif horizontal_lines > 8:
                    return "text_with_lines"
            
            # Check for text-heavy content
            text_pixels = cv2.countNonZero(edges)
            total_pixels = gray.shape[0] * gray.shape[1]
            text_ratio = text_pixels / total_pixels
            
            if text_ratio > 0.15:
                return "text_content"
            elif text_ratio > 0.05:
                return "mixed_content"
            else:
                return "image_or_diagram"
                
        except Exception as e:
            logger.debug(f"Visual classification failed: {e}")
            return "visual_content"

    def _extract_pdf_content(self, pdf_path: str) -> Tuple[List[str], List[str]]:
        """Universal PDF content extraction - works with ANY PDF"""
        texts = []
        image_paths = []
        
        try:
            doc = fitz.open(pdf_path)
            base_name = Path(pdf_path).stem
            
            logger.info(f"📄 Processing {base_name} ({len(doc)} pages)")
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                
                # Extract text
                text = page.get_text().strip()
                if text:
                    texts.append(text)
                    logger.debug(f"📝 Extracted text from page {page_num + 1}")
                
                # Extract visual elements
                visual_elements = self._extract_visual_elements(page, page_num, base_name)
                
                for element in visual_elements:
                    image_paths.append(element["path"])
                    logger.debug(f"🖼️ Extracted {element['type']} from page {page_num + 1}")
            
            doc.close()
            logger.info(f"✅ Processed {base_name}: {len(texts)} text sections, {len(image_paths)} visual elements")
            
            return texts, image_paths
            
        except Exception as e:
            logger.error(f"PDF processing error for {pdf_path}: {e}")
            return [], []

    def _image_to_base64(self, image_path: str) -> str:
        """Convert image to base64 for inline display"""
        try:
            with Image.open(image_path) as img:
                # Resize if too large (for performance)
                img.thumbnail((800, 600), Image.Resampling.LANCZOS)
                
                buffer = BytesIO()
                img.save(buffer, format='PNG')
                img_str = base64.b64encode(buffer.getvalue()).decode()
                return f"data:image/png;base64,{img_str}"
        except Exception as e:
            logger.error(f"Failed to convert image to base64: {e}")
            return ""

    async def retrieve_and_generate(
        self, 
        question: str, 
        use_images: bool = True
    ) -> Dict[str, Any]:
        """Universal retrieval and generation - works with any question about any content"""
        
        try:
            start_time = asyncio.get_event_loop().time()
            
            # Step 1: Text retrieval
            text_docs = []
            if self.retriever:
                logger.info("🔍 Retrieving relevant text content...")
                retrieved_docs = await asyncio.to_thread(
                    self.retriever.invoke, 
                    question
                )
                text_docs = [doc.page_content for doc in retrieved_docs[:config.SIMILARITY_TOP_K]]
                logger.info(f"✅ Retrieved {len(text_docs)} text documents")
            
            # Step 2: Visual element retrieval
            relevant_images = []
            if use_images:
                logger.info("🖼️ Finding relevant visual content...")
                relevant_images = self._find_relevant_images(question)
                logger.info(f"✅ Found {len(relevant_images)} relevant visual elements")
            
            # Step 3: Generate response
            logger.info("🧠 Generating comprehensive response...")
            generation_result = await vllm_server.generate_integrated_response(
                question=question,
                text_context=text_docs,
                image_paths=relevant_images
            )
            
            # Step 4: Create response with inline visuals
            response_text = generation_result.get("response", "")
            
            if relevant_images:
                # Add inline images to response
                inline_images_html = []
                for i, img_path in enumerate(relevant_images):
                    element_info = self.visual_elements.get(img_path, {})
                    element_type = element_info.get("type", "visual content")
                    element_desc = element_info.get("description", f"Visual element {i+1}")
                    
                    base64_img = self._image_to_base64(img_path)
                    if base64_img:
                        img_html = f"""
<div style="margin: 20px 0; text-align: center; background: rgba(255,255,255,0.1); padding: 15px; border-radius: 10px;">
    <img src="{base64_img}" alt="{element_desc}" style="max-width: 100%; height: auto; border-radius: 8px; box-shadow: 0 4px 8px rgba(0,0,0,0.3);">
    <p style="margin-top: 10px; color: rgba(255,255,255,0.8); font-size: 14px; font-style: italic;">
        {element_desc}
    </p>
</div>"""
                        inline_images_html.append(img_html)
                
                # Insert images appropriately in response
                if inline_images_html:
                    paragraphs = response_text.split('\n\n')
                    if len(paragraphs) > 1:
                        response_with_images = paragraphs[0] + "\n\n" + "\n".join(inline_images_html) + "\n\n" + "\n\n".join(paragraphs[1:])
                    else:
                        response_with_images = response_text + "\n\n" + "\n".join(inline_images_html)
                    
                    response_text = response_with_images
            
            # Calculate metrics
            end_time = asyncio.get_event_loop().time()
            latency = end_time - start_time
            
            return {
                "status": "success",
                "response": response_text,
                "text_sources": len(text_docs),
                "image_sources": len(relevant_images),
                "latency_seconds": round(latency, 3),
                "model_used": generation_result.get("model", ""),
                "tokens_generated": generation_result.get("tokens_used", 0),
                "integration_type": generation_result.get("integration_type", "unknown"),
                "method": "Universal Multimodal RAG",
                "innovation_used": [
                    "Universal Content Processing",
                    "LangChain MultiVector Retriever",
                    "vLLM Multimodal Generation",
                    "Adaptive Visual Integration"
                ],
                "sources": {
                    "texts": text_docs[:2],
                    "images": [Path(img).name for img in relevant_images],
                    "visual_types": [self.visual_elements.get(img, {}).get("type", "unknown") for img in relevant_images]
                }
            }
            
        except Exception as e:
            logger.error(f"Retrieve and generate error: {e}")
            return {
                "status": "error",
                "message": str(e),
                "latency_seconds": 0
            }

    async def process_documents(self, file_paths: List[str]) -> Dict[str, Any]:
        """Universal document processing - handles ANY PDF content"""
        try:
            all_text_docs = []
            all_image_paths = []
            doc_summaries = []
            content_types = {}
            
            for file_path in file_paths:
                file_path = Path(file_path)
                
                if file_path.suffix.lower() == '.pdf':
                    logger.info(f"📁 Processing {file_path.name}...")
                    
                    # Extract all content
                    texts, images = self._extract_pdf_content(str(file_path))
                    
                    all_image_paths.extend(images)
                    self.image_paths.extend(images)
                    
                    # Process text with MultiVector
                    for i, text in enumerate(texts):
                        text_chunks = self.text_splitter.split_text(text)
                        
                        for j, chunk in enumerate(text_chunks):
                            doc_id = str(uuid.uuid4())
                            
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
                            
                            # Create summary for vector search
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
            
            # Create MultiVector retriever
            if doc_summaries:
                self.retriever = MultiVectorRetriever(
                    vectorstore=self.vectorstore,
                    docstore=self.docstore,
                    id_key="doc_id",
                    search_kwargs={"k": config.SIMILARITY_TOP_K}
                )
                
                await asyncio.to_thread(
                    self.retriever.vectorstore.add_documents, 
                    doc_summaries
                )
                
                logger.info("✅ Universal MultiVector retriever created")
            
            # Analyze content types
            for img_path in all_image_paths:
                element_type = self.visual_elements.get(img_path, {}).get("type", "unknown")
                content_types[element_type] = content_types.get(element_type, 0) + 1
            
            return {
                "status": "success",
                "files_processed": len(file_paths),
                "text_documents": len(all_text_docs),
                "visual_elements": len(all_image_paths),
                "content_types_detected": content_types,
                "method": "Universal Multimodal RAG",
                "innovation": "Works with ANY PDF content - text, charts, tables, images, diagrams"
            }
            
        except Exception as e:
            logger.error(f"Document processing error: {e}")
            return {"status": "error", "message": str(e)}

    def _find_relevant_images(self, question: str, top_k: int = None) -> List[str]:
        """Find relevant visual elements using semantic similarity"""
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
            
            # Sort by similarity
            similarities.sort(key=lambda x: x[1], reverse=True)
            relevant_images = [img_path for img_path, _ in similarities[:top_k]]
            
            logger.info(f"🎯 Found {len(relevant_images)} relevant visual elements")
            return relevant_images
            
        except Exception as e:
            logger.warning(f"Visual element search failed: {e}")
            return self.image_paths[:top_k] if self.image_paths else []

# Global retriever instance
multimodal_retriever = UniversalMultimodalRetriever()