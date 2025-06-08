import asyncio
import logging
from typing import Optional, Dict, Any, List
import torch
from pathlib import Path
import base64
from io import BytesIO
from PIL import Image

# vLLM 0.9.0+ imports (Updated API)
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.sampling_params import SamplingParams
from vllm.inputs import TextPrompt
from vllm.usage.usage_lib import UsageContext

from config import config

logger = logging.getLogger(__name__)

class VLLMMultimodalServer:
    """vLLM 0.9.0+ server optimized for multimodal generation"""
    
    def __init__(self):
        self.engine: Optional[AsyncLLMEngine] = None
        self.model_name = config.VLLM_MODEL
        self.sampling_params = SamplingParams(
            temperature=config.VLLM_TEMPERATURE,
            max_tokens=config.VLLM_MAX_TOKENS,
            stop=["<|im_end|>", "</s>", "<|endoftext|>"]
        )
        
    async def initialize(self):
        """Initialize vLLM 0.9.0+ engine with updated API"""
        try:
            logger.info(f"🔥 Initializing vLLM 0.9.0+ with {self.model_name}")
            
            # Updated AsyncEngineArgs for v0.9.0+ (Removed deprecated parameters)
            engine_args = AsyncEngineArgs(
                model=self.model_name,
                trust_remote_code=True,
                max_model_len=4096,
                gpu_memory_utilization=0.9,
                tensor_parallel_size=1,
                dtype="float16" if config.USE_FP16 else "float32",
                enable_prefix_caching=True,  # Performance optimization
                max_num_batched_tokens=4096,
                max_num_seqs=256,
                # Removed deprecated parameters:
                # - worker_use_ray (no longer exists)
                # - enforce_eager (changed to different parameter)
                # - enable_chunked_prefill (not in this version)
            )
            
            # Create async engine using updated API
            self.engine = AsyncLLMEngine.from_engine_args(
                engine_args,
                usage_context=UsageContext.ENGINE_CONTEXT
            )
            
            logger.info(f"✅ vLLM 0.9.0+ engine ready with multimodal support")
            
        except Exception as e:
            logger.error(f"❌ vLLM initialization failed: {e}")
            raise

    def _encode_image(self, image_path: str) -> str:
        """Optimized image encoding for multimodal models"""
        try:
            with Image.open(image_path) as img:
                # Optimize for multimodal models
                img.thumbnail(config.MAX_IMAGE_SIZE, Image.Resampling.LANCZOS)
                
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Encode to base64
                buffer = BytesIO()
                img.save(buffer, format='JPEG', quality=config.IMAGE_QUALITY)
                img_str = base64.b64encode(buffer.getvalue()).decode()
                
                return f"data:image/jpeg;base64,{img_str}"
                
        except Exception as e:
            logger.error(f"Image encoding error {image_path}: {e}")
            return None

    def _create_multimodal_prompt(
        self, 
        question: str, 
        text_context: List[str], 
        image_paths: List[str]
    ) -> Dict[str, Any]:
        """Create universal prompt for any document content"""
        
        # Build comprehensive context
        context_text = "\n\n".join([
            f"Document {i+1}: {text}" 
            for i, text in enumerate(text_context)
        ]) if text_context else "No text content available."
        
        if image_paths:
            # Create references to visual content (any type)
            visual_refs = []
            for i, img_path in enumerate(image_paths):
                img_name = Path(img_path).name
                # Extract page info if available
                page_info = ""
                if "page_" in img_name:
                    try:
                        page_num = img_name.split("page_")[1].split("_")[0]
                        page_info = f" from page {page_num}"
                    except:
                        pass
                
                visual_refs.append(f"Visual Element {i+1}: Content{page_info}")
            
            visual_list = "\n".join(visual_refs)
            
            # Create universal multimodal prompt
            prompt_text = f"""You are an intelligent document analysis assistant. You have access to both textual content and visual elements from uploaded documents.

Question: {question}

Available Text Content:
{context_text}

Available Visual Elements:
{visual_list}

INSTRUCTIONS:
1. Provide a comprehensive answer that integrates both text and visual information when relevant
2. Reference visual elements naturally (e.g., "As shown in Visual Element 1...")
3. Describe what you observe in visual content when it supports your answer
4. Focus on answering the specific question asked
5. Use clear, informative language appropriate for the content type
6. If visual elements contain data, charts, diagrams, or important information, incorporate that into your explanation
7. Maintain accuracy and cite specific information from the available content

Please provide a thorough, well-structured response that addresses the question using all available information."""

            return {
                "prompt": prompt_text,
                "multi_modal_data": None
            }
        else:
            # Text-only prompt for universal content
            prompt_text = f"""You are an intelligent document analysis assistant. Please provide a comprehensive answer to the following question based on the available content.

Question: {question}

Available Content:
{context_text}

Please provide a detailed, well-structured response that thoroughly addresses the question using the available information."""
            
            return {
                "prompt": prompt_text,
                "multi_modal_data": None
            }

    async def generate_integrated_response(
        self, 
        question: str, 
        text_context: List[str], 
        image_paths: List[str] = None
    ) -> Dict[str, Any]:
        """Generate response using vLLM 0.9.0+ API with correct method signature"""
        
        if not self.engine:
            raise RuntimeError("vLLM engine not initialized")
        
        try:
            # Create prompt with proper format
            prompt_data = self._create_multimodal_prompt(
                question, 
                text_context, 
                image_paths or []
            )
            
            # Create request using updated vLLM 0.9.0+ API
            request_id = f"req_{hash(question)}"
            
            # FIXED: Use correct vLLM 0.9.0+ API signature
            # The generate method expects: generate(prompt, sampling_params, request_id)
            if prompt_data["multi_modal_data"]:
                # For multimodal inputs, use TextPrompt format
                text_prompt = TextPrompt(
                    prompt=prompt_data["prompt"],
                    multi_modal_data=prompt_data["multi_modal_data"]
                )
                results_generator = self.engine.generate(
                    text_prompt,
                    self.sampling_params,
                    request_id
                )
            else:
                # For text-only inputs, use simple string
                results_generator = self.engine.generate(
                    prompt_data["prompt"],
                    self.sampling_params,
                    request_id
                )
            
            # Collect results
            final_output = None
            async for request_output in results_generator:
                final_output = request_output
            
            if final_output and final_output.outputs:
                generated_text = final_output.outputs[0].text
                
                return {
                    "response": generated_text,
                    "model": self.model_name,
                    "tokens_used": len(final_output.outputs[0].token_ids) if hasattr(final_output.outputs[0], 'token_ids') else 0,
                    "finish_reason": getattr(final_output.outputs[0], 'finish_reason', 'completed'),
                    "images_processed": len(image_paths) if image_paths else 0,
                    "integration_type": "multimodal_aware" if image_paths else "text_only"
                }
            else:
                return {
                    "response": "No output generated",
                    "error": True
                }
                    
        except Exception as e:
            logger.error(f"Generation error: {e}")
            return {
                "response": f"Error generating response: {str(e)}",
                "error": True
            }

# Global server instance
vllm_server = VLLMMultimodalServer()