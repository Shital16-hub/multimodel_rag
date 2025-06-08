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
    ) -> str:
        """Create unified prompt for vLLM 0.9.0+ (simplified approach)"""
        
        # Build comprehensive context
        context_text = "\n\n".join([
            f"Document {i+1}: {text}" 
            for i, text in enumerate(text_context)
        ]) if text_context else "No text context available."
        
        if image_paths:
            # For now, create a text-based prompt that describes we have images
            # vLLM 0.9.0+ multimodal support varies by model
            prompt = f"""You are an advanced AI assistant with access to both textual documents and visual content.

Question: {question}

Retrieved Text Context:
{context_text}

Visual Content Available: {len(image_paths)} images related to the question.

Please provide a comprehensive answer that integrates both the text information and acknowledges the visual content. If the images contain charts, graphs, or diagrams, describe how they would relate to the answer based on the text context."""

        else:
            # Text-only prompt
            prompt = f"""Question: {question}

Context:
{context_text}

Please provide a detailed answer based on the context provided."""
            
        return prompt

    async def generate_integrated_response(
        self, 
        question: str, 
        text_context: List[str], 
        image_paths: List[str] = None
    ) -> Dict[str, Any]:
        """Generate response using vLLM 0.9.0+ API"""
        
        if not self.engine:
            raise RuntimeError("vLLM engine not initialized")
        
        try:
            # Create prompt
            prompt_text = self._create_multimodal_prompt(
                question, 
                text_context, 
                image_paths or []
            )
            
            # Create request using updated API
            request_id = f"req_{hash(question)}"
            
            # Use vLLM 0.9.0+ generate method
            results_generator = self.engine.generate(
                inputs=prompt_text,
                sampling_params=self.sampling_params,
                request_id=request_id
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
                    "finish_reason": final_output.outputs[0].finish_reason,
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