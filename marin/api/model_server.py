#!/usr/bin/env python3
"""
FastAPI server for serving the Marin model.
"""

import logging
import os
import time
from typing import Any, Dict, List, Optional

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
)
import gc
import jinja2

# SETTINGS
MODEL_PATH = "~/MarinModel"


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Global model and tokenizer
model: Optional[AutoModelForCausalLM] = None
tokenizer: Optional[AutoTokenizer] = None
device: str = "cuda" if torch.cuda.is_available() else "cpu"

# Note: By default, USE_CPU_OFFLOAD=false for full GPU loading
# Set USE_CPU_OFFLOAD=true only if you need CPU offloading for memory constraints

def load_chat_template() -> jinja2.Template:
    """Load the chat template from the Jinja file."""
    model_path = os.getenv("MODEL_PATH", MODEL_PATH)
    model_path = os.path.expanduser(model_path)
    template_path = os.path.join(model_path, "chat_template.jinja")
    try:
        with open(template_path, 'r') as f:
            template_content = f.read()
        
        # Replace custom {% generation %} tags with standard Jinja2 syntax
        # The generation tag seems to be for assistant messages, so we'll treat them normally
        template_content = template_content.replace("{% generation %}", "")
        template_content = template_content.replace("{% endgeneration %}", "")
        
        return jinja2.Template(template_content)
    except FileNotFoundError:
        logger.error(f"Chat template not found. Ensure {template_path} exists.")
        raise FileNotFoundError(f"Chat template not found. Ensure {template_path} exists.")

def format_chat_prompt(messages: List[Dict[str, str]], add_generation_prompt: bool = True) -> str:
    """Format messages using the chat template."""
    template = load_chat_template()    
    return template.render(
        messages=messages,
        add_generation_prompt=add_generation_prompt,
        bos_token=""
    )

# Mock class to mimic Hugging Face generation output format
class MockGenerationOutput:
    def __init__(self, sequences, generated_tokens):
        self.sequences = [sequences] if hasattr(sequences, 'shape') and len(sequences.shape) == 1 else sequences
        self.generated_tokens = generated_tokens

# Pydantic models for API
class GenerateRequest(BaseModel):
    prompt: str = Field(..., description="Input text prompt")
    max_length: int = Field(2048, ge=1, le=8192, description="Maximum length of generated text")
    max_think_effort: Optional[int] = Field(None, ge=1, description="Maximum tokens before injecting <|end_think|>")
    temperature: float = Field(0.5, ge=0.0, le=2.0, description="Sampling temperature")
    top_p: float = Field(0.9, ge=0.0, le=1.0, description="Top-p sampling parameter")
    top_k: int = Field(20, ge=1, le=1000, description="Top-k sampling parameter")
    repetition_penalty: float = Field(1.1, ge=0.0, le=2.0, description="Repetition penalty")
    do_sample: bool = Field(True, description="Whether to use sampling")
    num_return_sequences: int = Field(1, ge=1, le=5, description="Number of sequences to return")
    stop_sequences: Optional[List[str]] = Field(None, description="Sequences to stop generation at")
    stream: bool = Field(True, description="Whether to stream the response")

    def __init__(self, **data):
        super().__init__(**data)
        # Validate that max_think_effort is less than max_length if provided
        if self.max_think_effort is not None and self.max_think_effort >= self.max_length:
            raise ValueError("max_think_effort must be less than max_length")

class GenerateResponse(BaseModel):
    generated_text: str
    prompt_tokens: int
    generated_tokens: int
    total_tokens: int
    generation_time: float
    model_name: str

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
    model_name: str
    memory_usage: Optional[Dict[str, Any]] = None
    optimization_mode: str

# Initialize FastAPI app
app = FastAPI(
    title="Marin Model API",
    description="API for serving the Marin language model",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def load_model(model_path: str) -> None:
    """Load the model and tokenizer."""
    global model, tokenizer
    
    logger.info(f"Loading model from {model_path}")
    logger.info(f"Using device: {device}")
    
    try:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            use_fast=False,
        )
        
        # Set pad token if not present
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Check if we should use CPU offloading
        use_cpu_offload = os.getenv("USE_CPU_OFFLOAD", "false").lower() == "true"
        
        if device == "cuda" and use_cpu_offload:
            logger.info("Using CPU offloading: weights in CPU, computations on GPU")
            
            # Load model with proper CPU offloading
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map="cpu",  # Proper CPU offloading - weights on CPU
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            )
            
            # Enable aggressive memory optimization
            model = model.half()  # Convert to float16 for memory efficiency
            
            # Enable gradient checkpointing to save memory during generation
            if hasattr(model, 'gradient_checkpointing_enable'):
                model.gradient_checkpointing_enable()
            
            # Set model to evaluation mode
            model.eval()
            
            logger.info("Model loaded with proper CPU offloading - weights on CPU, will move to GPU during generation")
            
        elif device == "cuda":
            logger.info("Using full GPU loading")
            
            # Load model directly on GPU with memory optimization
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map="auto",  # Let transformers handle GPU placement optimally
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            )
            
            # Ensure model is in evaluation mode
            model.eval()
            
            logger.info("Model loaded directly on GPU with automatic device mapping")
            
        else:
            logger.info("Using CPU mode")
            
            # Load model on CPU
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float32,
                device_map=None,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            )
            model = model.to("cpu")
        
        logger.info("Model loaded successfully!")
        
        # Log tokenizer configuration
        # if tokenizer:
            # logger.info(f"Tokenizer loaded: {type(tokenizer).__name__}")
            # logger.info(f"Vocab size: {tokenizer.vocab_size}")
            # logger.info(f"BOS token ID: {tokenizer.bos_token_id}")
            # logger.info(f"EOS token ID: {tokenizer.eos_token_id}")
            # logger.info(f"PAD token ID: {tokenizer.pad_token_id}")
            # logger.info(f"Special tokens: {tokenizer.special_tokens_map}")
        
        # Validate the configuration
        # if not validate_model_configuration():
        #     logger.warning("Model configuration validation failed, but continuing...")
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise

def get_generation_config(request: GenerateRequest) -> GenerationConfig:
    """Create generation config from request parameters."""
    logger.info(f"EOT token ID: {tokenizer.convert_tokens_to_ids('<|eot_id|>')}")
    return GenerationConfig(
        max_length=request.max_length,
        temperature=request.temperature,
        top_p=request.top_p,
        top_k=request.top_k,
        repetition_penalty=request.repetition_penalty,
        do_sample=request.do_sample,
        num_return_sequences=request.num_return_sequences,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eot_token_id=tokenizer.convert_tokens_to_ids("<|eot_id|>"),
        end_think_token=tokenizer.convert_tokens_to_ids("<|end_think|>"),
        start_think_token=tokenizer.convert_tokens_to_ids("<|start_think|>"),
        # Explicitly set model-specific defaults to avoid warnings
        begin_suppress_tokens=[128000, 128001] if tokenizer and hasattr(tokenizer, 'eos_token_id') and tokenizer.eos_token_id == 128001 else None,
        decoder_start_token_id=128000 if tokenizer and hasattr(tokenizer, 'bos_token_id') and tokenizer.bos_token_id == 128000 else None,
    )

def generate_with_think(
        model: AutoModelForCausalLM,
        inputs: Dict[str, torch.Tensor],
        generation_config: GenerationConfig,
        max_think_effort: int,
        max_length: int,
        device: str
    ) -> MockGenerationOutput:
    """
    Generate text with think tokens.
    
    The flow is:
    1. Inject <|start_think|> into the prompt
    2. Let model generate continuously until <|end_think|> or max_think_effort
    3. If no <|end_think|> by max_think_effort, force inject <|end_think|>
    4. Continue generation until completion
    
    Args:
        model: The language model
        inputs: Input tensors
        generation_config: Generation configuration
        max_think_effort: Maximum tokens before injecting <|end_think|>
        device: Device to generate on
        
    Returns:
        Generated outputs with <|end_think|> injected when max_think_effort is reached
    """
    # Get the token IDs
    start_think_token_id = generation_config.start_think_token
    end_think_token_id = generation_config.end_think_token
    logger.info(f"Using start_think token ID: {start_think_token_id}, end_think token ID: {end_think_token_id}")
    
    # Step 1: Inject <|start_think|> into the prompt
    batch_size = inputs["input_ids"].shape[0]
    start_think_tensor = torch.full((batch_size, 1), start_think_token_id, device=device, dtype=inputs["input_ids"].dtype)
    prompt_with_start_think = torch.cat(
        [inputs["input_ids"], start_think_tensor],
        dim=-1
    )
    
    # Step 2a: Generate limit to prevent runaway generation
    logger.info(f"Starting generation with max_think_effort={max_think_effort}")
    outputs = model.generate(
        input_ids=prompt_with_start_think,
        attention_mask=torch.cat([
            inputs.get("attention_mask", torch.ones((batch_size, 1), device=device)), 
            torch.ones((batch_size, 1), device=device)
        ], dim=-1) if inputs.get("attention_mask") is not None else None,
        generation_config=generation_config,
        return_dict_in_generate=True,
        output_scores=False,
        forced_eos_token_id=end_think_token_id,
        eos_token_id=[end_think_token_id],
        early_stopping=True,
        bad_words_ids=[[generation_config.eot_token_id]],
        max_new_tokens=max_think_effort,
    )

    # Step 3: Continue generation until completion
    logger.info("Continuing generation after <|end_think|>...")
    post_think_outputs = model.generate(
        input_ids=outputs.sequences,
        attention_mask=torch.ones_like(outputs.sequences),
        generation_config=generation_config,
        return_dict_in_generate=True,
        output_scores=False,
        max_new_tokens=max_length - max_think_effort,  # Use remaining tokens up to max_length
        eos_token_id=[generation_config.eot_token_id],
        forced_eos_token_id=generation_config.eot_token_id,
    )
    final_sequence = post_think_outputs.sequences[0]
    
    # Return all generated tokens as one sequence
    all_generated_tokens = final_sequence[inputs["input_ids"].shape[1]:]
    return MockGenerationOutput(final_sequence, [all_generated_tokens])


def generate_without_think(
    model: AutoModelForCausalLM,
    inputs: Dict[str, torch.Tensor],
    generation_config: GenerationConfig,
    max_length: int,
) -> Any:
    """Generate text without think effort monitoring."""
    return model.generate(
        **inputs,
        generation_config=generation_config,
        return_dict_in_generate=True,
        output_scores=False,
        max_new_tokens=max_length,
        eos_token_id=generation_config.eot_token_id,
    )

def _clear_gpu_cache():
    """Clear GPU cache with synchronization."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        gc.collect()

def _is_cpu_offload_enabled():
    """Check if CPU offloading is enabled."""
    return os.getenv("USE_CPU_OFFLOAD", "false").lower() == "true"

def _should_use_cpu_offload():
    """Check if we should use CPU offloading based on environment and device."""
    return _is_cpu_offload_enabled() and device == "cuda"

def optimize_memory_for_generation():
    """Optimize memory usage before generation."""
    global model
    
    if model is None:
        return False
    
    if _should_use_cpu_offload():
        # Clear GPU cache to free up memory
        _clear_gpu_cache()
        
        # Move model to GPU temporarily for generation
        logger.info("Moving model to GPU for generation...")
        model = model.to(device)
        
        # Ensure model is in evaluation mode
        model.eval()
        
        return True
    
    return False


def cleanup_after_generation():
    """Perform aggressive memory cleanup."""
    global model
    
    if model is None:
        return
    
    # Force garbage collection
    gc.collect()
    
    if torch.cuda.is_available():
        # Clear all GPU caches
        _clear_gpu_cache()
        
        # Check if we should move model to CPU
        if _should_use_cpu_offload():
            logger.info("Performing aggressive memory cleanup - moving model to CPU...")
            model = model.to("cpu")
            _clear_gpu_cache()

def validate_model_configuration():
    """Validate that the model and tokenizer are properly configured."""
    global model, tokenizer
    
    if model is None or tokenizer is None:
        return False
    
    try:
        # Check if model is properly configured
        if not hasattr(model, 'config'):
            logger.error("Model missing config attribute")
            return False
        
        # Check tokenizer configuration
        if not hasattr(tokenizer, 'vocab_size'):
            logger.error("Tokenizer missing vocab_size attribute")
            return False
        
        # Log current configuration
        logger.info(f"Model type: {model.config.model_type}")
        logger.info(f"Model config: {model.config}")
        logger.info(f"Tokenizer vocab size: {tokenizer.vocab_size}")
        
        return True
        
    except Exception as e:
        logger.error(f"Model validation failed: {e}")
        return False

@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    model_path = os.getenv("MODEL_PATH", MODEL_PATH)
    model_path = os.path.expanduser(model_path)
    
    if not os.path.exists(model_path):
        logger.error(f"Model path does not exist: {model_path}")
        return
    
    load_model(model_path)

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    memory_info = None
    optimization_mode = "full_gpu"
    
    if device == "cuda" and torch.cuda.is_available():
        memory_info = {
            "allocated": f"{torch.cuda.memory_allocated() / 1024**3:.2f} GB",
            "reserved": f"{torch.cuda.memory_reserved() / 1024**3:.2f} GB",
            "total": f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB",
        }
        
        # Check optimization mode
        use_cpu_offload = os.getenv("USE_CPU_OFFLOAD", "false").lower() == "true"
        if use_cpu_offload:
            optimization_mode = "cpu_offload"
    
    return HealthResponse(
        status="healthy",
        model_loaded=model is not None and tokenizer is not None,
        device=device,
        model_name="Marin Model",
        memory_usage=memory_info,
        optimization_mode=optimization_mode,
    )

@app.post("/generate", response_model=GenerateResponse)
async def generate_text(request: GenerateRequest):
    """Generate text from prompt."""
    global model, tokenizer
    
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    start_time = time.time()
    
    # Initialize moved_to_gpu variable
    moved_to_gpu = False
    
    try:
        # Check if we should use CPU offloading
        use_cpu_offload = os.getenv("USE_CPU_OFFLOAD", "false").lower() == "true"
        
        # Optimize memory before generation
        moved_to_gpu = optimize_memory_for_generation()
        
        # Get the current device for the model and ensure consistency
        if model is not None:
            current_device = next(model.parameters()).device
            logger.info(f"Model is on device: {current_device}")
        else:
            raise HTTPException(status_code=503, detail="Model not available")
        
        # Tokenize input
        inputs = tokenizer(
            request.prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=request.max_length // 2,  # Reserve space for generation
        ).to(current_device)
        
        prompt_tokens = inputs.input_ids.shape[1]
        logger.info(f"Input tensors on device: {inputs.input_ids.device}")
        
        # Create generation config
        generation_config = get_generation_config(request)
        
        # Generate with memory optimization and think effort monitoring
        with torch.no_grad():
            try:
                if request.max_think_effort is not None:
                    logger.info(f"Generating with think effort monitoring (max_think_effort={request.max_think_effort})")
                    # Custom generation loop to monitor think effort
                    outputs = generate_with_think(
                        model,
                        inputs,
                        generation_config,
                        request.max_think_effort,
                        request.max_length,
                        current_device,
                    )
                else:
                    logger.info("Generating without think effort monitoring")
                    # Standard generation
                    outputs = generate_without_think(
                        model,
                        inputs,
                        generation_config,
                        request.max_length
                    )
            except torch.cuda.OutOfMemoryError:
                logger.warning("GPU out of memory during generation, attempting memory cleanup...")
                
                # Clear GPU memory and try again
                if torch.cuda.is_available():
                    _clear_gpu_cache()
                
                # If still out of memory, move model to CPU and generate there
                if use_cpu_offload and device == "cuda":
                    logger.info("Moving model to CPU for generation due to GPU memory constraints...")
                    model = model.to("cpu")
                    inputs = {k: v.to("cpu") for k, v in inputs.items()}
                    
                    if request.max_think_effort is not None:
                        outputs = generate_with_think(
                            model,
                            inputs,
                            generation_config,
                            request.max_think_effort,
                            request.max_length,
                            "cpu",
                        )
                    else:
                        outputs = generate_without_think(
                            model,
                            inputs,
                            generation_config,
                            request.max_length
                        )
                    
                    # Move model back to CPU (it was already there)
                    logger.info("Generation completed on CPU")
                else:
                    raise
        
        # Decode generated text
        generated_text = tokenizer.decode(
            outputs.sequences[0][prompt_tokens:],
            skip_special_tokens=False,
        )
        
        # Log the generated text for debugging
        logger.info(f"Generated text: '{generated_text}'")
        logger.info(f"Generated text length: {len(generated_text)}")
        
        # Also log without skip_special_tokens to see if special tokens are there
        generated_text_with_special = tokenizer.decode(
            outputs.sequences[0][prompt_tokens:],
            skip_special_tokens=False,
        )
        # logger.info(f"Generated text with special tokens: '{generated_text_with_special}'")
        
        generated_tokens = len(outputs.sequences[0]) - prompt_tokens
        generation_time = time.time() - start_time
        
        # Clean up memory after generation
        if moved_to_gpu:
            cleanup_after_generation()
        
        return GenerateResponse(
            generated_text=generated_text_with_special,  # Use the version with special tokens
            prompt_tokens=prompt_tokens,
            generated_tokens=generated_tokens,
            total_tokens=prompt_tokens + generated_tokens,
            generation_time=generation_time,
            model_name="Marin Model",
        )
        
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        # Ensure cleanup happens even on error
        if moved_to_gpu:
            cleanup_after_generation()
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

@app.post("/chat")
async def chat_completion(request: GenerateRequest):
    """Chat completion endpoint (similar to OpenAI API)."""
    global model, tokenizer
    
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Format as chat using the proper template format
    if not request.prompt.strip().startswith("<|start_header_id|>"):
        # Format using the chat template structure
        messages = [{"role": "user", "content": request.prompt}]
        formatted_prompt = format_chat_prompt(messages, add_generation_prompt=True)
    else:
        formatted_prompt = request.prompt
    
    # Create a new request with formatted prompt
    chat_request = GenerateRequest(
        prompt=formatted_prompt,
        max_length=request.max_length,
        max_think_effort=request.max_think_effort,
        temperature=request.temperature,
        top_p=request.top_p,
        top_k=request.top_k,
        repetition_penalty=request.repetition_penalty,
        do_sample=request.do_sample,
        num_return_sequences=1,
        stream=request.stream,
    )
    
    # Generate response
    creation_time = time.time()
    response = await generate_text(chat_request)
    generation_time = time.time() - creation_time
    
    # Format as chat response
    return {
        "id": f"chatcmpl-{int(time.time())}",
        "object": "chat.completion",
        "created": int(creation_time),
        "model": "Marin Model",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": response.generated_text,  # This now includes special tokens
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": response.prompt_tokens,
            "completion_tokens": response.generated_tokens,
            "total_tokens": response.total_tokens,
        },
        "statistics": {
            "generation_time": generation_time,
        },
        "request_params": {
            "raw_prompt": request.prompt,
            "input_prompt": formatted_prompt,
            "max_length": request.max_length,
            "max_think_effort": request.max_think_effort,
            "temperature": request.temperature,
            "top_p": request.top_p,
            "top_k": request.top_k,
        }
    }

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Marin Model API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "generate": "/generate",
            "chat": "/chat",
            "optimize": "/optimize",
        },
        "docs": "/docs",
    }

@app.post("/optimize")
async def optimize_memory():
    """Manually trigger memory optimization."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Force garbage collection
        gc.collect()
        
        if device == "cuda" and torch.cuda.is_available():
            # Clear GPU cache
            _clear_gpu_cache()
            
            # Check if we should move model to CPU
            use_cpu_offload = os.getenv("USE_CPU_OFFLOAD", "false").lower() == "true"
            if use_cpu_offload:
                logger.info("Moving model to CPU for memory optimization...")
                model.to("cpu")
                _clear_gpu_cache()
        
        return {
            "message": "Memory optimization completed",
            "device": device,
            "optimization_mode": "cpu_offload" if os.getenv("USE_CPU_OFFLOAD", "false").lower() == "true" else "full_gpu"
        }
        
    except Exception as e:
        logger.error(f"Memory optimization failed: {e}")
        raise HTTPException(status_code=500, detail=f"Memory optimization failed: {str(e)}")

@app.post("/aggressive-cleanup")
async def cleanup_after_generation_endpoint():
    """Manually trigger aggressive memory cleanup."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        cleanup_after_generation()
        
        return {
            "message": "Aggressive memory cleanup completed",
            "device": device,
            "optimization_mode": "cpu_offload" if os.getenv("USE_CPU_OFFLOAD", "false").lower() == "true" else "full_gpu"
        }
        
    except Exception as e:
        logger.error(f"Aggressive memory cleanup failed: {e}")
        raise HTTPException(status_code=500, detail=f"Aggressive memory cleanup failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    
    # Load model if not loaded via startup event
    if model is None:
        model_path = os.getenv("MODEL_PATH", "~/MarinModel")
        model_path = os.path.expanduser(model_path)
        load_model(model_path)
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.getenv("PORT", 9123)),
        log_level="info",
    )
