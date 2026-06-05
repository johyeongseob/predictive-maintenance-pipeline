#!/usr/bin/env python3
"""
FastAPI LLM Server - Keeps OpenVINO LLM loaded for persistent KV cache.

This server maintains the LLM instance in memory, preserving KV cache across requests.
Benefits:
- No model loading overhead for subsequent requests
- Persistent KV cache for system prompts
- Can serve multiple clients

Usage:
    python scripts/llm_server.py
    # Or: uvicorn scripts.llm_server:app --host 127.0.0.1 --port 8000
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
import yaml
import json
import logging
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents.utility.openvino_llm import OpenVINOLLM
from src.utility import load_prompts

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="PACE LLM Server",
    description="OpenVINO LLM inference server with KV cache persistence",
    version="1.0.0"
)

# Global LLM instance
llm: Optional[OpenVINOLLM] = None
system_prompt: str = ""


class GenerateRequest(BaseModel):
    """Request model for text generation."""
    prompt: str = Field(..., description="Input prompt for generation")
    max_new_tokens: int = Field(512, description="Maximum tokens to generate", ge=1, le=2048)
    temperature: float = Field(0.7, description="Sampling temperature", ge=0.0, le=2.0)
    response_format: Optional[Dict[str, Any]] = Field(None, description="JSON schema for structured output")


class GenerateResponse(BaseModel):
    """Response model for text generation."""
    response: str = Field(..., description="Generated text")
    cached: bool = Field(..., description="Whether KV cache was used")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    cache_enabled: bool
    cache_prefix_length: int


@app.on_event("startup")
async def load_model():
    """Load the LLM model on server startup."""
    global llm, system_prompt
    
    try:
        logger.info("🚀 Starting PACE LLM Server...")
        
        # Load config
        # Read default use case from config.json
        with open("config.json", "r") as f:
            main_config = json.load(f)
        use_case_id = main_config.get("default-use-case", "pipeline_defects_detection")
        config_path = Path(f"config/{use_case_id}/config.yaml")
        
        if not config_path.exists():
            logger.error(f"❌ {config_path} not found")
            raise FileNotFoundError(f"{config_path} not found")
        
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
        
        glue_cfg = cfg.get("glue", {})
        model_id = glue_cfg.get("model_id", "models/ov_models/llms/Phi-4-mini-instruct-int4gq")
        device = glue_cfg.get("device", "GPU")
        verbose = glue_cfg.get("verbose", False)
        
        # Check if model exists
        model_path = Path(model_id)
        if not model_path.exists():
            logger.error(f"❌ Model not found: {model_id}")
            raise FileNotFoundError(f"Model not found: {model_id}")
        
        # Load LLM
        logger.info(f"📦 Loading model: {model_id} on {device}")
        suppress_thinking = glue_cfg.get("suppress_thinking", True)
        llm = OpenVINOLLM(model_path=str(model_path), device=device, verbose=verbose, suppress_thinking=suppress_thinking)
        logger.info("✅ Model loaded successfully")
        
        # Warm up the model with a dummy inference
        logger.info("🔥 Warming up model...")
        llm.warmup()
        logger.info("✅ Model warmup complete")
        
        # Load and set system prompt for KV cache
        use_case = glue_cfg.get("use_case", "pipeline_defects_detection")
        prompt_file = f"prompts/{use_case}.txt"
        
        if Path(prompt_file).exists():
            prompts = load_prompts(prompt_file)
            system_prompt = prompts.get("system", "")
            
            if system_prompt:
                llm.set_cache_prefix(system_prompt)
                logger.info(f"📦 KV cache enabled ({len(system_prompt)} chars system prompt)")
            else:
                logger.warning("⚠️ No system prompt found, KV cache not configured")
        else:
            logger.warning(f"⚠️ Prompt file not found: {prompt_file}")
        
        logger.info("✅ LLM Server ready to accept requests")
        
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        raise


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check server health and model status."""
    return HealthResponse(
        status="healthy" if llm is not None else "unhealthy",
        model_loaded=llm is not None,
        cache_enabled=llm.response_cache.enabled if llm else False,
        cache_prefix_length=len(system_prompt) if system_prompt else 0
    )


@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    """Generate text from prompt using the loaded LLM."""
    if llm is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        logger.info(f"🧠 Generating response for prompt ({len(request.prompt)} chars)...")
        
        response = llm.invoke(
            prompt=request.prompt,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            response_format=request.response_format
        )
        
        logger.info(f"✅ Generated response ({len(response)} chars)")
        
        return GenerateResponse(
            response=response,
            cached=False  # Cache tracking happens via response_cache internally
        )
        
    except Exception as e:
        logger.error(f"❌ Generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")


@app.post("/reset-cache")
async def reset_cache():
    """Reset the response cache (clears all cached responses)."""
    if llm is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    llm.response_cache.clear()
    logger.info("🔄 Response cache cleared")
    return {"status": "cache_cleared"}


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "service": "PACE LLM Server",
        "status": "running",
        "endpoints": {
            "health": "/health",
            "generate": "/generate",
            "reset_cache": "/reset-cache",
            "docs": "/docs"
        }
    }


if __name__ == "__main__":
    import uvicorn
    
    # Load config to get port
    try:
        # Read default use case from config.json
        with open("config.json", "r") as f:
            main_config = json.load(f)
        use_case_id = main_config.get("default-use-case", "pipeline_defects_detection")
        
        with open(f"config/{use_case_id}/config.yaml", "r") as f:
            cfg = yaml.safe_load(f)
        glue_cfg = cfg.get("glue", {})
        port = glue_cfg.get("server_port", 8000)
    except (FileNotFoundError, KeyError, yaml.YAMLError, json.JSONDecodeError):
        port = 8000
    
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=port,
        log_level="info"
    )
