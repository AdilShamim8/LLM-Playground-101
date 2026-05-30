"""
Model management routes.
Load, unload, list, and get info about models.
"""

import os
import time
from typing import Optional

import torch
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from loguru import logger

from api.middleware.auth import get_current_user, require_admin
from api.routes.chat import ModelRegistry

router = APIRouter()


class LoadModelRequest(BaseModel):
    model_name: str
    model_path: str
    tokenizer_path: Optional[str] = None
    device: str = "auto"
    dtype: str = "bfloat16"
    use_flash_attn: bool = True


class ModelInfo(BaseModel):
    name: str
    path: str
    device: str
    dtype: str
    num_parameters: Optional[int] = None
    loaded_at: float
    status: str


@router.get("/models/list")
async def list_models(
    user: dict = Depends(get_current_user)
):
    """List all loaded models with metadata."""
    models = []
    for name in ModelRegistry.list_models():
        info = ModelRegistry.get(name)
        model = info.get("model")
        params = None
        if model and hasattr(model, "num_parameters"):
            try:
                params = model.num_parameters()
            except Exception:
                pass

        models.append(ModelInfo(
            name=name,
            path=info.get("path", "unknown"),
            device=str(info.get("device", "unknown")),
            dtype=str(info.get("dtype", "unknown")),
            num_parameters=params,
            loaded_at=info.get("loaded_at", 0),
            status="loaded"
        ))

    return {"models": [m.dict() for m in models]}


@router.post("/models/load")
async def load_model(
    request: LoadModelRequest,
    user: dict = Depends(require_admin)
):
    """
    Load a model into memory.
    Admin only. Supports GPTModel and HuggingFace models.
    """
    try:
        logger.info(f"Loading model: {request.model_name}")

        # Determine device
        if request.device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device = request.device

        # Determine dtype
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        dtype = dtype_map.get(request.dtype, torch.bfloat16)

        # Try loading as our custom GPTModel
        model = None
        tokenizer = None

        if os.path.exists(
            os.path.join(request.model_path, "config.json")
        ):
            try:
                from models.gpt import GPTModel
                model = GPTModel.from_pretrained(
                    request.model_path,
                    map_location=device
                )
                model = model.to(dtype).to(device)
                model.eval()
                logger.info("Loaded as GPTModel")
            except Exception as e:
                logger.warning(f"GPTModel load failed: {e}")

        if model is None:
            # Fallback: try HuggingFace
            try:
                from transformers import AutoModelForCausalLM
                model = AutoModelForCausalLM.from_pretrained(
                    request.model_path,
                    torch_dtype=dtype,
                    device_map=device,
                )
                model.eval()
                logger.info("Loaded as HuggingFace model")
            except Exception as e:
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to load model: {e}"
                )

        # Load tokenizer
        tok_path = request.tokenizer_path or request.model_path
        try:
            from data.tokenization.bpe_tokenizer import (
                ByteLevelBPETokenizer
            )
            tokenizer = ByteLevelBPETokenizer.load(tok_path)
            logger.info("Loaded custom BPE tokenizer")
        except Exception:
            try:
                from transformers import AutoTokenizer
                tokenizer = AutoTokenizer.from_pretrained(tok_path)
                logger.info("Loaded HuggingFace tokenizer")
            except Exception as e:
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to load tokenizer: {e}"
                )

        # Build generator
        from generation.strategies import GenerationConfig
        from generation.sampler import TextGenerator
        gen_config = GenerationConfig()
        generator = TextGenerator(model, tokenizer, gen_config)

        # Register
        ModelRegistry.register(
            request.model_name, model, tokenizer, generator
        )
        ModelRegistry._models[request.model_name]["path"] = (
            request.model_path
        )
        ModelRegistry._models[request.model_name]["device"] = device
        ModelRegistry._models[request.model_name]["dtype"] = (
            request.dtype
        )

        params = None
        if hasattr(model, "num_parameters"):
            params = model.num_parameters()

        logger.info(
            f"Model loaded: {request.model_name} | "
            f"Device: {device} | Params: {params}"
        )
        return {
            "status": "loaded",
            "model_name": request.model_name,
            "device": device,
            "dtype": request.dtype,
            "num_parameters": params,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Load model error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/models/{model_name}")
async def unload_model(
    model_name: str,
    user: dict = Depends(require_admin)
):
    """Unload a model from memory."""
    if model_name not in ModelRegistry._models:
        raise HTTPException(
            status_code=404, detail="Model not found"
        )

    del ModelRegistry._models[model_name]

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    logger.info(f"Unloaded model: {model_name}")
    return {"status": "unloaded", "model_name": model_name}


@router.get("/models/{model_name}/info")
async def model_info(
    model_name: str,
    user: dict = Depends(get_current_user)
):
    """Get detailed info about a loaded model."""
    info = ModelRegistry.get(model_name)
    if not info:
        raise HTTPException(
            status_code=404, detail="Model not found"
        )

    model = info.get("model")
    result = {
        "name": model_name,
        "loaded_at": info.get("loaded_at"),
        "device": str(info.get("device", "unknown")),
        "dtype": str(info.get("dtype", "unknown")),
    }

    if model and hasattr(model, "config"):
        config = model.config
        result["config"] = vars(config) if hasattr(
            config, "__dict__"
        ) else {}

    if model and hasattr(model, "num_parameters"):
        result["num_parameters"] = model.num_parameters()
        result["num_parameters_all"] = model.num_parameters(
            trainable_only=False
        )

    if torch.cuda.is_available():
        result["gpu_memory_allocated_gb"] = round(
            torch.cuda.memory_allocated() / 1e9, 2
        )
        result["gpu_memory_reserved_gb"] = round(
            torch.cuda.memory_reserved() / 1e9, 2
        )

    return result