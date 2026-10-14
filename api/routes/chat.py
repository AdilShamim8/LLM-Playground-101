"""
Chat API routes with streaming support.
OpenAI-compatible API format.
"""

import asyncio
import json
import time
import uuid
from collections.abc import AsyncIterator

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from loguru import logger
from pydantic import BaseModel, Field

router = APIRouter()


# ─── Request/Response Models ──────────────────────────────────────


class Message(BaseModel):
    role: str = Field(..., pattern="^(system|user|assistant)$")
    content: str = Field(..., min_length=1, max_length=32000)


class ChatCompletionRequest(BaseModel):
    model: str = "llm-playground-7b"
    messages: list[Message]
    max_tokens: int = Field(default=256, ge=1, le=4096)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    top_k: int = Field(default=50, ge=0, le=100)
    decoding_strategy: str = Field(
        default="sampling", pattern="^(sampling|greedy|beam_search)$"
    )
    stream: bool = False
    stop: list[str] | None = None
    repetition_penalty: float = Field(default=1.1, ge=1.0, le=2.0)
    n: int = Field(default=1, ge=1, le=5)


class ChatCompletionChoice(BaseModel):
    index: int
    message: Message
    finish_reason: str


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionChoice]
    usage: dict


class ModelRegistry:
    """Simple in-memory model registry."""

    _models = {}

    @classmethod
    def register(cls, name: str, model, tokenizer, generator):
        cls._models[name] = {
            "model": model,
            "tokenizer": tokenizer,
            "generator": generator,
            "loaded_at": time.time(),
        }

    @classmethod
    def get(cls, name: str) -> dict | None:
        return cls._models.get(name)

    @classmethod
    def list_models(cls) -> list[str]:
        return list(cls._models.keys())


def _format_messages_to_prompt(messages: list[Message]) -> str:
    """Format chat messages into a single prompt string."""
    prompt = ""
    for msg in messages:
        if msg.role == "system":
            prompt += f"<|im_start|>system\n{msg.content}<|im_end|>\n"
        elif msg.role == "user":
            prompt += f"<|im_start|>user\n{msg.content}<|im_end|>\n"
        elif msg.role == "assistant":
            prompt += f"<|im_start|>assistant\n{msg.content}<|im_end|>\n"
    prompt += "<|im_start|>assistant\n"
    return prompt


def _demo_response(messages: list[Message]) -> str:
    last_user = ""
    for msg in reversed(messages):
        if msg.role == "user":
            last_user = msg.content.strip()
            break

    if last_user:
        user_line = f"You said: {last_user}"
    else:
        user_line = "Send a message to get a response."

    return "\n".join(
        [
            "Demo mode: no model is loaded yet.",
            "",
            user_line,
            "",
            "To get real responses:",
            "1) Load a model via POST /api/v1/models/load",
            "2) Select it in the Model dropdown",
            "3) Send your message again",
        ]
    )


async def _stream_response(
    generator,
    prompt: str,
    request: ChatCompletionRequest,
    completion_id: str,
) -> AsyncIterator[str]:
    """
    Async streaming response generator.

    PERFORMANCE: The generator.generate_stream() is a synchronous iterator.
    We wrap each next() call in asyncio.to_thread() so the event loop
    isn't blocked while waiting for the next token. This allows other
    requests to be served between tokens.
    """
    try:
        # Create the synchronous iterator
        sync_iter = generator.generate_stream(
            prompt,
            strategy=request.decoding_strategy,
            max_new_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            repetition_penalty=request.repetition_penalty,
        )

        # Consume tokens in a thread pool to avoid blocking the event loop
        import queue
        import threading

        token_queue: queue.Queue = queue.Queue()
        _SENTINEL = object()

        def _produce_tokens():
            """Run the blocking generator in a background thread."""
            try:
                for token in sync_iter:
                    token_queue.put(token)
            except Exception as e:
                token_queue.put(e)
            finally:
                token_queue.put(_SENTINEL)

        # Start the producer thread
        producer = threading.Thread(target=_produce_tokens, daemon=True)
        producer.start()

        # Consume tokens from the queue, yielding to the event loop
        while True:
            try:
                # Use asyncio.to_thread to avoid blocking while waiting
                item = await asyncio.to_thread(token_queue.get, timeout=30.0)
            except Exception:
                # Timeout — producer may have hung
                break

            if item is _SENTINEL:
                break
            if isinstance(item, Exception):
                raise item

            token = item

            # Stop sequence check
            if request.stop and any(stop in token for stop in request.stop):
                break

            chunk = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": request.model,
                "choices": [
                    {"index": 0, "delta": {"content": token}, "finish_reason": None}
                ],
            }
            yield f"data: {json.dumps(chunk)}\n\n"
            await asyncio.sleep(0)  # Yield control to event loop

        # Final chunk
        final_chunk = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": request.model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }
        yield f"data: {json.dumps(final_chunk)}\n\n"
        yield "data: [DONE]\n\n"

    except Exception as e:
        logger.error(f"Streaming error: {e}")
        error_chunk = {"error": str(e)}
        yield f"data: {json.dumps(error_chunk)}\n\n"


@router.post("/chat/completions")
async def chat_completions(request: ChatCompletionRequest, http_request: Request):
    """
    OpenAI-compatible chat completions endpoint.
    Supports streaming and batch responses.
    """
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"

    # Get model from registry
    model_info = ModelRegistry.get(request.model)
    if not model_info:
        # Fall back to demo mode
        demo_response = _demo_response(request.messages)
        if request.stream:

            async def demo_stream():
                for word in demo_response.split():
                    chunk = {
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": request.model,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": word + " "},
                                "finish_reason": None,
                            }
                        ],
                    }
                    yield f"data: {json.dumps(chunk)}\n\n"
                    await asyncio.sleep(0.05)
                yield "data: [DONE]\n\n"

            return StreamingResponse(demo_stream(), media_type="text/event-stream")

        return ChatCompletionResponse(
            id=completion_id,
            created=int(time.time()),
            model=request.model,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=Message(role="assistant", content=demo_response),
                    finish_reason="stop",
                )
            ],
            usage={
                "prompt_tokens": 0,
                "completion_tokens": len(demo_response.split()),
                "total_tokens": len(demo_response.split()),
            },
        )

    generator = model_info["generator"]
    prompt = _format_messages_to_prompt(request.messages)

    if request.stream:
        return StreamingResponse(
            _stream_response(generator, prompt, request, completion_id),
            media_type="text/event-stream",
        )

    # Non-streaming response
    #
    # PERFORMANCE: generator.generate() is a synchronous PyTorch call that
    # can take 1-30 seconds. Calling it directly in an async def BLOCKS
    # the entire event loop — no other request can be served during
    # generation. With 1 worker, the server handles 1 request at a time.
    #
    # FIX: Run the blocking call in a thread pool via asyncio.to_thread().
    # This frees the event loop to handle other requests while generation
    # runs in the background. The thread pool size is managed by asyncio
    # (default: min(32, os.cpu_count() + 4)).
    #
    # For true production scale, use a dedicated inference server (vLLM,
    # TGI, TensorRT-LLM) that supports continuous batching. This fix
    # addresses the immediate event-loop blocking issue.
    try:
        t0 = time.time()

        # Run the blocking generate() call in a thread pool
        response_text = await asyncio.to_thread(
            generator.generate,
            prompt,
            strategy=request.decoding_strategy,
            max_new_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            repetition_penalty=request.repetition_penalty,
        )
        latency = time.time() - t0

        # Rough token count
        prompt_tokens = len(prompt.split())
        completion_tokens = len(response_text.split())

        logger.info(
            f"Chat | Model: {request.model} | "
            f"Latency: {latency:.2f}s | "
            f"Tokens: {completion_tokens}"
        )

        return ChatCompletionResponse(
            id=completion_id,
            created=int(time.time()),
            model=request.model,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=Message(role="assistant", content=response_text),
                    finish_reason="stop",
                )
            ],
            usage={
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
                "latency_seconds": latency,
            },
        )

    except Exception as e:
        logger.error(f"Generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/models")
async def list_models():
    """List available loaded models."""
    return {"models": ModelRegistry.list_models()}
