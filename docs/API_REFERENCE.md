# API Reference — LLM Playground

Base URL: `http://localhost:8000/api/v1`

Interactive docs available at: `http://localhost:8000/docs` (Swagger UI)

---

## Authentication

All endpoints except `/health` and `/ready` require authentication.

### JWT Bearer Token
```bash
curl -H "Authorization: Bearer <JWT_TOKEN>" http://localhost:8000/api/v1/models
```

### API Key
```bash
curl -H "X-API-Key: <API_KEY>" http://localhost:8000/api/v1/models
```

---

## Chat

### POST /chat/completions

OpenAI-compatible chat completions endpoint. Supports streaming and
non-streaming responses.

**Request:**
```json
{
  "model": "llm-playground-7b",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is 2+2?"}
  ],
  "max_tokens": 256,
  "temperature": 0.7,
  "top_p": 0.9,
  "top_k": 50,
  "decoding_strategy": "sampling",
  "stream": false,
  "stop": ["\n\n"],
  "repetition_penalty": 1.1,
  "n": 1
}
```

**Response (non-streaming):**
```json
{
  "id": "chatcmpl-abc123",
  "object": "chat.completion",
  "created": 1697299200,
  "model": "llm-playground-7b",
  "choices": [{
    "index": 0,
    "message": {"role": "assistant", "content": "2 + 2 = 4"},
    "finish_reason": "stop"
  }],
  "usage": {
    "prompt_tokens": 20,
    "completion_tokens": 5,
    "total_tokens": 25,
    "latency_seconds": 1.23
  }
}
```

**Response (streaming — `stream: true`):**
```
data: {"id":"chatcmpl-abc","object":"chat.completion.chunk","choices":[{"delta":{"content":"2"},"index":0}]}

data: {"id":"chatcmpl-abc","object":"chat.completion.chunk","choices":[{"delta":{"content":" +"},"index":0}]}

data: {"id":"chatcmpl-abc","object":"chat.completion.chunk","choices":[{"delta":{"content":" 2"},"index":0}]}

data: {"id":"chatcmpl-abc","object":"chat.completion.chunk","choices":[{"delta":{"content":" = 4"},"index":0}]}

data: {"id":"chatcmpl-abc","object":"chat.completion.chunk","choices":[{"delta":{},"finish_reason":"stop"}]}

data: [DONE]
```

**Decoding strategies:**
| Strategy | Description | Best For |
|----------|-------------|----------|
| `greedy` | Always pick highest probability token | Deterministic tasks, code |
| `beam_search` | Maintain N hypotheses in parallel | Translation, summarization |
| `sampling` | Top-k + top-p + temperature sampling | Creative writing, chat |

---

## Models

### GET /models

List all loaded models.

**Response:**
```json
{
  "models": ["llm-playground-7b", "llm-playground-1b"]
}
```

### POST /models/load

Load a model into memory for inference.

**Request:**
```json
{
  "model_name": "llm-playground-7b",
  "model_path": "./checkpoints/pretrain/best",
  "tokenizer_path": "./tokenizers/bpe_50k",
  "dtype": "float16",
  "device": "cuda"
}
```

### DELETE /models/{model_name}

Unload a model from memory to free GPU VRAM.

### GET /models/{model_name}/info

Get detailed information about a loaded model (parameters, architecture, memory usage).

---

## Evaluation

### POST /evaluation/run

Run an evaluation benchmark on a loaded model.

**Request:**
```json
{
  "model": "llm-playground-7b",
  "benchmark": "mmlu",
  "subset": "all",
  "max_samples": 1000
}
```

### GET /evaluation/leaderboard

Get the ELO leaderboard from human evaluations.

---

## Health

### GET /health

Liveness check. Returns 200 if the process is alive.

### GET /ready

Readiness check. Returns 200 if the app is ready to serve traffic
(model loaded, dependencies connected).

### GET /health/detailed

Detailed health with component status (redis, model registry).

### GET /metrics

Prometheus-format metrics for monitoring.

---

## Error Responses

All errors return a consistent JSON structure:

```json
{
  "error": "Error type",
  "detail": "Human-readable description",
  "error_id": "uuid for log correlation",
  "path": "/api/v1/chat/completions"
}
```

| Status | Meaning |
|--------|---------|
| 400 | Bad request (invalid input) |
| 401 | Authentication required or failed |
| 403 | Authenticated but not authorized |
| 404 | Resource not found |
| 429 | Rate limit exceeded |
| 500 | Internal server error (check error_id in logs) |
| 503 | Service unavailable (model not loaded) |

---

## Rate Limits

| Tier | Requests/hour | Endpoints |
|------|---------------|-----------|
| Free | 100 | All |
| Pro | 1,000 | All |
| Enterprise | 10,000 | All |

Rate limit headers:
```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 87
X-RateLimit-Reset: 1697302800
```
