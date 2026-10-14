# System Architecture — LLM Playground 101

> **Target scale**: 100–200M monthly active users (MAU)
> **Current state**: Single-node, ~10 concurrent users
> **Status**: Target architecture — see migration phases below

---

## 1. High-Level Architecture

```
                          ┌──────────────────────────┐
                          │     Users / Clients      │
                          │  (Web, Mobile, API SDK)   │
                          └────────────┬──────────────┘
                                       │ HTTPS
                          ┌────────────▼──────────────┐
                          │   CDN / WAF / DDoS Protect │
                          │  (Cloudflare / AWS Shield)  │
                          └────────────┬──────────────┘
                                       │
                          ┌────────────▼──────────────┐
                          │    Global Load Balancer    │
                          │  (Geo-DNS → nearest region)│
                          └────────────┬──────────────┘
                                       │
                    ┌──────────────────┼──────────────────┐
                    │                  │                  │
             ┌──────▼──────┐    ┌──────▼──────┐    ┌──────▼──────┐
             │  Region 1    │    │  Region 2    │    │  Region 3    │
             │  (us-east)   │    │  (eu-west)   │    │  (apac)      │
             │              │    │              │    │              │
             │  Nginx LB    │    │  Nginx LB    │    │  Nginx LB    │
             │  ↓           │    │  ↓           │    │  ↓           │
             │  API ×N      │    │  API ×N      │    │  API ×N      │
             │  ↓           │    │  ↓           │    │  ↓           │
             │  vLLM Pool   │    │  vLLM Pool   │    │  vLLM Pool   │
             │  (Ray Serve) │    │  (Ray Serve) │    │  (Ray Serve) │
             └──────┬──────┘    └──────┬──────┘    └──────┬──────┘
                    │                  │                  │
                    └──────────────────┼──────────────────┘
                                       │
                          ┌────────────▼──────────────┐
                          │     Shared State Layer      │
                          │                             │
                          │  ┌─────────┐  ┌───────────┐ │
                          │  │  Redis  │  │ PostgreSQL│ │
                          │  │ Cluster │  │  (HA + RR)│ │
                          │  │(rate,   │  │(users,    │ │
                          │  │ cache)  │  │ conv logs)│ │
                          │  └─────────┘  └───────────┘ │
                          │  ┌─────────┐  ┌───────────┐ │
                          │  │  S3/GCS │  │  Kafka    │ │
                          │  │(models) │  │(events)   │ │
                          │  └─────────┘  └───────────┘ │
                          └─────────────────────────────┘
```

---

## 2. Component Breakdown

### 2.1 Edge Layer (CDN + WAF)

| Component | Purpose | Tool |
|-----------|---------|------|
| CDN | Cache static assets (UI, JS, CSS) globally | Cloudflare / CloudFront |
| WAF | Block SQL injection, XSS, DDoS | Cloudflare WAF / AWS WAF |
| DDoS Protection | Absorb traffic spikes | AWS Shield / Cloudflare |
| TLS Termination | SSL/TLS at the edge | CDN handles |

### 2.2 API Gateway Layer

| Component | Purpose | Tool |
|-----------|---------|------|
| Load Balancer | Distribute traffic across API instances | AWS ALB / Nginx |
| Rate Limiting | Per-user request limits | Redis + Nginx |
| Authentication | JWT validation, API key lookup | FastAPI middleware |
| Request Routing | Route /api/v1/chat → inference, /api/v1/models → registry | FastAPI routers |

### 2.3 Application Layer (API)

| Component | Purpose | Tech |
|-----------|---------|------|
| Web Framework | HTTP request handling | FastAPI + uvicorn + uvloop |
| Workers | Async event loop per process | 4 workers × 2 replicas = 8 processes |
| Middleware | Auth, CORS, security headers, request ID, metrics | Custom Starlette middleware |
| Background Tasks | Async job queue (evaluations, training) | Celery + Redis broker |

### 2.4 Inference Layer (vLLM + Ray Serve)

This is the most critical layer for scale. The current custom inference
loop (PyTorch forward pass) is replaced with vLLM for production:

| Component | Purpose | Why |
|-----------|---------|-----|
| vLLM | High-throughput inference server | Continuous batching, PagedAttention |
| Ray Serve | Multi-node model orchestration | Auto-scaling, fault tolerance |
| Tensor Parallelism | Split model across GPUs | Required for 70B+ models |
| Continuous Batching | Batch tokens from multiple requests | 6-19× throughput improvement |
| KV Cache Management | PagedAttention for memory efficiency | 3× more concurrent requests |

**Why not custom inference?**
The current `generation/sampler.py` processes one request at a time.
GPU utilization is 5-15%. vLLM achieves 80-95% via continuous batching.

### 2.5 Data Layer

| Component | Purpose | HA Strategy |
|-----------|---------|-------------|
| PostgreSQL | Users, API keys, conversations, billing | Primary + 2 read replicas, automated failover |
| Redis Cluster | Rate limiting, session cache, KV cache | 6-node cluster (3 primary + 3 replica) |
| S3 / GCS | Model weights, checkpoints, logs | Cross-region replication |
| Kafka | Event stream (generations, analytics) | 3-broker cluster, partitioned by user_id |

---

## 3. Request Flow (Chat Completion)

```
1. Client sends POST /api/v1/chat/completions
   ↓
2. CDN/WAF: TLS termination, DDoS check, rate limit
   ↓
3. Nginx: Load balance to least-busy API instance
   ↓
4. FastAPI middleware stack (in order):
   a. RequestIDMiddleware: assign X-Request-ID
   b. PrometheusMiddleware: record start time
   c. SecurityHeadersMiddleware: add security headers
   d. CORSMiddleware: check origin
   e. AuthMiddleware: validate JWT / API key → user_id
   f. RateLimitMiddleware: check Redis rate limit
   ↓
5. Route handler: /api/v1/chat/completions
   a. Validate request (Pydantic)
   b. Format messages → prompt
   c. Call vLLM via Ray Serve (async, thread pool)
   d. Stream tokens back via SSE
   ↓
6. Ray Serve: route to vLLM worker
   a. vLLM adds request to continuous batch
   b. GPU generates tokens (shared with other requests)
   c. Tokens streamed back
   ↓
7. Post-generation:
   a. Record metrics (Prometheus)
   b. Log generation (PostgreSQL generation_logs)
   c. Return response to client
```

---

## 4. Scaling Strategy

### 4.1 Horizontal Scaling (More Instances)

| Layer | Scaling Trigger | Scale Unit |
|-------|----------------|------------|
| API | CPU > 70% or request queue > 100 | +1 pod (HPA) |
| vLLM | GPU utilization > 85% or queue depth > 10 | +1 GPU node |
| Redis | Memory > 70% | Add shard |
| PostgreSQL | CPU > 80% or connections > 80% | Add read replica |

### 4.2 Vertical Scaling (Bigger Instances)

| Model Size | GPU per Instance | TP Size | Notes |
|------------|------------------|---------|-------|
| 7B FP16 | 1× A100 40GB | 1 | Single GPU |
| 13B FP16 | 1× A100 80GB | 1 | Single GPU |
| 70B FP16 | 2× A100 80GB | 2 | TP=2 |
| 70B INT4 | 1× A100 40GB | 1 | Quantized |
| 175B FP16 | 4× A100 80GB | 4 | TP=4 |

### 4.3 Auto-Scaling Rules (Kubernetes HPA)

```yaml
# Scale API pods based on CPU
apiVersion: autoscaling/v2
spec:
  minReplicas: 3
  maxReplicas: 50
  metrics:
    - type: Resource
      resource:
        name: cpu
        target: { type: Utilization, averageUtilization: 70 }
    - type: Pods
      pods:
        metric: { name: http_requests_in_progress }
        target: { type: AverageValue, averageValue: "10" }
```

---

## 5. Capacity Planning (150M MAU)

### Traffic Model

| Metric | Value |
|--------|-------|
| Monthly active users | 150M |
| Daily active users (10% of MAU) | 15M |
| Messages per DAU | 5 |
| Daily messages | 75M |
| Average messages/sec | ~870 |
| Peak messages/sec (3× avg) | ~2,600 |
| Avg tokens per request (prompt + response) | ~500 |
| Peak tokens/sec | ~1.3M |
| Tokens/sec per A100 (7B, vLLM) | ~5,000 |
| **Min GPUs for inference** | **~260** |
| With 2× redundancy | **~520 GPUs** |

### Cost Estimate

| Component | Monthly Cost |
|-----------|-------------|
| GPU compute (520 × A100) | $936,000 |
| Load balancers + CDN | $5,000 |
| Redis Cluster | $3,000 |
| PostgreSQL (managed HA) | $5,000 |
| S3 storage | $2,000 |
| Kubernetes cluster | $10,000 |
| Monitoring | $8,000 |
| Networking | $15,000 |
| **Total infrastructure** | **~$984,000/month** |
| **Annual** | **~$11.8M/year** |

---

## 6. Migration Phases

### Phase 1: Fix Critical Bugs ✅ (Done)
Security fixes, CI, Dockerfile, version pinning.

### Phase 2: Integrate vLLM (2-4 weeks)
Replace custom inference with vLLM for 6-19× throughput.

### Phase 3: Redis Rate Limiting (1 week)
Replace in-memory dict with Redis sliding window.

### Phase 4: PostgreSQL User Storage (1-2 weeks)
Move API keys from memory to database with Alembic migrations.

### Phase 5: Kubernetes (2-3 weeks)
Helm chart, HPA, cluster autoscaler, ingress controller.

### Phase 6: Observability (1-2 weeks)
Prometheus metrics endpoint (added), Grafana dashboards, alerting.

### Phase 7: Multi-Region (4-6 weeks)
Geo-DNS, cross-region DB replication, S3 replication.

### Phase 8: Auto-Scaling (2-3 weeks)
GPU auto-scaling on queue depth, predictive scaling, warm pools.

---

## 7. Technology Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Web framework | FastAPI | Async, OpenAPI docs, Pydantic validation |
| Inference server | vLLM | Continuous batching, PagedAttention, TP |
| Orchestration | Ray Serve | Multi-node, fault tolerance, auto-scaling |
| Container runtime | Docker + Kubernetes | Industry standard, portable |
| Database | PostgreSQL | ACID, HA, mature ecosystem |
| Cache/Rate limit | Redis Cluster | Sub-ms latency, HA |
| Message queue | Kafka | Ordered events, replay capability |
| Monitoring | Prometheus + Grafana | Industry standard, open source |
| Tracing | OpenTelemetry | Vendor-neutral standard |
| IaC | Terraform | Declarative, version-controlled infra |
| CI/CD | GitHub Actions | Integrated with repo, free for OSS |
