# Production Deployment Guide — Scaling to 100–200M MAU

> **Target**: 150M monthly active users (MAU), with peak capacity for 2,600 requests/second.
> **Reference**: ChatGPT has ~1B MAU. This guide targets ~15% of that scale.
> **Status**: Target architecture. The current codebase handles ~10 concurrent users.

---

## 1. Traffic Estimates

| Metric | Value | How Calculated |
|--------|-------|----------------|
| Monthly active users (MAU) | 150M | Target |
| Daily active users (DAU) | 15M | ~10% of MAU (industry avg) |
| Messages per DAU | 5 | ChatGPT avg |
| Daily messages | 75M | DAU × messages |
| Average messages/sec | ~870 | 75M / 86,400s |
| Peak messages/sec | ~2,600 | 3× average |
| Avg tokens per request | ~500 | prompt + response |
| Peak tokens/sec | ~1.3M | peak msg/s × tokens |
| Tokens/sec per A100 (7B, vLLM) | ~5,000 | benchmark |
| **Min GPUs for inference** | **~260** | 1.3M / 5,000 |
| With 2× redundancy | **~520 GPUs** | fault tolerance |

---

## 2. Cost Estimates

### Inference (GPU Compute)

| Item | Value |
|------|-------|
| A100 80GB hourly (cloud, reserved 1yr) | ~$2.50/hr |
| GPUs needed (with redundancy) | 520 |
| Monthly GPU cost | **$936,000/month** |
| Annual GPU cost | **~$11.2M/year** |

### Additional Infrastructure

| Component | Monthly Cost (est.) |
|-----------|-------------------|
| Load balancers (global, multi-region) | $5,000 |
| Redis Cluster (rate limit, cache) | $3,000 |
| PostgreSQL (managed, HA, replicas) | $5,000 |
| S3/GCS storage (model weights, logs) | $2,000 |
| Kubernetes cluster (EKS/GKE) + nodes | $10,000 |
| CDN (static assets) | $3,000 |
| Monitoring (Datadog/New Relic) | $8,000 |
| Networking (egress, VPC peering) | $15,000 |
| **Total non-GPU infra** | **~$51,000/month** |
| **Total (GPU + infra)** | **~$987,000/month** |
| **Annual total (inference only)** | **~$11.8M/year** |

### Engineering Team (Annual)

| Role | Headcount | Fully-loaded cost |
|------|-----------|------------------|
| ML Engineer | 4 | $800K |
| Infra/DevOps Engineer | 3 | $600K |
| Backend Engineer | 3 | $550K |
| Frontend Engineer | 2 | $350K |
| SRE (24/7 on-call) | 2 | $450K |
| Eng Manager | 1 | $200K |
| **Total** | **15** | **~$2.95M/year** |

### Grand Total: ~$14.8M/year

---

## 3. Production Architecture

```
                    ┌─────────────────────────────────────────────────────┐
                    │              Global Load Balancer                   │
                    │         (AWS ALB / Cloudflare / GCP LB)             │
                    │     Geo-DNS → Nearest Region (us/eu/apac)           │
                    └────────────────────────┬────────────────────────────┘
                                             │
                    ┌────────────────────────┴────────────────────────────┐
                    │              API Gateway / CDN                       │
                    │  (Static assets via CDN, API via ALB)                │
                    │  Rate limiting (Redis), Auth (JWT), WAF              │
                    └────────────────────────┬────────────────────────────┘
                                             │
              ┌──────────────────────────────┼──────────────────────────────┐
              │                              │                              │
    ┌─────────┴─────────┐      ┌────────────┴───────────┐      ┌──────────┴──────────┐
    │   Inference Pool 1  │      │   Inference Pool 2      │      │  Inference Pool N     │
    │  (us-east-1)        │      │  (eu-west-1)            │      │  (ap-southeast-1)    │
    │                     │      │                         │      │                      │
    │  vLLM Ray Serve     │      │  vLLM Ray Serve         │      │  vLLM Ray Serve       │
    │  8×A100 per node    │      │  8×A100 per node        │      │  8×A100 per node     │
    │  Tensor Parallel=4  │      │  Tensor Parallel=4      │      │  Tensor Parallel=4   │
    │  Continuous batching │      │  Continuous batching    │      │  Continuous batching │
    │  32 nodes = 256 GPU  │      │  16 nodes = 128 GPU     │      │  16 nodes = 128 GPU   │
    └─────────┬───────────┘      └────────────┬───────────┘      └──────────┬───────────┘
              │                                │                             │
              └───────────────────────────────┼─────────────────────────────┘
                                              │
                    ┌─────────────────────────┴─────────────────────────┐
                    │              Shared State Layer                     │
                    │                                                     │
                    │  Redis Cluster (rate limit, session, KV cache)       │
                    │  PostgreSQL (users, API keys, conversation log)     │
                    │  S3/GCS (model weights, checkpoints)                 │
                    │  Kafka (event stream, analytics, audit log)         │
                    └─────────────────────────────────────────────────────┘
```

---

## 4. Components Required (Not in Current Codebase)

| Component | Purpose | Status |
|-----------|---------|--------|
| **vLLM / TGI** | High-throughput inference with continuous batching | Not integrated |
| **Redis Cluster** | Distributed rate limit + KV cache | In docker-compose (single node) |
| **Ray Serve** | Multi-node model orchestration | Not present |
| **Kubernetes** | Container orchestration, auto-scaling | Not present |
| **Prometheus + Grafana** | Metrics + dashboards | Not present |
| **OpenTelemetry** | Distributed tracing | Not present |
| **Kafka / SQS** | Async job queue | Celery in deps, no worker |
| **MLflow** | Model registry + experiment tracking | Not present |
| **S3 / GCS** | Model weight storage | Local filesystem only |
| **WAF + DDoS protection** | Security (Cloudflare / AWS WAF) | Not present |
| **Terraform / Pulumi** | Infrastructure as Code | Not present |
| **ArgoCD / Flux** | GitOps deployment | Not present |

---

## 5. Migration Path (Current → Production)

### Phase 1: Fix Critical Bugs (Done in this audit)
- Security: secrets, CORS, torch.load, DEV_MODE
- Dockerfile: non-root, healthcheck
- docker-compose: healthchecks, env-based config
- Dependencies: version pinning
- CI: GitHub Actions

### Phase 2: Integrate vLLM (2-4 weeks)
- Replace custom inference with vLLM
- vLLM provides: continuous batching, PagedAttention, tensor parallelism
- Expected improvement: 6-19× throughput on same hardware
- Add vLLM Docker image to docker-compose

### Phase 3: Add Redis-backed Rate Limiting (1 week)
- Replace in-memory dict with Redis sliding window
- Enables multi-worker and multi-server rate limiting
- Use Redis Cluster for HA in production

### Phase 4: Add PostgreSQL User Storage (1-2 weeks)
- Move API keys from in-memory dict to database
- Add Alembic migrations
- Add user registration + API key management endpoints

### Phase 5: Container Orchestration (2-3 weeks)
- Write Kubernetes manifests (Deployment, Service, HPA, Ingress)
- Set up Helm chart for templating
- Configure HPA on GPU utilization
- Set up cluster autoscaler for node-level scaling

### Phase 6: Observability (1-2 weeks)
- Add Prometheus metrics endpoint (`/metrics`)
- Add OpenTelemetry tracing
- Set up Grafana dashboards (QPS, latency, GPU util, error rate)
- Set up alerting (PagerDuty/Opsgenie)

### Phase 7: Multi-Region (4-6 weeks)
- Deploy to 3 regions (us-east, eu-west, ap-southeast)
- Set up geo-DNS routing (Route 53 / Cloudflare)
- Configure cross-region replication for PostgreSQL
- Set up S3 cross-region replication for model weights

### Phase 8: Auto-Scaling (2-3 weeks)
- GPU auto-scaling based on request queue depth
- Predictive scaling based on time-of-day patterns
- Model warm-up pools for fast scale-out
- Graceful drain on scale-in

---

## 6. Key Metrics to Monitor

| Metric | Target | Alert Threshold |
|--------|--------|-----------------|
| P50 latency | < 500ms | > 1s |
| P99 latency | < 2s | > 5s |
| Error rate | < 0.1% | > 1% |
| GPU utilization | 70-85% | < 50% (over-provisioned) or = 100% (saturated) |
| Request queue depth | < 10 | > 100 |
| Token throughput | varies | drops > 20% from baseline |

---

## 7. Security Checklist for Production

- [ ] All secrets in a vault (AWS Secrets Manager, HashiCorp Vault)
- [ ] TLS everywhere (no HTTP in transit)
- [ ] WAF rules for SQL injection, XSS, rate limiting
- [ ] DDoS protection (Cloudflare, AWS Shield)
- [ ] Regular dependency scanning (Dependabot, Snyk)
- [ ] Container image scanning (Trivy, Clair)
- [ ] Penetration testing before launch
- [ ] GDPR/CCPA compliance for user data
- [ ] Audit logging for all API calls
- [ ] Backup and disaster recovery plan
