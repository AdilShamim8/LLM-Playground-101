# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| 1.2.x   | ✅        |
| < 1.2   | ❌        |

## Reporting a Vulnerability

If you discover a security vulnerability, please:

1. **DO NOT** open a public GitHub issue
2. Email: security@example.com (replace with real address)
3. Include:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact
   - Suggested fix (if any)

We will respond within 48 hours and provide a timeline for a fix.

## Security Measures Implemented

### Authentication & Authorization
- JWT-based authentication (HS256, 24h expiry)
- API key support (hashed with bcrypt, stored in DB)
- Role-based access control (user / admin)
- Token revocation via blacklist

### Input Validation
- Pydantic models on all API inputs (type + range validation)
- Request size limits (50MB max, enforced by nginx)
- Content length validation on all string fields

### Transport Security
- TLS 1.2+ enforced (nginx terminates TLS)
- HSTS header in production (1 year, include subdomains)
- HTTP→HTTPS redirect in production

### Application Security
- `torch.load(weights_only=True)` — prevents pickle RCE
- Non-root Docker user (UID 1000)
- No hardcoded secrets (fail-fast if JWT_SECRET_KEY missing)
- DEV_MODE defaults to false (must explicitly enable)
- Security headers (X-Frame-Options, X-Content-Type-Options, XSS-Protection)

### Rate Limiting
- Per-user rate limiting (configurable, default 100 req/hour)
- Sliding window algorithm
- Separate limits for chat vs general API

### Dependency Security
- Pinned exact dependency versions (reproducible builds)
- Pre-commit hooks prevent secrets in commits
- `.gitignore` prevents `.env` and checkpoints from being committed

## Security Checklist for Production

- [ ] Generate JWT_SECRET_KEY: `python -c "import secrets; print(secrets.token_urlsafe(32))"`
- [ ] Set DEV_MODE=false
- [ ] Set ENVIRONMENT=production
- [ ] Set CORS_ORIGINS to explicit allowed origins
- [ ] Set POSTGRES_PASSWORD to a strong random value
- [ ] Configure TLS certificates (Let's Encrypt or commercial)
- [ ] Enable WAF rules (Cloudflare / AWS WAF)
- [ ] Set up DDoS protection
- [ ] Enable audit logging
- [ ] Configure database backups (daily, 30-day retention)
- [ ] Set up secret rotation policy
- [ ] Run dependency scan (Dependabot / Snyk)
- [ ] Run container image scan (Trivy / Clair)
- [ ] Penetration testing before launch
