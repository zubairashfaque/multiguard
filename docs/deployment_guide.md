# Deployment Guide

## Local Serving

```bash
# Start FastAPI server
make serve

# API available at http://localhost:8000
# Docs at http://localhost:8000/docs
```

## Docker Deployment

```bash
# Full stack (API + MLflow + Redis + Prometheus + Grafana)
make serve-docker

# Development mode (with hot reload)
docker-compose -f infrastructure/docker-compose.dev.yml up
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/ready` | GET | Readiness check |
| `/api/v1/predict` | POST | Classify content |
| `/api/v1/embed` | POST | Generate embeddings |
| `/api/v1/explain` | POST | Explainability output |

## Monitoring

- **Grafana:** http://localhost:3000 (admin/admin)
- **Prometheus:** http://localhost:9090
- **MLflow:** http://localhost:5000
