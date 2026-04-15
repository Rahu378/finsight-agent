# FinSight AI Agent — Production Dockerfile
# Multi-stage build: keeps final image lean (~200MB vs ~900MB)

# ---- Stage 1: Build dependencies ----
FROM python:3.11-slim AS builder

WORKDIR /build

# Install system deps needed to compile some packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


# ---- Stage 2: Runtime ----
FROM python:3.11-slim AS runtime

# Non-root user for security (Capital One security standards)
RUN groupadd -r finsight && useradd -r -g finsight finsight

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application source
COPY src/ ./src/
COPY .env.example .env.example

# Set ownership
RUN chown -R finsight:finsight /app
USER finsight

# Expose API port
EXPOSE 8000

# Health check — ECS / K8s will use this
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

# Start the API
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
