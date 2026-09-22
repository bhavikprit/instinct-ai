# Minimal production container for Instinct AI Gateway
FROM python:3.11-slim as builder

WORKDIR /app

# Install build dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Copy package metadata and source code
COPY pyproject.toml README.md ./
COPY instinct/ ./instinct/
COPY reflex/ ./reflex/

# Install zero-dependency instinct-ai
RUN pip install --no-cache-dir .

# Final runtime image
FROM python:3.11-slim

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin/instinct /usr/local/bin/instinct

# Non-root user for security
RUN useradd -m -u 1000 reflexuser
USER reflexuser

EXPOSE 8000

# Kubernetes / Docker health check
HEALTHCHECK --interval=15s --timeout=3s --retries=3 \
  CMD python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

ENTRYPOINT ["instinct", "serve-api", "--host", "0.0.0.0", "--port", "8000"]
