FROM python:3.12-slim

LABEL org.opencontainers.image.title="VecGrep Action"
LABEL org.opencontainers.image.description="Semantic code search for CI/CD pipelines"
LABEL org.opencontainers.image.source="https://github.com/VecGrep/action"
LABEL org.opencontainers.image.licenses="MIT"

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install vecgrep pinned to Python 3.12
RUN pip install --no-cache-dir "vecgrep>=1.5.0"

# Copy action entrypoint
COPY entrypoint.py /entrypoint.py

ENTRYPOINT ["python", "/entrypoint.py"]
