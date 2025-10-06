FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    curl \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y \
    nodejs \
    && rm -rf /var/lib/apt/lists/*

COPY package*.json ./
RUN npm install

COPY --from=docker.io/astral/uv:latest /uv /uvx /bin/
COPY pyproject.toml uv.lock .
# Create a project-local virtualenv and sync dev deps
RUN uv sync --frozen --group dev

# Ensure the venv Python is used for subsequent commands and at runtime
ENV PATH=/app/.venv/bin:$PATH

COPY . .

ENV DEV=true

ENTRYPOINT ["python", "run-dev.py"]
