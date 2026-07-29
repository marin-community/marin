FROM node:22-slim AS dashboard

WORKDIR /build
COPY dashboard/package.json dashboard/package-lock.json ./
RUN npm ci
COPY dashboard/ ./
RUN npm run build

FROM python:3.12-slim AS runtime

ENV PYTHONUNBUFFERED=1

RUN pip install --no-cache-dir \
    "fastapi>=0.115" "uvicorn>=0.30" \
    "cloud-sql-python-connector[pg8000]>=1.9" "sqlalchemy>=2" "pgvector>=0.3" \
    "fastembed>=0.8,<0.9"

WORKDIR /app
COPY schema.py .
COPY hybrid_search.py .
COPY search_config.py .
COPY api/app.py .
COPY api/dashboard.py .
COPY --from=dashboard /build/dist /app/dist

# Bake the query embedding model into the image so the first request is not a cold download.
RUN python -c "import search_config; from fastembed import TextEmbedding; TextEmbedding(search_config.EMBED_MODEL)"

CMD ["sh", "-c", "exec uvicorn app:app --host 0.0.0.0 --port ${PORT:-8080}"]
