FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1

RUN pip install --no-cache-dir \
    "fastapi>=0.115" "uvicorn>=0.30" \
    "cloud-sql-python-connector[pg8000]>=1.9" "sqlalchemy>=2" "pgvector>=0.3" \
    "fastembed>=0.3"

WORKDIR /app
COPY schema.py .
COPY api/app.py .

# Bake the query embedding model into the image so the first request is not a cold download.
RUN python -c "from fastembed import TextEmbedding; TextEmbedding('BAAI/bge-small-en-v1.5')"

CMD exec uvicorn app:app --host 0.0.0.0 --port ${PORT:-8080}
