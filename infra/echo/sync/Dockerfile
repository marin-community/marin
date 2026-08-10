FROM python:3.12-slim

# Progress lines must reach Cloud Logging as they happen, not on process exit.
ENV PYTHONUNBUFFERED=1

RUN pip install --no-cache-dir \
    "cloud-sql-python-connector[pg8000]>=1.9" "sqlalchemy>=2" "pgvector>=0.3" \
    "fastembed>=0.8,<0.9"

WORKDIR /app
COPY schema.py .
COPY search_config.py .
COPY repository_files.py .
COPY sync/github_repository.py .
COPY sync/main.py .

# Bake the passage embedding model into the image so hourly runs do not download it.
RUN python -c "import search_config; from fastembed import TextEmbedding; TextEmbedding(search_config.EMBED_MODEL)"

CMD ["python", "main.py"]
