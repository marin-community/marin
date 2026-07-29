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
COPY sync/main.py .

# Bake the passage embedding model into the image so hourly runs do not download it.
RUN python -c "from fastembed import TextEmbedding; TextEmbedding('BAAI/bge-small-en-v1.5')"

CMD ["python", "main.py"]
