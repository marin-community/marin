FROM python:3.12-slim

# Progress lines must reach Cloud Logging as they happen, not on process exit.
ENV PYTHONUNBUFFERED=1

RUN pip install --no-cache-dir "pg8000>=1.31" "sqlalchemy>=2" "pgvector>=0.3"

WORKDIR /app
# Build context is infra/echo so the sync shares schema.py with migrations.
COPY schema.py .
COPY sync/main.py .

CMD ["python", "main.py"]
