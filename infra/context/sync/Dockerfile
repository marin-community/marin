FROM python:3.12-slim

RUN pip install --no-cache-dir "pg8000>=1.31"

WORKDIR /app
COPY main.py .

CMD ["python", "main.py"]
