FROM python:3.11-slim-bookworm
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
RUN mkdir -p /app /tests /logs/verifier && chmod 755 /app /tests
RUN pip install --no-cache-dir sympy==1.13.3 antlr4-python3-runtime==4.11
WORKDIR /app
