# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PORT=8080

# Cloud Run injects $PORT.
CMD uvicorn main:app --host 0.0.0.0 --port $PORT
