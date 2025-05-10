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

COPY pyproject.toml .
COPY poetry.lock .
RUN pip install poetry && \
    poetry config virtualenvs.create false && \
    poetry install --no-root

COPY . .

ENV DEV=true

ENTRYPOINT ["python", "run-dev.py"]
