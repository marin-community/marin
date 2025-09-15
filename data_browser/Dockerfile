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

COPY pyproject.toml uv.lock .
RUN curl -LsSf https://astral.sh/uv/install.sh | sh -s -- -y \
    && mv /root/.cargo/bin/uv /usr/local/bin/uv \
    && uv pip install --system -r uv.lock

COPY . .

ENV DEV=true

ENTRYPOINT ["python", "run-dev.py"]
