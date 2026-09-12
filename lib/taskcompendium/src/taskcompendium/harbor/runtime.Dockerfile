FROM python:3.12-slim@sha256:78387bc3881b8273120a12ebe6c1ab22b018ccc2c9adf565ae1ac9b536e184ea
RUN apt-get update && apt-get install -y --no-install-recommends g++ bash tmux && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir msgspec==0.19.0 tomlkit==0.13.3 fsspec==2025.3.0 pyarrow==23.0.1 openai==2.24.0 pytest==8.4.2 pytest-json-report==1.5.0
RUN pip install --no-cache-dir mini-swe-agent==2.4.6
RUN mkdir -p /root/.local/bin && touch /root/.local/bin/env
RUN mkdir -p /opt/tasktrove-pytest/bin && ln -s /usr/local/bin/python3 /opt/tasktrove-pytest/bin/python
WORKDIR /app
