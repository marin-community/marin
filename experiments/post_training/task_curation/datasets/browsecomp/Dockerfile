FROM python:3.12-slim-trixie@sha256:5e2bdb1b0d542d5039cffb8181c65930093949194a65c365dfb1b0fe13a4fa5b
RUN apt-get update && apt-get install -y --no-install-recommends openjdk-21-jre-headless gcc g++ && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir torch==2.7.0+cpu --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir pyserini==1.2.0 transformers==4.53.2 huggingface-hub==0.33.4 pydantic==2.9.2
RUN pip install --no-cache-dir openai==1.97.0
COPY browsecomp /opt/browsecomp
ENV PYTHONPATH=/opt HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64
WORKDIR /app
