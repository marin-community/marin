FROM ubuntu:24.04

WORKDIR /app
RUN mkdir -p /output && chmod 777 /output

RUN apt-get update && apt-get install -y python3 python3-pip && rm -rf /var/lib/apt/lists/*

# Pre-install test deps so the verifier works fully offline
RUN pip3 install --break-system-packages pytest==8.4.1 pytest-json-ctrf==0.3.5
