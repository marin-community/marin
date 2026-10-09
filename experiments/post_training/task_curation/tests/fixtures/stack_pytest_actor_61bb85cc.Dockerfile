FROM ubuntu:24.04

WORKDIR /app

# Install python, pip, and bsdutils (for script command needed by terminus-2 agent)
RUN apt-get update && apt-get install -y python3 python3-pip python3-venv bsdutils && rm -rf /var/lib/apt/lists/*
RUN python3 -m venv --system-site-packages /opt/tasktrove-pytest && /opt/tasktrove-pytest/bin/pip install --no-cache-dir pytest pytest-json-report
