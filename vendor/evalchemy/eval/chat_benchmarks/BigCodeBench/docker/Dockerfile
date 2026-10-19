FROM nvcr.io/nvidia/pytorch:24.09-py3

RUN apt-get update --yes --quiet \
    && apt-get upgrade --yes --quiet \
    && apt-get install git wget curl unzip software-properties-common --yes --quiet

RUN pip install --upgrade pip

RUN git clone https://github.com/mlfoundations/evalchemy.git /evalchemy

RUN pip install -I --timeout 2000 -r https://raw.githubusercontent.com/bigcode-project/bigcodebench/refs/heads/main/Requirements/requirements.txt \
    && pip install --no-cache-dir ninja packaging psutil 
    && pip install --no-cache-dir -I --timeout 2000 -r https://raw.githubusercontent.com/bigcode-project/bigcodebench/refs/heads/main/Requirements/requirements-eval.txt \
    && cd /evalchemy && pip install --no-cache-dir -e . && pip install -e eval/chat_benchmarks/alpaca_eval \
    && pip install lm-eval[vllm]@git+https://github.com/EleutherAI/lm-evaluation-harness@v0.4.12 \
    && pip install "opencv-python-headless<4.3" \
    && MAX_JOBS=10 pip install -U flash-attn --no-build-isolation

WORKDIR /app