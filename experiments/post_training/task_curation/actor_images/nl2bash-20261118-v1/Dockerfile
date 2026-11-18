FROM ubuntu:24.04@sha256:78d4744ead1f5b19b04d0df558ab182a5300903900755e515b2ca4d10eb2044d
LABEL org.opencontainers.image.base.digest=sha256:78d4744ead1f5b19b04d0df558ab182a5300903900755e515b2ca4d10eb2044d
RUN mkdir -p /workspace
RUN (cd /workspace && export DEBIAN_FRONTEND=noninteractive; mkdir -p /output /logs/verifier && chmod 777 /output /logs/verifier)
RUN (cd /workspace && export DEBIAN_FRONTEND=noninteractive; apt-get update && apt-get install -y --no-install-recommends bash build-essential ca-certificates curl git jq less python3 python3-dev python3-pip python3-venv tmux unzip vim wget && rm -rf /var/lib/apt/lists/*)
