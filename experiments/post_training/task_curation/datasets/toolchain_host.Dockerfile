# The host programs a WORKER-placement grader environment assumed the worker image provides.
# The composed toolchain image bakes exactly this set, so the same host programs the role
# used on the worker are inside the image, and image backends can host the grader.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    build-essential ca-certificates curl ffmpeg git libjemalloc2 lldb openssh-client unzip \
    && rm -rf /var/lib/apt/lists/*
