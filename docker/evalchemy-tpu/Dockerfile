# syntax=docker/dockerfile:1
#
# :evalchemy-tpu — TPU downstream-eval image for the marin SFT launcher.
#
# Stack: FROM the OT-Agent :tpu image (vLLM-TPU 0.20.0 PyTorch/XLA+JAX, already the known-good
# engine the eval-agentic-launch-iris TPU path runs on) + the marin-community/evalchemy fork
# (MATH500/AIME24/gsm8k + boxed-answer graders + lm_eval_compat shim) + lm-eval EleutherAI v0.4.12.
#
# WHY FROM :tpu (the vLLM-TPU derisk): reusing that base means vllm-tpu==0.20.0 + tpu-inference==0.20.0
# are inherited BYTE-IDENTICAL to the validated Iris TPU serving stack (OT-Agent pyproject.toml
# [datagen-tpu] extra, lines 89-98). No unproven vLLM-TPU port — we inherit the engine already
# proven on Iris.
#
# BUILD: this is a CPU-only linux/amd64 layer (no CUDA/nvcc, no big-memory wheel step) on top of the
# pullable :tpu base -> the Mac (arm64) CAN cross-build + push it via `docker buildx --platform
# linux/amd64 --push` (QEMU emulation of the base's RUN python-imports works; minutes end-to-end).
# Verified 2026-07-15: built + pushed :evalchemy-tpu-<gitsha> entirely on the M4 Max Mac. (This differs
# from gpu-rl, which genuinely needs the in-cluster kaniko path — ~512GB RAM + hours of nvcc. The base
# :tpu image is built in-cluster only because there is no in-cluster buildx primitive, NOT because the
# Mac cannot cross-build a CPU-only amd64 image.) The in-cluster Iris kaniko job on CoreWeave
# cw-us-east-02a (skill build-tpu-image-iris) remains a valid fallback:
#
#   1. PINNED-first: push :evalchemy-tpu-<gitsha> (immutable), promote floating :evalchemy-tpu via
#      `crane tag` only AFTER a live TPU smoke (evalchemy imports; `python -m eval.eval --model vllm`
#      serves on a TPU slice; MATH-500 seed42 runs).
#   2. Capture the DIGEST and pin it in marin_evalchemy_tpu.py (EVALCHEMY_IMAGE @sha256:<digest>).
#
# The evalchemy fork commit below is the RUNTIME grader/task pin. Bumping EVALCHEMY_COMMIT changes this
# layer's cache key (guaranteed rebuild) and --force-reinstall rebuilds from the exact commit,
# regardless of any cached wheel (mirrors the :tpu image's HARBOR_COMMIT discipline).
#
FROM ghcr.io/open-thoughts/openthoughts-agent:tpu

ENV DEBIAN_FRONTEND=noninteractive

# The venv baked by the :tpu base (carries vllm-tpu 0.20.0 + tpu-inference 0.20.0).
ENV VIRTUAL_ENV=/opt/openthoughts/.venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
ENV UV_HTTP_TIMEOUT=300
ENV UV_RETRIES=5

# marin-community/evalchemy branch `main` @ a84543c (current main HEAD — the consolidated fork carrying
# PRs #14/#11/#13/#16; supersedes the old c4598e64 pin). Pinned to the exact main-HEAD sha so the
# :evalchemy-tpu-<gitsha> image stays immutable/reproducible while tracking the de-pinned main branch.
ARG EVALCHEMY_REPO=https://github.com/marin-community/evalchemy.git
ARG EVALCHEMY_COMMIT=a84543cda02f76bd4db0ae9a84039d792d3ed6e7

WORKDIR /opt/eval

# 1. Clone the evalchemy fork at the pinned commit (it has local file deps -> editable install, not a
#    wheel; graders live in the tree). --force via a fresh clone each build (commit pin is the cache key).
RUN git clone "${EVALCHEMY_REPO}" evalchemy && \
    cd evalchemy && git checkout "${EVALCHEMY_COMMIT}" && \
    echo "pinned evalchemy commit: $(git rev-parse HEAD)"

# 2. Install lm-eval v0.4.12 BARE (no [vllm] extra — its vllm>=0.18 pin would fight the base's
#    vllm-tpu 0.20.0). vLLM comes from the base image; lm-eval must NOT pull its own.
RUN . "${VIRTUAL_ENV}/bin/activate" && \
    uv pip install --no-cache-dir \
        "lm-eval @ git+https://github.com/EleutherAI/lm-evaluation-harness@v0.4.12"

# 3. Editable-install the evalchemy fork WITHOUT deps (--no-deps: keep the base's vllm-tpu + the pinned
#    lm-eval; evalchemy's own requirements would otherwise try to re-resolve vLLM/torch).
RUN . "${VIRTUAL_ENV}/bin/activate" && \
    uv pip install --no-deps --force-reinstall -e /opt/eval/evalchemy && \
    python -c "import eval, lm_eval, importlib.metadata as m; print('baked lm-eval', m.version('lm-eval'))"

# 3b. Curator backend import-deps (the eval.eval CLI top-level does
#     `from eval.chat_benchmarks.curator_lm import CuratorAPIModel` UNCONDITIONALLY, which needs
#     `bespokelabs.curator` -> without this the CLI cannot even `--help`). We NEVER use --model curator
#     (the launcher drives --model vllm), so we only need the IMPORT to succeed. Install curator + sandbox
#     with --no-deps (their full dep set would DOWNGRADE the base's derisked pins — datasets 5.0->3.6,
#     huggingface-hub 1.22->1.16, protobuf 6->5, pydantic, openai) and add ONLY the light import-time deps
#     (nest_asyncio/litellm/instructor). Verified: this leaves vllm-tpu/tpu-inference/torch/transformers/
#     datasets/huggingface-hub/numpy/pydantic/protobuf BYTE-IDENTICAL to the :tpu base (derisk preserved).
RUN . "${VIRTUAL_ENV}/bin/activate" && \
    uv pip install --no-cache-dir nest_asyncio litellm instructor && \
    uv pip install --no-cache-dir --no-deps bespokelabs-curator bespokelabs-sandbox && \
    python -c "import eval.eval; from bespokelabs import curator; print('eval.eval CLI imports OK (curator backend registered)')" && \
    python -c "import importlib.metadata as m; assert m.version('vllm-tpu')=='0.20.0', m.version('vllm-tpu'); print('derisk intact: vllm-tpu', m.version('vllm-tpu'))"

# 4. Pre-cache nltk punkt/punkt_tab so IFEval/AIME24 don't abort mid-run trying to download offline.
RUN . "${VIRTUAL_ENV}/bin/activate" && \
    python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')" || \
    echo "WARN: nltk pre-cache skipped (verify at runtime)"

# 5. Writable cache redirects (the eval workers run read-only-HOME on some clusters; mirror the
#    delphi_eval.sbatch redirects so vLLM/triton/flashinfer/XDG caches land somewhere writable).
ENV HF_HOME=/opt/eval/.cache/hf \
    TRITON_CACHE_DIR=/opt/eval/.cache/triton \
    VLLM_CACHE_ROOT=/opt/eval/.cache/vllm \
    XDG_CACHE_HOME=/opt/eval/.cache/xdg
RUN mkdir -p /opt/eval/.cache/{hf,triton,vllm,xdg}

# JAX onto TPU (inherited from base, re-asserted for clarity).
ENV JAX_PLATFORMS=tpu

CMD ["bash"]
