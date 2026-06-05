# Redirect every cache that vLLM/HF/Torch/Triton/CUDA writes off the home
# volume onto the model-tracing scratch volume, so home quota doesn't fill up
# and cause silent slowdowns or hard failures (e.g. Triton JIT writes failing
# with "Disk quota exceeded" mid-inference).
#
# Source from ~/.bashrc.user:
#   source /juice4/scr4/nlp/model-tracing/marin/experiments/downstream_scaling/evals/gpu/set_cache_env.sh
#
# Safe to re-source; mkdir -p is idempotent.

CACHE_ROOT=/juice4/scr4/nlp/model-tracing/cache

export HF_HOME="$CACHE_ROOT/huggingface"                  # model weights, tokenizers, datasets (umbrella for HF_HUB_CACHE etc.)
export TRITON_CACHE_DIR="$CACHE_ROOT/triton"              # Triton JIT-compiled kernels
export VLLM_CACHE_ROOT="$CACHE_ROOT/vllm"                 # vLLM persistent cache (torch.compile artifacts vLLM owns)
export TORCHINDUCTOR_CACHE_DIR="$CACHE_ROOT/torchinductor" # torch.compile / Inductor
export CUDA_CACHE_PATH="$CACHE_ROOT/cuda"                 # NVIDIA driver's kernel-binary cache (~/.nv/ComputeCache by default)
export TORCH_HOME="$CACHE_ROOT/torch"                     # torch.hub downloads
export XDG_CACHE_HOME="$CACHE_ROOT/xdg"                   # umbrella for anything that respects XDG and has no specific env var
export FLASHINFER_WORKSPACE_BASE="$CACHE_ROOT/flashinfer-root"  # FlashInfer JIT writes to $FLASHINFER_WORKSPACE_BASE/.cache/flashinfer/
