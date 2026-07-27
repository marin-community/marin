# NCCL UB-X direct EP8 gate

This gate answers one question before any JAX integration: does stock UB-X
BF16 MoE transport beat Marin's exact Ring transport by at least 10% on one
H100x8 NVLink node without changing routing or outputs?

The default shape is EP8, 64 experts, top-k 4, hidden size 2560, and 16,384
tokens per source rank. Each rank therefore originates 65,536 assignments.
The benchmark runs balanced and deterministic learned-skew routing separately.
Skewed routes use distinct top-k choices from synthetic router logits with a
Zipf-like expert bias (`alpha=1.2`).

The comparison keeps expert compute out of both arms:

- UB-X uses `a2av_token_bf16_bf16_topk` followed by the upstream-recommended
  `combine_push3_bf16_bf16`.
- Ring uses the same fixed routes and Marin's source-major, expert-order rank
  cap. It all-gathers BF16 tokens and route metadata, packs the local expert
  assignments, scatters weighted identity-expert outputs into a dense global
  buffer, then uses BF16 reduce-scatter to return owner-local outputs.

Every timing sample uses CUDA events. The reported sample is the maximum rank
duration for that iteration; the headline is p50 over those slowest-rank
samples. Routing-map construction and correctness checks are outside timing.

## Admission

Each routing case fails the process unless all conditions hold:

1. UB-X token offsets, top-k dispatch maps, PUSH inverse maps, expert counts,
   accepted routes, and drops exactly match the independent Ring oracle.
2. BF16 dispatch is bitwise exact.
3. Dispatch output, Ring output, UB-X output, and UB-X-versus-Ring output are
   finite and no worse than `0.002` relative L2 against their stated reference.
4. `ring_p50_ms / ubx_p50_ms >= 1.10`.

The two-route launcher uses `set -o pipefail`, so success means both route
processes passed admission.

## Pinned source and build

Use NVIDIA/nccl commit
`db0c814185a0415cc2e23dca387fecb9282de551`. Its `version.mk` identifies
NCCL 2.30.7-1. At this commit UB-X lives in `contrib/nccl_ubx`, links directly
to `libnccl`, requires SM90+, CUDA 12+, PyTorch 2.1+, and uses the matching
in-tree `bindings/nccl4py` package for NCCL device communicator, symmetric
window, and memory APIs.

Run these commands inside a CUDA/PyTorch environment on the target node. Keep
the source-built NCCL first in both the link and runtime search paths; using a
different wheel-provided `libnccl.so` invalidates the gate.

```bash
export NCCL_UBX_SOURCE=/opt/nccl-ubx-db0c814
export CUDA_HOME=/usr/local/cuda

git clone https://github.com/NVIDIA/nccl.git "${NCCL_UBX_SOURCE}"
git -C "${NCCL_UBX_SOURCE}" checkout db0c814185a0415cc2e23dca387fecb9282de551

make -C "${NCCL_UBX_SOURCE}" -j"$(nproc)" src.build CUDA_HOME="${CUDA_HOME}"

export NCCL_INCLUDE_DIR="${NCCL_UBX_SOURCE}/build/include"
export NCCL_LIBRARY_DIR="${NCCL_UBX_SOURCE}/build/lib"
export LD_LIBRARY_PATH="${NCCL_LIBRARY_DIR}:${LD_LIBRARY_PATH:-}"

python -m pip install --no-deps -e "${NCCL_UBX_SOURCE}/bindings/nccl4py"
TORCH_CUDA_ARCH_LIST=9.0a \
  NCCL_INCLUDE_DIR="${NCCL_INCLUDE_DIR}" \
  NCCL_LIBRARY_DIR="${NCCL_LIBRARY_DIR}" \
  python -m pip install --no-build-isolation --no-deps \
  -e "${NCCL_UBX_SOURCE}/contrib/nccl_ubx"
```

The environment must already provide PyTorch, the build requirements declared
by both upstream packages, and the CUDA bindings required by `nccl4py`. Verify
that all three layers resolve the pinned library before running:

```bash
git -C "${NCCL_UBX_SOURCE}" rev-parse HEAD
python -c 'import torch, ubx; print(torch.__version__, torch.cuda.nccl.version(), ubx.get_version())'
ldd "$(python -c 'import ubx._C; print(ubx._C.__file__)')" | grep libnccl
```

Expected source commit:

```text
db0c814185a0415cc2e23dca387fecb9282de551
```

## CPU/static preflight

These commands do not initialize CUDA or import PyTorch:

```bash
uv run pytest experiments/grug/moe/test_benchmark_nccl_ubx.py

uv run python experiments/grug/moe/benchmark_nccl_ubx.py \
  --plan-only \
  --routing balanced \
  --ubx-source "${NCCL_UBX_SOURCE}"

uv run python experiments/grug/moe/benchmark_nccl_ubx.py \
  --plan-only \
  --routing learned_skew \
  --ubx-source "${NCCL_UBX_SOURCE}"
```

The target plans must report EP8, 131,072 global tokens, 65,536 assignments per
source rank, and 524,288 global assignments.

## GPU gate

No Iris or cluster configuration is embedded in this harness. On an allocated
one-node H100x8 shell:

```bash
export NCCL_UBX_SOURCE=/opt/nccl-ubx-db0c814
export LD_LIBRARY_PATH="${NCCL_UBX_SOURCE}/build/lib:${LD_LIBRARY_PATH:-}"
export UBX_GRAPH_POOL_SHARE=0.1
export NCCL_UBX_OUTPUT_DIR="${PWD}/nccl-ubx-direct-results"

experiments/grug/moe/run_nccl_ubx_direct_gate.sh
```

For initial hang localization only, rebuild the UB-X extension with
`UBX_BUILD_TIMEOUT=1` and set `UBX_TIMEOUT_SEC`. That diagnostic build adds
polling overhead and must not supply admission timing.

The launcher writes one JSONL file per routing case. Preserve the full files,
the exact container image, driver/CUDA versions, topology output, and `ldd`
result with the eventual GPU run.
