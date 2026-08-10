#!/usr/bin/env bash
set -euxo pipefail

rm -rf /tmp/attn-venv /tmp/attn-training-preflight
mkdir -p /tmp/attn-training-preflight
uv venv /tmp/attn-venv --python 3.12
uv pip install --python /tmp/attn-venv/bin/python \
  'jax[cuda13]==0.10.1' \
  'torch==2.11.0' \
  'triton==3.6.0'

uv pip freeze --python /tmp/attn-venv/bin/python | sort | tee /tmp/attn-training-preflight/packages.txt

export JAX_PLATFORMS=cpu
export PYTHONPATH=lib/tile_lifetime/src

/tmp/attn-venv/bin/python - <<'PY'
from __future__ import annotations

import json
from pathlib import Path

import jax
import jaxlib
import torch
import triton

from tile_lifetime.jax_streaming_attention_backward_ffi import (
    StreamingAttentionBackwardFfiBufferLayout,
    StreamingAttentionBackwardResultPolicy,
    generate_streaming_attention_backward_ffi,
)
from tile_lifetime.plan import NumericalPolicy
from tile_lifetime.stablehlo_import import import_stablehlo
from tile_lifetime.stablehlo_streaming_attention_backward import recover_stablehlo_streaming_attention_backward
from tile_lifetime.streaming_attention import StreamingTileSchedule
from tile_lifetime.streaming_attention_backward import (
    StreamingAttentionBackwardDomainTraversal,
    derive_streaming_attention_backward_tile_schedule,
    eliminate_normalized_exp_maximum_vjp,
)
from tile_lifetime.streaming_attention_backward_reference import (
    STREAMING_ATTENTION_BACKWARD_INPUT_NAMES,
    StreamingAttentionBackwardDebugConfig,
    export_debug_streaming_attention_training,
)

assert jax.__version__ == "0.10.1"
assert jaxlib.__version__ == "0.10.1"
assert torch.__version__.split("+")[0] == "2.11.0"
assert triton.__version__ == "3.6.0"

site_packages = Path(jax.__file__).resolve().parents[1]
cuda_root = site_packages / "nvidia" / "cu13"
nvcc = cuda_root / "bin" / "nvcc"
ptxas = cuda_root / "bin" / "ptxas"
runtime_header = cuda_root / "include" / "cuda_runtime_api.h"
runtime_library = cuda_root / "lib" / "libcudart.so.13"
for required in (nvcc, ptxas, runtime_header, runtime_library):
    if not required.exists():
        raise FileNotFoundError(required)

config = StreamingAttentionBackwardDebugConfig(
    batch=1,
    query_length=2048,
    key_length=2048,
    query_heads=32,
    key_value_heads=8,
    head_dimension=128,
    scale=128**-0.5,
)
hlo = export_debug_streaming_attention_training(config)
graph = import_stablehlo(hlo, input_names=STREAMING_ATTENTION_BACKWARD_INPUT_NAMES)
recovered = recover_stablehlo_streaming_attention_backward(
    graph,
    schedule=StreamingTileSchedule(query_tile_size=32, key_value_tile_size=32, pipeline_depth=3),
)
program = eliminate_normalized_exp_maximum_vjp(
    recovered.program,
    numerical_policy=NumericalPolicy.ALLOW_ROUNDING_REORDER,
)
schedule = derive_streaming_attention_backward_tile_schedule(
    program,
    query_tile_size=32,
    key_value_tile_size=32,
    domain_traversal=StreamingAttentionBackwardDomainTraversal.LOWER_TRIANGULAR,
)
generated = generate_streaming_attention_backward_ffi(
    program,
    schedule,
    target_name="shuttle.streaming_training.environment_preflight_v1",
    result_policy=StreamingAttentionBackwardResultPolicy.FORWARD_OUTPUT_AND_GRADIENTS,
    output_layouts=tuple(
        StreamingAttentionBackwardFfiBufferLayout(name, (3, 1, 2, 0))
        for name in ("forward_output", "query_cotangent", "key_cotangent", "value_cotangent")
    ),
)
assert tuple(kernel.kernel_name for kernel in generated.aot_kernels) == (
    "_streaming_grouped_query_forward",
    "_streaming_dq_kernel",
    "_streaming_dkdv_kernel",
)
assert tuple(output.name for output in generated.outputs) == (
    "forward_output",
    "query_cotangent",
    "key_cotangent",
    "value_cotangent",
)

report = {
    "cuda_root": str(cuda_root),
    "jax": jax.__version__,
    "jax_backend": jax.default_backend(),
    "jaxlib": jaxlib.__version__,
    "kernel_names": [kernel.kernel_name for kernel in generated.aot_kernels],
    "nvcc": str(nvcc),
    "output_layouts": [list(output.layout) for output in generated.outputs],
    "ptxas": str(ptxas),
    "semantic_fingerprint": generated.semantic_fingerprint,
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "triton": triton.__version__,
}
Path("/tmp/attn-training-preflight/report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
print(json.dumps(report, indent=2, sort_keys=True))
PY

NVCC=/tmp/attn-venv/lib/python3.12/site-packages/nvidia/cu13/bin/nvcc
PTXAS=/tmp/attn-venv/lib/python3.12/site-packages/nvidia/cu13/bin/ptxas
"$NVCC" --version | tee /tmp/attn-training-preflight/nvcc.txt
"$PTXAS" --version | tee /tmp/attn-training-preflight/ptxas.txt
sha256sum /tmp/attn-training-preflight/packages.txt /tmp/attn-training-preflight/report.json
