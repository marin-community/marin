"""The vLLM engine kwargs and environment that make a vLLM engine decode-invariant."""

from __future__ import annotations

from importlib.metadata import version
from typing import Any

# vLLM's CUSTOM attention backend, which ``plugin.register_attention_backend`` sets in every vLLM process.
DECODE_INVARIANT_ATTENTION_BACKEND = "CUSTOM"
# Inductor settings with which Inductor builds every kernel without timing candidate kernels.
INDUCTOR_COMPILE_CONFIG = {"deterministic": True, "force_shape_pad": True, "combo_kernels": False}
# vLLM's default CUDA graph capture sizes up to 512 tokens, without size 1: vLLM's compiled graph computes a token
# in a one-token step differently than in a step of two or more, so a one-token step runs padded to two.
CUDA_GRAPH_MODE = "FULL_AND_PIECEWISE"
CUDA_GRAPH_CAPTURE_SIZES = (2, 4, *range(8, 256, 8), *range(256, 513, 16))
# Without NVLink SHARP or NCCL symmetric memory, vLLM's expert-parallel combine runs NCCL's ring reduce-scatter.
DECODE_INVARIANT_ENGINE_ENVIRONMENT = {"NCCL_NVLS_ENABLE": "0", "VLLM_USE_NCCL_SYMM_MEM": "0"}


def decode_invariant_engine_init_kwargs() -> dict[str, Any]:
    """The engine kwargs a decode-invariant engine adds to its ``engine_init_kwargs``."""
    return {
        # vLLM's GrugMoE computes its router logits and head gates with grug_row_invariant_gemm.
        "additional_config": {"grug_row_invariant_gates": True},
        "compilation_config": {
            "inductor_compile_config": dict(INDUCTOR_COMPILE_CONFIG),
            "cudagraph_mode": CUDA_GRAPH_MODE,
            "cudagraph_capture_sizes": list(CUDA_GRAPH_CAPTURE_SIZES),
        },
    }


def require_grug_row_invariant_gemm() -> None:
    """Raise unless the installed vLLM's GrugMoE has ``grug_row_invariant_gemm``, which ``grug_row_invariant_gates``
    selects."""
    from vllm.model_executor.models import grugmoe  # noqa: PLC0415 - vLLM is a GPU-only extra

    if not hasattr(grugmoe, "grug_row_invariant_gemm"):
        raise RuntimeError(
            "exact numerics compute GrugMoE's router logits and head gates with grug_row_invariant_gemm from "
            f"marin-community/vllm#96; vLLM {version('vllm')} does not have it"
        )
