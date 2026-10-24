"""The decode-invariant engine's fused regions of Grug, compiled by the trainer from the torch code vLLM compiles.

vLLM compiles Grug with Inductor, which fuses the elementwise and reduction code between two GEMMs, attention calls or
custom ops into kernels whose arithmetic follows from that code, the Inductor settings and the row count the compile
takes as its size hint. Each region below compiles that code, taken from the pinned vLLM wheel, with the engine's
settings (``INDUCTOR_COMPILE_CONFIG``), so on the trainer's bf16 rows it computes the engine's values bit for bit, under
three conditions this module enforces: the call records no autograd, the settings select kernels without timing them
(``deterministic``), and the first compile takes the engines' ``max_num_batched_tokens`` as its row hint, which
``set_compile_rows`` gives it. A region compiles once per process; the callers attach the gradients (``layers``).

The regions copy or call code from the pinned vLLM wheel and compile under the environment's torch and Triton. A change
to the vLLM pin, torch or Triton therefore requires rerunning the GPU tests ``tests/gpu/test_grug_exact_rollouts.py``,
``test_grug_exact_gradients.py``, ``test_grug_exact_reference_kernels.py``, ``test_grug_exact_ep_combine.py`` and
``test_decode_invariant_engine.py``. The torch version also decides which adds Inductor folds into a GEMM: on torch
2.13 the engine adds the routed and shared experts' outputs to the residual in one fp32 kernel, which
``residual_square_sum`` and ``final_norm`` follow, so a rollout test that fails after such a change means these regions
must follow the engine's new graph. ``require_regions_torch_release`` refuses any torch release but
``REGIONS_TORCH_RELEASE``.
"""

from __future__ import annotations

import functools
from collections.abc import Callable

import torch
import torch.nn.functional as F

from skyrl_train.inference_engines.vllm.decode_invariant.engine import INDUCTOR_COMPILE_CONFIG
from skyrl_train.models.grug_moe import GRUG_ATTN_GATE_SCALE, GRUG_XSA_EPS

# The torch release whose Inductor graphs of the engine these regions follow.
REGIONS_TORCH_RELEASE = "2.13"
# The row hint of every region this process compiles; torch.compile keeps compiled code per process.
_compile_rows: int | None = None


def require_regions_torch_release() -> None:
    """Raise unless this process runs ``REGIONS_TORCH_RELEASE``."""
    release = ".".join(torch.__version__.split(".")[:2])
    if release != REGIONS_TORCH_RELEASE:
        raise RuntimeError(
            f"the exact regions follow the engine's Inductor graphs under torch {REGIONS_TORCH_RELEASE}, not torch "
            f"{torch.__version__}; see the docstring of skyrl_train.models.grug_exact.engine_regions"
        )


def set_compile_rows(rows: int) -> None:
    """Compile this process's regions at ``rows`` rows, the decode-invariant engines' ``max_num_batched_tokens``, at
    which vLLM compiles theirs."""
    global _compile_rows
    if not INDUCTOR_COMPILE_CONFIG.get("deterministic"):
        raise ValueError(f"the engines' Inductor settings {INDUCTOR_COMPILE_CONFIG} choose kernels by timing them")
    if _compile_rows not in (None, rows):
        raise ValueError(f"this process compiles its regions at {_compile_rows} rows, not {rows}")
    _compile_rows = rows


class _Region:
    """One of the engine's fused regions, compiled by ``torch.compile`` at its first call and run without autograd.

    The first ``row_inputs`` arguments are per-token inputs that share the first argument's leading dimensions.
    Each is flattened to a contiguous ``[rows, width]`` or ``[rows]`` tensor, the layout the engine's compiled region
    reads, and the outputs are reshaped back to those leading dimensions. The first call marks the row dimension
    dynamic, with the engines' row count as Inductor's size hint. A later call that would need a new compile
    raises."""

    def __init__(self, function: Callable, row_inputs: int):
        self._function = function
        self._row_inputs = row_inputs
        self._compiled: Callable | None = None

    def __call__(self, *inputs):
        if _compile_rows is None:
            raise RuntimeError("set_compile_rows sets the engines' row count before a region compiles")
        leading = inputs[0].shape[:-1]
        rows = [
            value.detach().reshape(leading.numel(), *value.shape[len(leading) :]).contiguous()
            for value in inputs[: self._row_inputs]
        ]
        others = [value.detach() if isinstance(value, torch.Tensor) else value for value in inputs[self._row_inputs :]]
        compiled = self._compiled
        if compiled is None:
            compiled = torch.compile(self._function, fullgraph=True, options=dict(INDUCTOR_COMPILE_CONFIG))
            for tensor in rows:
                torch._dynamo.mark_dynamic(tensor, 0, hint_override=_compile_rows)
        with torch.no_grad(), torch.compiler.set_stance("default" if self._compiled is None else "fail_on_recompile"):
            output = compiled(*rows, *others)
        self._compiled = compiled
        if isinstance(output, tuple):
            return tuple(value.view(*leading, *value.shape[1:]) for value in output)
        return output.view(*leading, *output.shape[1:])


def _rms_norm(x, weight, epsilon):
    """``vllm.ir.ops.rms_norm``'s native implementation, vLLM's ``RMSNorm``."""
    orig_dtype = x.dtype
    x = x.to(torch.float32)
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    x = x * torch.rsqrt(variance + epsilon)
    x = x.to(weight.dtype) * weight
    return x.to(orig_dtype)


def _rms_norm_from_square_sum(x, square_sum, weight, epsilon):
    """``_rms_norm`` of the stored bf16 rows ``x``, with the variance taken from ``square_sum``, the fp32 sum of
    squares of the unrounded rows. In the engine this is the second half of the fused kernel that
    ``_residual_square_sum`` and ``_gated_product_square_sum`` start."""
    orig_dtype = x.dtype
    x = x.to(torch.float32)
    x = x * torch.rsqrt(square_sum[:, None] / x.shape[-1] + epsilon)
    x = x.to(weight.dtype) * weight
    return x.to(orig_dtype)


def _residual_square_sum(residual, routed, shared):
    """A layer's output ``residual + (routed + shared)`` (``GrugMoeDecoderLayer.forward``) and the fp32 sum of squares
    of each unrounded row, which the next layer's input ``RMSNorm`` reduces in the same kernel.

    Inductor folds an add into the GEMM that produces one of its operands (``addmm``) only when a non-pointwise op
    reads the sum. Only the residual add reads ``routed + shared``, so the shared expert's down projection stays a
    plain GEMM, and this kernel adds the residual, the routed output and the shared output in fp32 without rounding
    between the adds."""
    summed = residual + (routed + shared)
    return summed, summed.to(torch.float32).pow(2).sum(dim=-1)


def _final_norm(residual, routed, shared, weight, epsilon):
    """The last layer's ``residual + (routed + shared)`` and the final ``RMSNorm`` (``GrugMoeModel.forward``), one
    kernel."""
    return _rms_norm(residual + (routed + shared), weight, epsilon)


def _gate_silu(gate_hidden):
    """``GrugMoeGatedNorm.forward``'s activation."""
    return F.silu(gate_hidden)


def _gated_product(x, gate):
    """``GrugMoeGatedNorm.forward``'s product."""
    return x * torch.sigmoid(gate).to(x.dtype)


def _gated_product_square_sum(x, gate):
    """The embedding gated norm's product (``GrugMoeModel.forward``) and the fp32 sum of squares of each unrounded
    row, which layer 0's input ``RMSNorm`` reduces in the same kernel."""
    product = _gated_product(x, gate)
    return product, product.to(torch.float32).pow(2).sum(dim=-1)


def _shared_activation(gate, up):
    """``GrugMoeDenseMLP.forward``'s activation."""
    return F.silu(gate) * up


def _xsa_head_gate(attn_output, value_states, gate_logits, head_dim):
    """``GrugMoeAttention.apply_xsa`` after the head gate's GEMM, whose fp32 output the engine's
    ``GrugRowInvariantLinearMethod.apply`` rounds to bf16."""
    from vllm.model_executor.models import grugmoe  # noqa: PLC0415 - vLLM is a GPU-only extra

    num_tokens, num_heads = gate_logits.shape
    value_heads = value_states.view(num_tokens, -1, head_dim)
    aligned_value = grugmoe._align_kv_heads(value_heads, num_heads)
    attn_heads = attn_output.view(num_tokens, num_heads, head_dim)
    dot = torch.sum(attn_heads * aligned_value, dim=-1, keepdim=True)
    value_norm_sq = torch.sum(aligned_value * aligned_value, dim=-1, keepdim=True)
    attn_heads = attn_heads - (dot / (value_norm_sq + GRUG_XSA_EPS)) * aligned_value
    gate = GRUG_ATTN_GATE_SCALE * torch.sigmoid(gate_logits.to(attn_output.dtype))
    attn_heads = gate[..., None].to(attn_heads.dtype) * attn_heads
    return attn_heads.reshape(num_tokens, -1)


def _normalized_query_key(q, k, head_dim):
    """``GrugMoeAttention.forward``'s weightless q/k norms."""
    from vllm.model_executor.models import grugmoe  # noqa: PLC0415 - vLLM is a GPU-only extra

    num_tokens = q.shape[0]
    q = grugmoe._rms_norm(q.view(num_tokens, -1, head_dim))
    k = grugmoe._rms_norm(k.view(num_tokens, -1, head_dim))
    return q.reshape(num_tokens, -1), k.reshape(num_tokens, -1)


def _query_key(q, k, head_dim, qk_mult, qk_mult_scale):
    """``GrugMoeAttention.forward`` from the q and k projections to attention on a layer without RoPE."""
    q, k = _normalized_query_key(q, k, head_dim)
    return q * qk_mult * qk_mult_scale, k


def _query_key_rope(q, k, positions, cos_sin_cache, head_dim, qk_mult, qk_mult_scale):
    """The same on a RoPE layer: ``RotaryEmbedding.forward_native`` rotates half of each head, NeoX style, the
    rotation the gradient kernels implement (``reference_kernels``)."""
    from vllm.model_executor.layers.rotary_embedding.base import RotaryEmbedding  # noqa: PLC0415 - GPU-only extra

    q, k = _normalized_query_key(q, k, head_dim)
    q, k = RotaryEmbedding.forward_static(positions, q, k, head_dim, head_dim // 2, cos_sin_cache, True)
    return q * qk_mult * qk_mult_scale, k


rms_norm = _Region(_rms_norm, row_inputs=1)
rms_norm_from_square_sum = _Region(_rms_norm_from_square_sum, row_inputs=2)
residual_square_sum = _Region(_residual_square_sum, row_inputs=3)
final_norm = _Region(_final_norm, row_inputs=3)
gate_silu = _Region(_gate_silu, row_inputs=1)
gated_product = _Region(_gated_product, row_inputs=2)
gated_product_square_sum = _Region(_gated_product_square_sum, row_inputs=2)
shared_activation = _Region(_shared_activation, row_inputs=2)
xsa_head_gate = _Region(_xsa_head_gate, row_inputs=3)
query_key = _Region(_query_key, row_inputs=2)
_query_key_rope_region = _Region(_query_key_rope, row_inputs=3)


@functools.cache
def _rotary_table(device: int, base: float, positions: int, rotary_dim: int) -> torch.Tensor:
    """The engine's bf16 cos/sin table, built on the GPU as vLLM's ``RotaryEmbedding._compute_cos_sin_cache`` builds
    it at load."""
    with torch.device(f"cuda:{device}"):
        inv_freq = 1.0 / (base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim))
        freqs = torch.einsum("i,j -> ij", torch.arange(positions, dtype=torch.float), inv_freq)
        return torch.cat((freqs.cos(), freqs.sin()), dim=-1).to(torch.bfloat16)


def query_key_rope(q, k, head_dim: int, qk_mult: float, qk_mult_scale: float, base: float, positions: int):
    """``query_key`` on a RoPE layer of ``base`` and a ``positions``-row table for ``[S, B, width]`` q and k, each
    token at its position in its sequence."""
    sequence, batch = q.shape[:2]
    token_positions = torch.arange(sequence, device=q.device).repeat_interleave(batch)
    table = _rotary_table(q.device.index, base, positions, head_dim // 2)
    return _query_key_rope_region(q, k, token_positions, table, head_dim, qk_mult, qk_mult_scale)
