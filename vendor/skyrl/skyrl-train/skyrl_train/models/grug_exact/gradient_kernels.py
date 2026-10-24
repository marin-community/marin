"""Tensors that hold a kernel's output values and take their gradient from the trainer's own computation of the same
values: either from running that computation under autograd (``kernel_value``), or from Triton kernels that evaluate
autograd's elementwise formulas in autograd's order and with its roundings, so the gradients equal autograd's bit for
bit."""

from __future__ import annotations

import functools
from collections.abc import Callable

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from skyrl_train.models.grug_exact.support import VERIFIED_MODEL
from skyrl_train.models.grug_moe import GRUG_ATTN_GATE_SCALE, GRUG_QK_RMS_NORM_EPS, GRUG_XSA_EPS

HEAD_DIM = VERIFIED_MODEL.head_dim
# Grug rotates half of each head.
ROTARY_DIM = HEAD_DIM // 2

_BLOCK = 1024
# Neither implicit fused multiply-adds nor flushed subnormals, as in PyTorch's elementwise kernels; the kernels call
# libdevice.fma where PyTorch's kernel has one.
_EXACT_LAUNCH = {"enable_fp_fusion": False, "enable_reflect_ftz": False, "num_warps": 4}
# The head geometry the XSA and q/k kernels address.
_HEAD = tl.constexpr(HEAD_DIM)
_ROTARY = tl.constexpr(ROTARY_DIM)
_HALF_ROTARY = tl.constexpr(ROTARY_DIM // 2)
_INVERSE_HEAD = tl.constexpr(1.0 / HEAD_DIM)


class _ValueWithGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        # Autograd wraps the returned tensor in a view whose backward is this function's; nothing is copied.
        return value

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        return None, grad


def kernel_value(value: torch.Tensor, reference: Callable[[], torch.Tensor]) -> torch.Tensor:
    """Return ``value`` with its bits unchanged. With gradients enabled, its gradient is the gradient of
    ``reference()``, which runs only then."""
    if not torch.is_grad_enabled():
        return value
    return _ValueWithGradient.apply(value, reference())


def _grid(numel: int) -> tuple[int]:
    return (-(-numel // _BLOCK),)


@triton.jit
def _gated_product_backward(dy_ptr, normalized_ptr, gate_ptr, d_normalized_ptr, d_gate_ptr, numel, BLOCK: tl.constexpr):
    offsets = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < numel
    grad = tl.load(dy_ptr + offsets, mask=mask).to(tl.float32)  # ToCopyBackward of .to(bf16)
    normalized = tl.load(normalized_ptr + offsets, mask=mask).to(tl.float32)
    gate = tl.load(gate_ptr + offsets, mask=mask).to(tl.float32)
    sigmoid = libdevice.div_rn(1.0, 1.0 + libdevice.exp(-gate))  # torch.sigmoid's saved result
    d_normalized = grad * sigmoid  # MulBackward0: grad * other
    d_sigmoid = grad * normalized  # MulBackward0: grad * self
    d_gate = (d_sigmoid * (1.0 - sigmoid)) * sigmoid  # sigmoid_backward: a * (1 - b) * b
    tl.store(d_normalized_ptr + offsets, d_normalized.to(tl.bfloat16), mask=mask)  # ToCopyBackward of .float()
    tl.store(d_gate_ptr + offsets, d_gate.to(tl.bfloat16), mask=mask)


@triton.jit
def _swiglu_backward(dy_ptr, fc1_ptr, d_fc1_ptr, numel, WIDTH: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < numel
    gate_offsets = (offsets // WIDTH) * (2 * WIDTH) + offsets % WIDTH
    grad = tl.load(dy_ptr + offsets, mask=mask).to(tl.float32)  # ToCopyBackward of .to(bf16)
    gate = tl.load(fc1_ptr + gate_offsets, mask=mask).to(tl.float32)
    up = tl.load(fc1_ptr + gate_offsets + WIDTH, mask=mask).to(tl.float32)
    silu = libdevice.div_rn(gate, 1.0 + libdevice.exp(-gate))  # F.silu's saved result: x / (1 + exp(-x))
    d_silu = grad * up  # MulBackward0: grad * other
    d_up = grad * silu  # MulBackward0: grad * self
    sigmoid = libdevice.div_rn(1.0, 1.0 + libdevice.exp(-gate))
    # silu_backward: dy * s * (1 + x * (1 - s)), where nvcc fuses 1 + x * (1 - s) into one multiply-add.
    d_gate = (d_silu * sigmoid) * libdevice.fma(gate, 1.0 - sigmoid, 1.0)
    tl.store(d_fc1_ptr + gate_offsets, d_gate.to(tl.bfloat16), mask=mask)  # chunk's backward concatenates
    tl.store(d_fc1_ptr + gate_offsets + WIDTH, d_up.to(tl.bfloat16), mask=mask)


@triton.jit
def _hybrid_norm_backward_terms(
    dy_ptr,
    hidden_ptr,
    weight_ptr,
    rsqrt_ptr,
    d_hidden_ptr,
    weight_terms_ptr,
    rsqrt_terms_ptr,
    numel,
    WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < numel
    grad = tl.load(dy_ptr + offsets, mask=mask).to(tl.float32)  # ToCopyBackward of .to(bf16)
    hidden = tl.load(hidden_ptr + offsets, mask=mask).to(tl.float32)
    weight = tl.load(weight_ptr + offsets % WIDTH, mask=mask).to(tl.float32)
    rsqrt = tl.load(rsqrt_ptr + offsets // WIDTH, mask=mask)
    normalized = hidden * rsqrt  # the forward's rounded.float() * rsqrt(variance + eps)
    d_normalized = grad * weight  # MulBackward0 of normalized * weight.float(): grad * other
    tl.store(weight_terms_ptr + offsets, grad * normalized, mask=mask)  # grad * self, summed to the weight
    d_hidden = d_normalized * rsqrt  # MulBackward0 of rounded.float() * rsqrt: grad * other
    tl.store(rsqrt_terms_ptr + offsets, d_normalized * hidden, mask=mask)  # grad * self, summed to the rsqrt
    tl.store(d_hidden_ptr + offsets, d_hidden.to(tl.bfloat16), mask=mask)  # ToCopyBackward of .float()


@triton.jit
def _variance_backward(
    d_variance_ptr, hidden_ptr, d_hidden_ptr, inverse_width, numel, WIDTH: tl.constexpr, BLOCK: tl.constexpr
):
    offsets = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < numel
    hidden = tl.load(hidden_ptr + offsets, mask=mask).to(tl.float32)
    d_variance = tl.load(d_variance_ptr + offsets // WIDTH, mask=mask)
    d_square = d_variance * inverse_width  # MeanBackward1: grad.expand(...) / width, as grad * (1 / width)
    d_hidden = d_square * (2.0 * hidden)  # PowBackward0: grad * (2 * self.pow(1))
    tl.store(d_hidden_ptr + offsets, d_hidden.to(tl.bfloat16), mask=mask)  # ToCopyBackward of .float()


@triton.jit
def _xsa_forward_terms(
    attention_ptr, value_ptr, products_ptr, squares_ptr, numel, GROUP: tl.constexpr, BLOCK: tl.constexpr
):
    """The chain's fp32 products ``attention * v`` and ``v.square()`` over ``[rows, heads, head_dim]``, with ``v`` the
    value head of each query head's group: element ``i`` of query head ``h`` reads element ``i`` of value head
    ``h // GROUP``."""
    offsets = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < numel
    attention = tl.load(attention_ptr + offsets, mask=mask).to(tl.float32)
    value = tl.load(value_ptr + (offsets // _HEAD // GROUP) * _HEAD + offsets % _HEAD, mask=mask).to(tl.float32)
    tl.store(products_ptr + offsets, attention * value, mask=mask)
    tl.store(squares_ptr + offsets, value * value, mask=mask)  # pow(v, 2) multiplies v by itself


@triton.jit
def _xsa_backward_terms(
    dy_ptr,
    attention_ptr,
    value_ptr,
    quotient_ptr,
    scale_ptr,
    scale_terms_ptr,
    quotient_terms_ptr,
    numel,
    GROUP: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < numel
    head_row = offsets // _HEAD
    grad = tl.load(dy_ptr + offsets, mask=mask).to(tl.float32)  # ToCopyBackward of .to(bf16)
    attention = tl.load(attention_ptr + offsets, mask=mask).to(tl.float32)
    value = tl.load(value_ptr + (offsets // _HEAD // GROUP) * _HEAD + offsets % _HEAD, mask=mask).to(tl.float32)
    quotient = tl.load(quotient_ptr + head_row, mask=mask)
    scale = tl.load(scale_ptr + head_row, mask=mask)
    projected = attention - quotient * value  # the forward's attention - (dot / denominator) * v
    # MulBackward0 grad * self, summed to the scale
    tl.store(scale_terms_ptr + offsets, grad * projected, mask=mask)
    d_projected = grad * scale  # MulBackward0 grad * other
    tl.store(quotient_terms_ptr + offsets, (-d_projected) * value, mask=mask)  # SubBackward0's -grad, times v


@triton.jit
def _xsa_backward_inputs(
    dy_ptr,
    attention_ptr,
    value_ptr,
    quotient_ptr,
    scale_ptr,
    d_dot_ptr,
    d_denominator_ptr,
    d_attention_ptr,
    d_value_heads_ptr,
    numel,
    GROUP: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < numel
    head_row = offsets // _HEAD
    grad = tl.load(dy_ptr + offsets, mask=mask).to(tl.float32)
    attention = tl.load(attention_ptr + offsets, mask=mask).to(tl.float32)
    value = tl.load(value_ptr + (offsets // _HEAD // GROUP) * _HEAD + offsets % _HEAD, mask=mask).to(tl.float32)
    quotient = tl.load(quotient_ptr + head_row, mask=mask)
    scale = tl.load(scale_ptr + head_row, mask=mask)
    d_dot = tl.load(d_dot_ptr + head_row, mask=mask)
    d_denominator = tl.load(d_denominator_ptr + head_row, mask=mask)
    d_projected = grad * scale
    minus_d_projected = -d_projected
    from_quotient = minus_d_projected * quotient  # MulBackward0 of quotient * v: grad * self
    from_square = d_denominator * (2.0 * value)  # PowBackward0 of v.pow(2): grad * (2 * v.pow(1))
    from_dot_value = d_dot * attention  # MulBackward0 of attention * v: grad * self
    from_dot_attention = d_dot * value  # MulBackward0 of attention * v: grad * other
    # Autograd adds a tensor's gradients in the order their nodes run: the subtraction's before the dot's for the
    # attention, and for v the quotient product's, then the square's, then the dot's.
    d_attention = d_projected + from_dot_attention
    d_value = (from_quotient + from_square) + from_dot_value
    tl.store(d_attention_ptr + offsets, d_attention.to(tl.bfloat16), mask=mask)
    tl.store(d_value_heads_ptr + offsets, d_value.to(tl.bfloat16), mask=mask)


@triton.jit
def _qk_scaled(dy_ptr, offsets, mask, multiplier, multiplier_scale, QUERY: tl.constexpr):
    """The gradient reaching the RoPE output: the query's two scale multiplications run backward (scale first)."""
    grad = tl.load(dy_ptr + offsets, mask=mask, other=0.0).to(tl.float32)  # ToCopyBackward of .to(bf16)
    if QUERY:
        grad = (grad * multiplier_scale) * multiplier
    return grad


@triton.jit
def _qk_rotary_grad(dy_ptr, offsets, mask, multiplier, multiplier_scale, QUERY: tl.constexpr):
    """The gradient of an element of the rotated half: the query's passes its bf16 round trip and the sum of the two
    zero-padded slice gradients (which turns -0 into +0); the key's reaches the rotation directly."""
    grad = _qk_scaled(dy_ptr, offsets, mask, multiplier, multiplier_scale, QUERY)
    if QUERY:
        grad = grad.to(tl.bfloat16).to(tl.float32) + 0.0
    return grad


@triton.jit
def _qk_backward_terms(
    dy_ptr,
    raw_ptr,
    rsqrt_ptr,
    cos_ptr,
    sin_ptr,
    scaled_ptr,
    products_ptr,
    numel,
    multiplier,
    multiplier_scale,
    HEADS_PER_POSITION: tl.constexpr,
    ROTARY: tl.constexpr,
    QUERY: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < numel
    head_row = offsets // _HEAD
    lane = offsets % _HEAD
    if ROTARY:
        position = head_row // HEADS_PER_POSITION
        rotary = lane < _ROTARY
        rotary_lane = tl.where(rotary, lane, 0)
        partner = tl.where(rotary_lane < _HALF_ROTARY, rotary_lane + _HALF_ROTARY, rotary_lane - _HALF_ROTARY)
        row_start = head_row * _HEAD
        rotary_mask = mask & rotary
        own = _qk_rotary_grad(dy_ptr, row_start + rotary_lane, rotary_mask, multiplier, multiplier_scale, QUERY)
        other = _qk_rotary_grad(dy_ptr, row_start + partner, rotary_mask, multiplier, multiplier_scale, QUERY)
        cos = tl.load(cos_ptr + position * _ROTARY + rotary_lane, mask=rotary_mask, other=0.0)
        sin = tl.load(sin_ptr + position * _ROTARY + partner, mask=rotary_mask, other=0.0)
        # Rotation backward: grad * cos, plus the partner's grad * sin (negated for the second half), the multiply's
        # and the chunk's gradients of the rotary slice added.
        from_partner = other * sin
        rotated = own * cos + tl.where(rotary_lane < _HALF_ROTARY, from_partner, -from_partner)
        passthrough = _qk_scaled(dy_ptr, offsets, mask & (lane >= _ROTARY), multiplier, multiplier_scale, QUERY)
        grad = tl.where(rotary, rotated, passthrough) + 0.0  # the slices' zero-padded gradients summed
    else:
        grad = _qk_scaled(dy_ptr, offsets, mask, multiplier, multiplier_scale, QUERY)
    raw = tl.load(raw_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    rsqrt = tl.load(rsqrt_ptr + head_row, mask=mask, other=0.0)
    tl.store(scaled_ptr + offsets, grad * rsqrt, mask=mask)  # MulBackward0 of fp32 * rsqrt: grad * other
    tl.store(products_ptr + offsets, grad * raw, mask=mask)  # grad * self, summed to the rsqrt


@triton.jit
def _qk_backward_inputs(scaled_ptr, raw_ptr, d_mean_ptr, d_raw_ptr, numel, BLOCK: tl.constexpr):
    offsets = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < numel
    raw = tl.load(raw_ptr + offsets, mask=mask).to(tl.float32)
    d_square = tl.load(d_mean_ptr + offsets // _HEAD, mask=mask) * _INVERSE_HEAD  # MeanBackward1: grad * (1 / dim)
    from_square = d_square * (2.0 * raw)  # PowBackward0: grad * (2 * self.pow(1))
    d_raw = tl.load(scaled_ptr + offsets, mask=mask) + from_square
    tl.store(d_raw_ptr + offsets, d_raw.to(tl.bfloat16), mask=mask)  # ToCopyBackward of .float()


class _ValueWithKernelGradients(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, gradients, *inputs):
        ctx.gradients = gradients
        ctx.save_for_backward(*inputs)
        return value

    @staticmethod
    def backward(ctx, grad):
        return None, None, *ctx.gradients(grad, *ctx.saved_tensors)


def _with_kernel_gradients(
    value: torch.Tensor, gradients: Callable[..., tuple[torch.Tensor | None, ...]], *inputs: torch.Tensor | None
) -> torch.Tensor:
    """``value``, whose gradients with respect to ``inputs`` are ``gradients(grad, *inputs)``."""
    if not torch.is_grad_enabled():
        return value
    return _ValueWithKernelGradients.apply(value, gradients, *inputs)


def _gated_product_gradients(grad, normalized, gate):
    grad = grad.contiguous()
    normalized, gate = normalized.contiguous(), gate.contiguous()
    d_normalized, d_gate = torch.empty_like(normalized), torch.empty_like(gate)
    _gated_product_backward[_grid(grad.numel())](
        grad, normalized, gate, d_normalized, d_gate, grad.numel(), BLOCK=_BLOCK, **_EXACT_LAUNCH
    )
    return d_normalized, d_gate


def gated_product_value(value: torch.Tensor, normalized: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    """``value``, differentiated as ``(normalized.float() * torch.sigmoid(gate.float())).to(bf16)``."""
    return _with_kernel_gradients(value, _gated_product_gradients, normalized, gate)


def _swiglu_gradients(grad, fc1_output):
    grad, fc1_output = grad.contiguous(), fc1_output.contiguous()
    d_fc1 = torch.empty_like(fc1_output)
    _swiglu_backward[_grid(grad.numel())](
        grad, fc1_output, d_fc1, grad.numel(), WIDTH=grad.shape[-1], BLOCK=_BLOCK, **_EXACT_LAUNCH
    )
    return (d_fc1,)


def swiglu_value(value: torch.Tensor, fc1_output: torch.Tensor) -> torch.Tensor:
    """``value``, differentiated as ``(F.silu(gate.float()) * up.float()).to(bf16)`` of ``fc1_output = [gate | up]``."""
    return _with_kernel_gradients(value, _swiglu_gradients, fc1_output)


def _hybrid_input_norm_gradients(grad, hidden, _hidden_for_variance, variance, weight, *, eps):
    """The gradients of ``hybrid_input_norm_value``'s inputs, one per read of ``hidden``: the norm's, then the
    variance's."""
    grad, hidden = grad.contiguous(), hidden.contiguous()
    width = hidden.shape[-1]
    rsqrt = torch.rsqrt(variance + eps)  # the forward's RsqrtBackward0 result, recomputed by the same ops
    d_hidden = torch.empty_like(hidden)
    weight_terms = torch.empty(hidden.shape, dtype=torch.float32, device=hidden.device)
    rsqrt_terms = torch.empty_like(weight_terms)
    _hybrid_norm_backward_terms[_grid(grad.numel())](
        grad,
        hidden,
        weight,
        rsqrt.contiguous(),
        d_hidden,
        weight_terms,
        rsqrt_terms,
        grad.numel(),
        WIDTH=width,
        BLOCK=_BLOCK,
        **_EXACT_LAUNCH,
    )
    # The broadcast operands' gradients, reduced as autograd's sum_to reduces them.
    d_weight = weight_terms.sum(list(range(hidden.dim() - 1)), keepdim=True).view(weight.shape).to(weight.dtype)
    d_rsqrt = rsqrt_terms.sum([hidden.dim() - 1], keepdim=True)
    d_variance = (-0.5 * d_rsqrt) * rsqrt.pow(3)  # RsqrtBackward0; the eps add passes the gradient through
    d_hidden_variance = torch.empty_like(hidden)
    _variance_backward[_grid(grad.numel())](
        d_variance.contiguous(),
        hidden,
        d_hidden_variance,
        float(1.0 / width),
        grad.numel(),
        WIDTH=width,
        BLOCK=_BLOCK,
        **_EXACT_LAUNCH,
    )
    return d_hidden, d_hidden_variance, None, d_weight


def hybrid_input_norm_value(
    value: torch.Tensor, hidden: torch.Tensor, variance: torch.Tensor, weight: torch.Tensor, eps: float
) -> torch.Tensor:
    """``value``, differentiated as ``(hidden.float() * rsqrt(variance + eps) * weight.float()).to(bf16)`` with
    ``variance`` differentiated as ``hidden.float().pow(2).mean(-1, keepdim=True)``."""
    gradients = functools.partial(_hybrid_input_norm_gradients, eps=eps)
    return _with_kernel_gradients(value, gradients, hidden, hidden, variance, weight)


def _xsa_head_gate_gradients(grad, attention, kv_value, gate):
    attention, kv_value, gate = attention.contiguous(), kv_value.contiguous(), gate.contiguous()
    grad = grad.contiguous()
    rows = kv_value.shape[:-2]
    kv_heads = kv_value.shape[-2]
    heads = attention.shape[-1] // HEAD_DIM
    group = heads // kv_heads
    numel = attention.numel()
    products = torch.empty(*rows, heads, HEAD_DIM, dtype=torch.float32, device=attention.device)
    squares = torch.empty_like(products)
    _xsa_forward_terms[_grid(numel)](
        attention, kv_value, products, squares, numel, GROUP=group, BLOCK=_BLOCK, **_EXACT_LAUNCH
    )
    dot = products.sum(dim=-1, keepdim=True)
    denominator = squares.sum(dim=-1, keepdim=True) + GRUG_XSA_EPS
    quotient = dot / denominator
    sigmoid = torch.sigmoid(gate.float())
    scale = GRUG_ATTN_GATE_SCALE * sigmoid
    scale_terms, quotient_terms = torch.empty_like(products), torch.empty_like(products)
    _xsa_backward_terms[_grid(numel)](
        grad,
        attention,
        kv_value,
        quotient,
        scale,
        scale_terms,
        quotient_terms,
        numel,
        GROUP=group,
        BLOCK=_BLOCK,
        **_EXACT_LAUNCH,
    )
    d_scale = scale_terms.sum(dim=-1, keepdim=True)
    d_quotient = quotient_terms.sum(dim=-1, keepdim=True)
    d_gate = torch.ops.aten.sigmoid_backward(d_scale.view(gate.shape) * GRUG_ATTN_GATE_SCALE, sigmoid)
    d_dot = d_quotient / denominator  # DivBackward0: grad / other
    # DivBackward0: -grad * ((self / other) / other)
    d_denominator = -d_quotient * ((dot / denominator) / denominator)
    d_attention = torch.empty_like(attention)
    d_value_heads = torch.empty(*rows, heads, HEAD_DIM, dtype=kv_value.dtype, device=kv_value.device)
    _xsa_backward_inputs[_grid(numel)](
        grad,
        attention,
        kv_value,
        quotient,
        scale,
        d_dot,
        d_denominator,
        d_attention,
        d_value_heads,
        numel,
        GROUP=group,
        BLOCK=_BLOCK,
        **_EXACT_LAUNCH,
    )
    # repeat_interleave is an expand of the unsqueezed value: its backward sums each group's heads (sum_to).
    d_value = d_value_heads.view(*rows, kv_heads, group, HEAD_DIM).sum([len(rows) + 1], keepdim=True)
    return d_attention, d_value.view(kv_value.shape), d_gate.to(gate.dtype)


def xsa_head_gate_value(
    value: torch.Tensor, attention: torch.Tensor, kv_value: torch.Tensor, gate: torch.Tensor
) -> torch.Tensor:
    """``value``, differentiated as XSA of ``attention`` against its group's ``kv_value`` head, times the
    ``2 * sigmoid(gate)`` head gate, in fp32 and rounded once."""
    return _with_kernel_gradients(value, _xsa_head_gate_gradients, attention, kv_value, gate)


def _query_key_gradients(grad, raw, freqs, *, multiplier, multiplier_scale, query):
    """The gradient of ``raw`` for one output of ``query_key_values``' chain."""
    raw, grad = raw.contiguous(), grad.contiguous()
    rsqrt = torch.rsqrt(raw.float().square().mean(dim=-1, keepdim=True) + GRUG_QK_RMS_NORM_EPS)
    rotary = freqs is not None
    if rotary:
        if freqs.shape[-1] != ROTARY_DIM:
            raise NotImplementedError(f"the q/k gradient kernels rotate {ROTARY_DIM} of each head's dims")
        cos = torch.cos(freqs).to(torch.bfloat16).float().contiguous()
        sin = torch.sin(freqs).to(torch.bfloat16).float().contiguous()
    else:
        cos = sin = rsqrt  # unread
    numel = raw.numel()
    scaled = torch.empty(raw.shape, dtype=torch.float32, device=raw.device)
    products = torch.empty_like(scaled)
    _qk_backward_terms[_grid(numel)](
        grad,
        raw,
        rsqrt.contiguous(),
        cos,
        sin,
        scaled,
        products,
        numel,
        multiplier,
        multiplier_scale,
        HEADS_PER_POSITION=raw.shape[-3] * raw.shape[-2],
        ROTARY=rotary,
        QUERY=query,
        BLOCK=_BLOCK,
        **_EXACT_LAUNCH,
    )
    d_rsqrt = products.sum(dim=-1, keepdim=True)
    d_mean = (-0.5 * d_rsqrt) * rsqrt.pow(3)  # RsqrtBackward0; the eps add passes the gradient through
    d_raw = torch.empty_like(raw)
    _qk_backward_inputs[_grid(numel)](scaled, raw, d_mean.contiguous(), d_raw, numel, BLOCK=_BLOCK, **_EXACT_LAUNCH)
    return d_raw, None


def query_key_values(
    query_value: torch.Tensor,
    key_value: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    freqs: torch.Tensor | None,
    multiplier: float,
    multiplier_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The vLLM query and key values, differentiated as Grug's fp32 per-head RMS norm of the raw ``[S, B, heads,
    dim]`` projections, then NeoX RoPE with the bf16 cos/sin table and the query scale at compiled vLLM's rounding
    points."""
    query_gradients = functools.partial(
        _query_key_gradients, multiplier=multiplier, multiplier_scale=multiplier_scale, query=True
    )
    key_gradients = functools.partial(_query_key_gradients, multiplier=1.0, multiplier_scale=1.0, query=False)
    return (
        _with_kernel_gradients(query_value, query_gradients, query, freqs),
        _with_kernel_gradients(key_value, key_gradients, key, freqs),
    )


def _logits_gradients(grad, input, weight):
    # Autograd of F.linear(input.float(), weight.float()) on [S, B, K] input runs MmBackward0 on the [S * B, K]
    # rows: grad.mm(weight) for the input and grad.t().mm(rows) for the weight, each rounded to bf16.
    rows = input.reshape(-1, input.shape[-1]).float()
    grad_rows = grad.reshape(-1, grad.shape[-1]).float()
    d_input = grad_rows.mm(weight.float()).view(input.shape).to(input.dtype)
    d_weight = grad_rows.t().mm(rows).to(weight.dtype)
    return d_input, d_weight


def logits_value(value: torch.Tensor, input: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """``value``, differentiated as ``F.linear(input.float(), weight.float())``."""
    return _with_kernel_gradients(value, _logits_gradients, input, weight)
