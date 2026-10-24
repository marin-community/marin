import pytest
import torch
import torch.nn.functional as F

from skyrl_train.models.grug_exact.kernel_values import kernel_value
from skyrl_train.models.grug_exact.reference_kernels import (
    HEAD_DIM,
    ROTARY_DIM,
    gated_product_value,
    hybrid_input_norm_value,
    query_key_values,
    swiglu_value,
    xsa_head_gate_value,
)
from skyrl_train.models.grug_exact.support import VERIFIED_MODEL
from skyrl_train.models.grug_moe import GRUG_ATTN_GATE_SCALE, GRUG_QK_RMS_NORM_EPS, GRUG_XSA_EPS
from tests.gpu.grug_gpu_gates import require_hoppers

# 3,328 rows put every non-NaN bf16 bit pattern into the head-gate logits ([rows, 1, 20]).
ROWS = 3328
HIDDEN, HEADS, KV_HEADS = (
    VERIFIED_MODEL.hidden_size,
    VERIFIED_MODEL.num_attention_heads,
    VERIFIED_MODEL.num_key_value_heads,
)


def _swiglu_chain(fc1_output):
    gate, up = torch.chunk(fc1_output, 2, dim=-1)
    return (F.silu(gate.float()) * up.float()).to(fc1_output.dtype)


def _hybrid_norm_chain(hidden, variance, weight, eps):
    variance = kernel_value(variance, lambda: hidden.float().pow(2).mean(dim=-1, keepdim=True))
    return (hidden.float() * torch.rsqrt(variance + eps) * weight.float()).to(weight.dtype)


def _xsa_gate_chain(core_attn_out, value, gate):
    heads = core_attn_out.shape[-1] // HEAD_DIM
    attention = core_attn_out.view(*value.shape[:-2], heads, HEAD_DIM).float()
    v = value.repeat_interleave(heads // value.shape[-2], dim=-2).float()
    dot = (attention * v).sum(dim=-1, keepdim=True)
    projected = attention - (dot / (v.square().sum(dim=-1, keepdim=True) + GRUG_XSA_EPS)) * v
    scale = (GRUG_ATTN_GATE_SCALE * torch.sigmoid(gate.float())).view(*gate.shape, 1)
    return (projected * scale).to(core_attn_out.dtype).reshape(core_attn_out.shape)


def _qk_norm(raw):
    fp32 = raw.float()
    return fp32 * torch.rsqrt(fp32.square().mean(dim=-1, keepdim=True) + GRUG_QK_RMS_NORM_EPS)


def _neox_rope(x, angles):
    """vLLM's NeoX rotation of the first ``ROTARY_DIM`` dims in fp32 with its bf16 cos/sin table."""
    cos = angles.cos().to(torch.bfloat16).float()[:, None, None, :]
    sin = angles.sin().to(torch.bfloat16).float()[:, None, None, :]
    first, second = x[..., : ROTARY_DIM // 2], x[..., ROTARY_DIM // 2 : ROTARY_DIM]
    return torch.cat((first * cos - second * sin, second * cos + first * sin, x[..., ROTARY_DIM:]), dim=-1)


def _query_key_chain(query, key, angles, multiplier):
    """Compiled vLLM's rounding points for the normalized q and k: k rounds once after RoPE; q's rotated half rounds
    before the scale and again after it, its pass-through half once."""
    if angles is not None:
        query = _neox_rope(query, angles)
        query = torch.cat((query[..., :ROTARY_DIM].to(torch.bfloat16).float(), query[..., ROTARY_DIM:]), dim=-1)
        key = _neox_rope(key, angles)
    return (query.float() * multiplier).to(torch.bfloat16), key.to(torch.bfloat16)


def _every_bf16(shape, generator) -> torch.Tensor:
    """bf16 values holding every non-NaN bit pattern (then random ones), shuffled into ``shape``."""
    patterns = torch.arange(-(2**15), 2**15, dtype=torch.int32).to(torch.int16).view(torch.bfloat16)
    patterns = patterns[~patterns.float().isnan()]
    count = int(torch.tensor(shape).prod())
    picks = patterns[torch.randint(0, patterns.numel(), (count,), generator=generator)]
    picks[: min(count, patterns.numel())] = patterns[: min(count, patterns.numel())]
    return picks[torch.randperm(count, generator=generator)].view(shape).cuda()


def _activations(shape, generator) -> torch.Tensor:
    """bf16 values from fp32-subnormal to 1e3 in magnitude, with runs of +0 and -0."""
    scale = 10.0 ** torch.empty(shape).uniform_(-42, 3, generator=generator)
    values = (torch.randn(shape, generator=generator) * scale).to(torch.bfloat16)
    values.view(-1)[:64] = 0.0
    values.view(-1)[64:128] = -0.0
    return values.cuda()


def _leaves(*tensors):
    return [tensor.detach().clone().requires_grad_() for tensor in tensors]


def _assert_same_bits(eager, fused):
    for eager_grad, fused_grad in zip(eager, fused, strict=True):
        width = torch.int16 if eager_grad.dtype == torch.bfloat16 else torch.int32
        assert torch.equal(eager_grad.contiguous().view(width), fused_grad.contiguous().view(width))


def _gated_product(generator):
    normalized, gate = _activations((ROWS, 1, HIDDEN), generator), _every_bf16((ROWS, 1, HIDDEN), generator)
    value, grad = _activations((ROWS, 1, HIDDEN), generator), _activations((ROWS, 1, HIDDEN), generator)
    results = []
    for fused in (False, True):
        n, g = _leaves(normalized, gate)
        if fused:
            out = gated_product_value(value, n, g)
        else:
            out = kernel_value(value, lambda n=n, g=g: (n.float() * torch.sigmoid(g.float())).to(torch.bfloat16))
        out.backward(grad)
        results.append((n.grad, g.grad))
    return results


def _swiglu(generator):
    fc1 = torch.cat((_every_bf16((ROWS, 1, HIDDEN), generator), _activations((ROWS, 1, HIDDEN), generator)), dim=-1)
    value, grad = _activations((ROWS, 1, HIDDEN), generator), _activations((ROWS, 1, HIDDEN), generator)
    results = []
    for fused in (False, True):
        (f,) = _leaves(fc1)
        out = swiglu_value(value, f) if fused else kernel_value(value, lambda f=f: _swiglu_chain(f))
        out.backward(grad)
        results.append((f.grad,))
    return results


def _hybrid_input_norm(generator):
    # The layer input feeds the norm, the variance's reference and the residual add after the attention, so its
    # gradient adds three contributions; the order autograd adds them in is part of the check.
    hidden, weight = _activations((ROWS, 1, HIDDEN), generator), _activations((HIDDEN,), generator)
    variance = (torch.rand(ROWS, 1, 1, generator=generator) * 3 + 1e-3).cuda()
    variance[:4] = torch.tensor([1e-30, 3e-38, 5e-6, 1e30]).view(4, 1, 1)
    value, grad = _activations((ROWS, 1, HIDDEN), generator), _activations((ROWS, 1, HIDDEN), generator)
    attention, residual_grad = _activations((ROWS, 1, HIDDEN), generator), _activations((ROWS, 1, HIDDEN), generator)
    results = []
    for fused in (False, True):
        h, w = _leaves(hidden, weight)
        if fused:
            normalized = hybrid_input_norm_value(value, h, variance, w, 1e-5)
        else:
            normalized = kernel_value(value, lambda h=h, w=w: _hybrid_norm_chain(h, variance, w, 1e-5))
        torch.autograd.backward((normalized, h + attention), (grad, residual_grad))
        results.append((h.grad, w.grad))
    return results


def _xsa_head_gate(generator):
    attention = _activations((ROWS, 1, HEADS * HEAD_DIM), generator)
    kv_value = _activations((ROWS, 1, KV_HEADS, HEAD_DIM), generator)
    gate = _every_bf16((ROWS, 1, HEADS), generator)
    value, grad = _activations((ROWS, 1, HIDDEN), generator), _activations((ROWS, 1, HIDDEN), generator)
    results = []
    for fused in (False, True):
        a, v, g = _leaves(attention, kv_value, gate)
        if fused:
            out = xsa_head_gate_value(value, a, v, g, HEAD_DIM)
        else:
            out = kernel_value(value, lambda a=a, v=v, g=g: _xsa_gate_chain(a, v, g))
        out.backward(grad)
        results.append((a.grad, v.grad, g.grad))
    return results


def _query_key(generator, rope: bool):
    raw_query = _activations((ROWS, 1, HEADS, HEAD_DIM), generator)
    raw_key = _activations((ROWS, 1, KV_HEADS, HEAD_DIM), generator)
    query_grad = _activations((ROWS, 1, HEADS, HEAD_DIM), generator)
    key_grad = _activations((ROWS, 1, KV_HEADS, HEAD_DIM), generator)
    multiplier = VERIFIED_MODEL.qk_mult
    inverse = 1.0 / (10000.0 ** (torch.arange(0, ROTARY_DIM, 2, dtype=torch.float32) / ROTARY_DIM))
    angles = torch.outer(torch.arange(ROWS, dtype=torch.float32), inverse).cuda() if rope else None
    freqs = torch.cat((angles, angles), dim=-1).view(ROWS, 1, 1, ROTARY_DIM) if rope else None
    results = []
    for fused in (False, True):
        q, k = _leaves(raw_query, raw_key)
        query_value, key_value = q.detach().clone(), k.detach().clone()
        if fused:
            out_q, out_k = query_key_values(
                query_value, key_value, q, k, (freqs, freqs) if rope else None, multiplier, 1.0
            )
        else:
            ref_q, ref_k = _query_key_chain(_qk_norm(q), _qk_norm(k), angles, multiplier)
            out_q = kernel_value(query_value, lambda ref_q=ref_q: ref_q)
            out_k = kernel_value(key_value, lambda ref_k=ref_k: ref_k)
        torch.autograd.backward((out_q, out_k), (query_grad, key_grad))
        results.append((q.grad, k.grad))
    return results


@pytest.mark.parametrize(
    "chain",
    [
        _gated_product,
        _swiglu,
        _hybrid_input_norm,
        _xsa_head_gate,
        lambda generator: _query_key(generator, rope=True),
        lambda generator: _query_key(generator, rope=False),
    ],
    ids=["gated_product", "swiglu", "hybrid_input_norm", "xsa_head_gate", "query_key_rope", "query_key"],
)
def test_reference_gradients_equal_autograd_of_their_chains_bit_for_bit(chain):
    require_hoppers(1)
    eager, fused = chain(torch.Generator().manual_seed(0))
    _assert_same_bits(eager, fused)
