"""Qwen3-Next GatedDeltaNet (GDN) kernel routing for 80B production RL.

Two problems this module solves, both surfaced by Stage-7/Stage-8 (see
notes/skyrl/stage8_scope.md and stage7_scope.md):

1. **fla masking (always, when the fla overlay is mounted).** The
   `flash-linear-attention==0.5.0` wheel/sdist installed in the Stage-8
   overlay is BROKEN — it ships only `fla/layers` + `fla/models` and drops
   `fla.modules` / `fla.ops` / `fla.utils`. transformers' module-level
   `from fla.modules import FusedRMSNormGated` +
   `from fla.ops.gated_delta_rule import chunk_gated_delta_rule` therefore
   HARD-CRASH the Qwen3-Next modeling import whenever
   `is_flash_linear_attention_available()` returns True. So we MASK it False
   before the modeling module is imported. transformers then falls back to its
   own `Qwen3NextRMSNormGated` + `torch_chunk_gated_delta_rule` — the
   autograd-differentiable pure-torch path that the Stage-7 capstone trained on
   (jobs 596157/596282, finite loss + grad). This mask is REQUIRED for any run
   that mounts the fla overlay; without it the import dies.

2. **FlashQLA fused tilelang kernel (selected by ``generator.gdn_backend``).** The
   pure-torch GDN path is correct but slow (Stage-8: 27x slower at S=8192).
   When the backend is ``flashqla``, after a model is constructed we rebind every
   `Qwen3NextGatedDeltaNet.chunk_gated_delta_rule` instance attribute to a
   FlashQLA shim that re-wraps FlashQLA's *functional* fwd/bwd tilelang kernels
   in a torch-2.9-correct autograd.Function (contiguity-fixed + grad-arity
   fixed). Requires the Stage-8 fla_tilelang overlay (tilelang 0.1.8 +
   FlashQLA + apache-tvm-ffi 0.1.9). Falls back to the pure-torch path (a
   warning, not a crash) if FlashQLA can't be imported.

Usage (call BEFORE transformers' qwen3_next modeling module is imported, then
again on each constructed model):

    from skyrl_train.models.qwen3_next_gdn import mask_fla, engage_flashqla
    mask_fla()                 # always — keeps the modeling import from crashing
    ... model = AutoModelForCausalLM.from_pretrained(...) ...
    engage_flashqla(model, enabled=gdn_backend == "flashqla")
"""

import logging

logger = logging.getLogger(__name__)

_FLA_MASKED = False


def mask_fla() -> bool:
    """Force transformers' `is_flash_linear_attention_available()` to False.

    Idempotent. Returns the pre-mask availability value. Must run before the
    qwen3_next modeling module is imported (the bad `from fla...` lines run at
    module scope). Safe to call when fla is absent (the lambda is harmless).
    """
    global _FLA_MASKED
    try:
        import transformers.utils.import_utils as _iu
    except Exception:  # pragma: no cover - transformers always present
        return False
    try:
        _iu.is_flash_linear_attention_available.cache_clear()
    except Exception:
        pass
    try:
        was = _iu.is_flash_linear_attention_available()
    except Exception:
        was = False
    _iu.is_flash_linear_attention_available = lambda: False
    _FLA_MASKED = True
    logger.info("[gdn] masked fla-availability False (was=%s)", was)
    return was


def _build_flashqla_chunk():
    """Build the FlashQLA chunk_gated_delta_rule drop-in. Returns None if the
    Stage-8 overlay (tilelang + flash_qla) is not importable."""
    import torch

    try:
        from flash_qla import (
            chunk_gated_delta_rule_fwd as _fqla_fwd,
            chunk_gated_delta_rule_bwd as _fqla_bwd,
        )
        from flash_qla.utils import l2norm as _fqla_l2norm
    except Exception as e:  # overlay not mounted / import failed
        logger.warning("[gdn] FlashQLA unavailable (%s); staying on pure-torch GDN", e)
        return None

    class _FlashQLAChunk(torch.autograd.Function):
        @staticmethod
        @torch.amp.custom_fwd(device_type="cuda")
        def forward(ctx, q, k, v, g, beta, scale, initial_state, output_final_state, cu_seqlens):
            q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
            g2, A, o, _h, final_state = _fqla_fwd(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                scale=scale,
                initial_state=initial_state,
                output_final_state=output_final_state,
                output_h=False,
                cu_seqlens=cu_seqlens,
            )
            ctx.save_for_backward(q, k, v, g2, beta, A, initial_state, cu_seqlens)
            ctx.scale = scale
            ctx.has_initial_state = initial_state is not None
            return o.to(q.dtype), final_state

        @staticmethod
        @torch.amp.custom_bwd(device_type="cuda")
        def backward(ctx, do, dht):
            q, k, v, g2, beta, A, initial_state, cu_seqlens = ctx.saved_tensors
            dq, dk, dv, db, dg, dh0 = _fqla_bwd(
                q=q,
                k=k,
                v=v,
                g=g2,
                beta=beta,
                A=A,
                do=do.contiguous(),
                dht=dht,
                scale=ctx.scale,
                initial_state=initial_state,
                cu_seqlens=cu_seqlens,
            )
            d_init = dh0 if ctx.has_initial_state else None
            return (
                dq.to(q.dtype),
                dk.to(k.dtype),
                dv.to(v.dtype),
                dg.to(g2.dtype),
                db.to(beta.dtype),
                None,
                d_init,
                None,
                None,
            )

    def flashqla_chunk(
        q,
        k,
        v,
        g=None,
        beta=None,
        scale=None,
        initial_state=None,
        output_final_state=False,
        use_qk_l2norm_in_kernel=False,
        cu_seqlens=None,
        head_first=False,
    ):
        assert not head_first, "head_first unsupported"
        if scale is None:
            scale = k.shape[-1] ** -0.5
        if use_qk_l2norm_in_kernel:
            q = _fqla_l2norm(q)
            k = _fqla_l2norm(k)
        return _FlashQLAChunk.apply(q, k, v, g, beta, scale, initial_state, output_final_state, cu_seqlens)

    flashqla_chunk._flashqla = True
    return flashqla_chunk


_FLASHQLA_FN = None


def engage_flashqla(model, *, enabled: bool = False) -> int:
    """Rebind every Qwen3NextGatedDeltaNet's chunk kernel to FlashQLA.

    No-op (returns 0) unless enabled by typed generator config. Builds the
    FlashQLA shim once (cached). Returns the number of GDN modules rebound.
    Safe to call on non-Qwen3-Next models (returns 0).
    """
    global _FLASHQLA_FN
    if not enabled:
        return 0
    if _FLASHQLA_FN is None:
        _FLASHQLA_FN = _build_flashqla_chunk()
    if _FLASHQLA_FN is None:
        return 0
    n = 0
    for m in model.modules():
        if type(m).__name__ == "Qwen3NextGatedDeltaNet":
            m.chunk_gated_delta_rule = _FLASHQLA_FN
            n += 1
    if n:
        logger.info("[gdn] engaged FlashQLA fused kernel on %d GatedDeltaNet modules", n)
    return n
