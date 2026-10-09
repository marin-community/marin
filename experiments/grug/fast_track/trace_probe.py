# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Per-token trace of how an AttnRes model builds each prediction, operation by operation.

The final hidden state is the final AttnRes gate's mix ``sum_s w_s src_s`` (plus any gate variants), and every layer
block is that layer's attention output plus its MLP output, so the pre-norm state splits exactly into terms: the
extra sources and the embedding, then ``attn_L`` and ``mlp_L`` for each layer, then ``rest`` (what the gate adds
beyond the weighted sum). With the final RMSNorm's per-token scale held fixed the logits split the same way, so:

- ``dla_target`` / ``dla_top1``: each term's logit for the target / the model's top prediction, centred over the
  vocabulary (direct logit attribution);
- ``loo_loss``: the loss with one term's direct contribution removed (its effect through later layers is kept);
- ``lens_loss`` / ``lens_top1``: the loss and top prediction from the terms up to each depth, re-normalised
  (a logit lens through the final gate's weights);
- MLA top keys per query and head (``trace_capture.TOPK_FIELDS``) and each layer's routed experts.
"""

import jax
import jax.numpy as jnp
from jaxtyping import Array, Int
from levanter.grug.attention import AttentionMask

from experiments.grug.fast_track.model import (
    _LAYER_KNOB_PREFIX,
    PROBE_STAT,
    ROUTING_SELECTED_KEY,
    ROUTING_WEIGHTS_KEY,
    Transformer,
    _batch_spec,
    _logit_cap,
    _soft_capped,
)
from experiments.grug.fast_track.trace_capture import trace_capture

_TOP_PREDICTIONS = 5


def trace_term_names(model: Transformer, num_extra: int) -> list[str]:
    extras = (["embed2"] if model.config.second_embed else []) + [f"extra{j}" for j in range(num_extra)]
    names = [*extras[:num_extra], "embed"]
    for i in range(model.config.num_layers):
        names += [f"attn{i}", f"mlp{i}"]
    return [*names, "rest"]


def trace_stats(model: Transformer, ids: Int[Array, "B S"], segments: Int[Array, "B S"]) -> dict[str, Array]:
    """Per-token trace arrays (see the module docstring) plus scalar reconstruction errors ``check_*``."""
    cfg = model.config
    if model.final_gated_norm is not None or model.lm_head_extra_norm is not None:
        raise ValueError("trace_stats linearises a plain final RMSNorm (no final_gated_norm or lm_head_extra_dim)")
    f32 = jnp.float32
    spec = _batch_spec()
    with trace_capture():
        hidden, metrics = model(ids, mask=AttentionMask.causal().with_segment_ids(segments), return_routing=True)
    layers = cfg.num_layers
    src = metrics[f"{PROBE_STAT}trace_final_sources"].astype(f32)  # [N, B, S, D]
    weights = metrics[f"{PROBE_STAT}trace_final_weights"].astype(f32)  # [N, B, S]
    mixed = metrics[f"{PROBE_STAT}trace_final_mixed"].astype(f32)
    num_extra = src.shape[0] - 1 - layers
    if num_extra < 0:
        raise ValueError(f"trace expects one AttnRes block per layer, got {src.shape[0]} final sources")
    attn = [metrics[f"{_LAYER_KNOB_PREFIX}trace_attn_out_L{i}"].astype(f32) for i in range(layers)]
    mlp = [metrics[f"{_LAYER_KNOB_PREFIX}trace_mlp_out_L{i}"].astype(f32) for i in range(layers)]
    block_err = jnp.max(jnp.stack([jnp.max(jnp.abs(src[num_extra + 1 + i] - attn[i] - mlp[i])) for i in range(layers)]))
    terms = [weights[j][..., None] * src[j] for j in range(num_extra + 1)]
    for i in range(layers):
        w = weights[num_extra + 1 + i][..., None]
        terms += [w * attn[i], w * mlp[i]]
    terms.append(mixed - sum(terms))

    # Final RMSNorm, linearised at the full state: c_t = gain * t / rms(mixed).
    eps = cfg.layer_norm_eps
    d = cfg.hidden_dim
    gain = model.final_norm(jnp.ones((d,), f32)).astype(f32) * jnp.sqrt(1.0 + eps)
    inv_rms = jax.lax.rsqrt(jnp.mean(jnp.square(mixed), axis=-1, keepdims=True) + eps)
    normed = model.final_norm(mixed.astype(hidden.dtype)).astype(f32)
    norm_err = jnp.max(jnp.abs(normed - hidden.astype(f32)))
    head_in, head = model._lm_head_operands(hidden, ids)
    head = jax.sharding.reshard(head.astype(f32), jax.sharding.PartitionSpec(None, None))
    w_out = head[:d]
    extra_logits = None
    if head_in.shape[-1] > d:
        extra_logits = jnp.einsum("bse,ev->bsv", head_in[..., d:].astype(f32), head[d:], out_sharding=spec)
    cap = _logit_cap(cfg)

    def to_logits(h_normed):
        logits = jnp.einsum("bsd,dv->bsv", h_normed, w_out, out_sharding=spec)
        return logits if extra_logits is None else logits + extra_logits

    full = to_logits(hidden.astype(f32))
    logp = jax.nn.log_softmax(_soft_capped(full, cap), axis=-1)
    labels = jnp.pad(ids[:, 1:], ((0, 0), (0, 1)))
    pick = lambda x, idx: jnp.take_along_axis(x, idx[..., None], axis=-1)[..., 0]  # noqa: E731
    loss = -pick(logp, labels)
    top_logp, top_ids = jax.lax.top_k(logp, _TOP_PREDICTIONS)
    top1 = top_ids[..., 0]

    dla_target, dla_top1, loo_loss = [], [], []
    recon = jnp.zeros_like(full) if extra_logits is None else extra_logits
    for term in terms:
        term_logits = jnp.einsum("bsd,dv->bsv", term * gain * inv_rms, w_out, out_sharding=spec)
        recon = recon + term_logits
        mean = jnp.mean(term_logits, axis=-1)
        dla_target.append(pick(term_logits, labels) - mean)
        dla_top1.append(pick(term_logits, top1) - mean)
        loo_loss.append(-pick(jax.nn.log_softmax(_soft_capped(full - term_logits, cap), axis=-1), labels))
    logit_err = jnp.max(jnp.abs(recon - full))

    lens_loss, lens_top1 = [], []
    cumulative = sum(terms[: num_extra + 1])
    stages = [cumulative]
    for term in terms[num_extra + 1 :]:
        cumulative = cumulative + term
        stages.append(cumulative)
    for stage in stages:
        lens_logp = jax.nn.log_softmax(_soft_capped(to_logits(model.final_norm(stage).astype(f32)), cap), axis=-1)
        lens_loss.append(-pick(lens_logp, labels))
        lens_top1.append(jnp.argmax(lens_logp, axis=-1))

    out = {
        "loss": loss,
        "entropy": -jnp.sum(jnp.exp(logp) * logp, axis=-1),
        "top_ids": top_ids,
        "top_p": jnp.exp(top_logp),
        "dla_target": jnp.stack(dla_target, axis=-1),
        "dla_top1": jnp.stack(dla_top1, axis=-1),
        "loo_loss": jnp.stack(loo_loss, axis=-1),
        "lens_loss": jnp.stack(lens_loss, axis=-1),
        "lens_top1": jnp.stack(lens_top1, axis=-1),
        "check_block_err": block_err,
        "check_norm_err": norm_err,
        "check_logit_err": logit_err,
        "routing_selected": metrics[ROUTING_SELECTED_KEY],
        "routing_weights": metrics[ROUTING_WEIGHTS_KEY],
    }
    for name, value in metrics.items():
        if "trace_topk" in name:
            out[name.removeprefix(_LAYER_KNOB_PREFIX)] = value
    return out
