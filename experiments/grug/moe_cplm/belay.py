# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Belay backbone with a tiled copy-paste output head and softcap-10 logits."""

import dataclasses

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P
from levanter.kernels.pallas.fused_cross_entropy_loss import (
    BlockSizes, fused_cross_entropy_loss_and_logsumexp_penalty,
)

from experiments.grug.fast_track import model
from experiments.grug.moe_cplm.head import copy_log_probabilities, mixture_nll


def capped(x, cap=10.0):
    return cap * jnp.tanh(x / cap)


def local_head(hidden, head, qw, kw, gw, sink, tokens, labels, segments,
               *, tile=128, reference=False):
    batch, length, dim = hidden.shape
    x = hidden.reshape(-1, dim)
    if reference:
        logits = capped(x.astype(jnp.float32) @ head.astype(jnp.float32))
        z = jax.nn.logsumexp(logits, -1)
        ce = z - jnp.take_along_axis(logits, labels.reshape(-1, 1), -1)[:, 0]
    else:
        ce = fused_cross_entropy_loss_and_logsumexp_penalty(
            x, labels.reshape(-1), head, reduction=None, logsumexp_weight=None,
            dtype=jnp.float32, logit_soft_cap=10.0, implementation="pallas_tpu",
            # TPU v4 has 16 MiB VMEM. The forward kernel buffers four V
            # subtiles and full H: d768/B8192 spills with V512 (16.75 MiB).
            # Smaller vocabulary tiles preserve the exact softmax objective.
            block_sizes=BlockSizes(b_block_size=1024, h_block_size=128,
                                   v_block_size=512))
        target = jnp.take(head, labels.reshape(-1), axis=1).T
        target_logit = capped(jnp.sum(x.astype(jnp.float32)*target.astype(jnp.float32), -1))
        z = ce + target_logit
    ce, z = ce.reshape(batch, length), z.reshape(batch, length)
    q = jnp.einsum("bsd,dh->bsh", hidden, qw, preferred_element_type=jnp.float32)
    k = jnp.einsum("bsd,dh->bsh", hidden, kw, preferred_element_type=jnp.float32)
    g = capped(jnp.sum(hidden.astype(jnp.float32)*gw.astype(jnp.float32), -1))
    sink_logp, target_logp = copy_log_probabilities(q, k, sink, tokens, labels, segments, query_tile=tile)
    mass = -jnp.expm1(sink_logp)
    nll = mixture_nll(ce, z, g, sink_logp, target_logp)
    gate = jax.nn.sigmoid(g-z)
    return nll, ce, gate, gate*mass


class CopyPasteBelay(model.Transformer):
    cp_query: jax.Array
    cp_key: jax.Array
    cp_gate: jax.Array
    cp_sink: jax.Array

    @staticmethod
    def init(cfg, *, key):
        base = model.Transformer.init(cfg, key=key)
        keys = jax.random.split(jax.random.fold_in(key, 261019247), 4)
        d = cfg.hidden_dim
        arrays = [jax.sharding.reshard(.02*jax.random.normal(k, shape), P())
                  for k, shape in zip(keys, ((d,d),(d,d),(d,),(d,)), strict=True)]
        return CopyPasteBelay(**{f.name: getattr(base, f.name) for f in dataclasses.fields(base)},
                             cp_query=arrays[0], cp_key=arrays[1], cp_gate=arrays[2], cp_sink=arrays[3])

    def next_token_loss(self, token_ids, loss_weight, *, mask=None, reduction="mean",
                        logsumexp_weight=None, loss_dtype=jnp.float32, return_router_metrics=False,
                        aux_loss_weight=None, loop_active=None, train_terms=False, route_key=None,
                        head_replay=None, byte_table=None, byte_aux_weight=None,
                        router_tie_active=None, future_aux_weight=None):
        if logsumexp_weight not in (None, 0, 0.0):
            raise ValueError("CPLM baseline uses next-token NLL without vocabulary z-loss")
        if any(x is not None for x in (head_replay, byte_table, byte_aux_weight, future_aux_weight)):
            raise ValueError("Auxiliary heads are outside the frozen Belay CPLM recipe")
        hidden, metrics = self(token_ids, mask=mask, loop_active=loop_active,
                               route_key=route_key, router_tie_active=router_tie_active)
        # The frozen recipe does not use auxiliary prediction/compression losses.
        for name in (model._AUX_HIDDEN, model._NITP_TARGET, model._ATTN_RES_Z,
                     model._MTP_EMBED, model._COMPRESS_LOSS):
            metrics.pop(name, None)
        labels = jnp.pad(token_ids[:, 1:], ((0,0),(0,1)))
        if mask is not None and not mask.is_causal:
            raise ValueError("Copy predictions require a causal mask")
        segments = jnp.zeros_like(token_ids) if mask is None or mask.segment_ids is None else mask.segment_ids[0]
        segments = jnp.broadcast_to(segments, token_ids.shape)
        ts, hs = P(model._BATCH_AXES, None), P(model._BATCH_AXES, None, None)
        specs = (hs,P(),P(),P(),P(),P(),ts,ts,ts)
        args = (hidden,self.output_proj,self.cp_query,self.cp_key,self.cp_gate,self.cp_sink,
                token_ids,labels,segments)
        args = tuple(jax.sharding.reshard(a,s) for a,s in zip(args,specs,strict=True))
        nll, ce, gate, effective = jax.shard_map(local_head, in_specs=specs,
                out_specs=(ts,ts,ts,ts), check_vma=False)(*args)
        weights = loss_weight.astype(loss_dtype)
        denom = jnp.maximum(weights.sum(),1)
        weighted = nll*weights
        if reduction == "none":
            loss = weighted
        elif reduction == "sum":
            loss = weighted.sum()
        elif reduction == "mean":
            loss = weighted.sum()/denom
        else:
            raise ValueError(reduction)
        if not return_router_metrics:
            return loss
        final = {f"train/attn_res/{k.removeprefix('attn_res_')}": metrics.pop(k)
                 for k in list(metrics) if k.startswith("attn_res_")}
        result = model.summarize_router_metrics(metrics)
        result.update(final)
        result.update({"train/cross_entropy_loss":loss,
            "train/aux/cplm/native_head_nll":jnp.sum(ce*weights)/denom,
            "train/aux/cplm/copy_gate":jnp.sum(gate*weights)/denom,
            "train/aux/cplm/effective_copy":jnp.sum(effective*weights)/denom,
            "train/router/z_loss_logging_only":jnp.mean(metrics["router_z_loss_per_layer"]),
            "moe/dropped_assignments":metrics["capacity_overflow_per_layer"],
            "moe/sender_dropped_assignments":metrics["sender_capacity_overflow_per_layer"],
            "moe/receiver_dropped_assignments":metrics["receiver_capacity_overflow_per_layer"]})
        return loss,result

