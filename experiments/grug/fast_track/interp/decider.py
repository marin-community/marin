# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Decider neurons: units whose strong fires predict that the full model is confidently right about the next token.

One pass over held-out tokens accumulates, for every routed unit ``(layer, expert, unit)`` and every shared unit, a
histogram over its activation (``NB`` bins, eighth-octave steps relative to the unit's mean top-16 activation of the
``stats`` pass; bin 0 = routed but zero, bin 1 = below ``2^LOG2_LO`` of that scale) of the next-token outcomes and the
unit's direct logit contribution to the target (``STATS``). On the host the histograms reduce to cumulative stats above
per-unit quantile cuts of the unit's nonzero activations (``LEVELS``): the cuts come from this pass (``discover``) or are
read from an earlier one (``replicate``, so held-out stats use the identical thresholds).

Per-token outcomes (loss-weighted positions with a same-document target): ``crX`` = top-1 correct and
``p(target) > X``; ``cwX`` = top-1 wrong with ``p(top-1) > X``. The direct logit of a unit is its lag-0 path through the
MLP SConv, its combine weight, ``w_latent_up``, the final AttnRes weight of its layer's block, the final RMSNorm scale
and gain and the centred unembedding column of the target; ``tlogit`` is the target's total centred (pre-cap) logit.

``replicate`` also takes a registry (``decider_registry.json``) of decider sets and units and records, on its tokens:
per token and set the earliest layer with a strong-firing set member and the member counts (coverage); one random
strong-fire position per registry unit per batch (examples, target types); then mean-ablates the registry's
``ablate`` units at their strong-fire positions (and a same-expert control unit at the same positions).
"""

import argparse
import dataclasses
import json
import logging
import time
from collections.abc import Iterator

import fsspec
import jax
import jax.numpy as jnp
import numpy as np
from haliax import Axis
from haliax.nn.ragged_dot import ragged_dot
from haliax.partitioning import set_mesh
from jax.sharding import PartitionSpec as P
from jax.sharding import get_abstract_mesh

from experiments.grug.fast_track.interp import categories, common, forward

logger = logging.getLogger(__name__)

BATCH = 4
S = common.SEQ_LEN
L, E, I, V, D = forward.NUM_LAYERS, forward.NUM_EXPERTS, forward.EXPERT_WIDTH, forward.VOCAB, forward.HIDDEN
K = forward.TOP_K
SI = forward.SHARED_WIDTH
LOG2_LO = -8
BINS_PER_OCTAVE = 8
NB = 2 + (2 - LOG2_LO) * BINS_PER_OCTAVE  # 82: zero, underflow, [2^-8, 2^2) in eighths (the last bin open)
TOKEN_STATS = ("n", "cr70", "cr90", "cr99", "cw50", "cw90", "top1", "logp", "tlogit")
UNIT_STATS = ("act", "dla", "is_top")
STATS = TOKEN_STATS + UNIT_STATS
# Cumulative levels: every routed slot (bin >= 0), every nonzero fire (bin >= 1), then the top fractions of fires.
LEVEL_FRACTIONS = (0.5, 0.2, 0.1, 0.03, 0.01, 0.003, 0.001)
LEVEL_NAMES = ("routed", "fired", *(f"top{f:g}" for f in LEVEL_FRACTIONS))
MAIN_LEVEL = LEVEL_NAMES.index("top0.01")
SOURCES = ("train", *common.PALOMA_SETS)
NUM_SETS = 8
NUM_REGISTRY = 4096
HASH_A = np.uint32(2654435761)
HASH_B = np.uint32(2246822519)
ABLATE_SEQUENCES = 16
SUBSAMPLE_TOKENS = 2_000_000
OUTCOME_BITS = ("top1", "cr70", "cr90", "cr99", "cw50", "cw90")
TYPE_NAMES = (*categories.CATEGORIES[1:], "ngram3", "bigram")
NGRAM_P, NGRAM_MIN = 0.9, 10


def stats_ref(tag: str) -> tuple[np.ndarray, np.ndarray]:
    """Each routed ``[L, E, I]`` and shared ``[L, SI]`` unit's mean top-16 activation from the ``stats`` pass."""
    with fsspec.open(f"{common.OUTPUT_ROOT}/{tag}/stats.npz", "rb") as f:
        z = np.load(f)
        top, s_top = z["top_val"], z["s_top_val"]

    def ref(x):
        m = np.nanmean(np.where(x > 0, x, np.nan), axis=-1)
        return np.nan_to_num(m, nan=1.0).astype(np.float32)

    return ref(top), ref(s_top)


def unit_bin(act: jax.Array, ref: jax.Array) -> jax.Array:
    r = jnp.maximum(act, 1e-30) / ref
    b = jnp.floor((jnp.log2(r) - LOG2_LO) * BINS_PER_OCTAVE).astype(jnp.int32) + 2
    return jnp.where(act > 0, jnp.clip(b, 1, NB - 1), 0)


def bin_lower_edge(b: np.ndarray) -> np.ndarray:
    """Activation / ref at the lower edge of bin ``b`` (0 for bins 0 and 1)."""
    return np.where(b >= 2, 2.0 ** (LOG2_LO + (b - 2) / BINS_PER_OCTAVE), 0.0)


def unit_directions(model) -> tuple[jax.Array, jax.Array]:
    """Lag-0 residual directions through the final gain: routed ``[L, E, D, I]`` (transposed for ``ragged_dot``) and
    shared ``[L, SI, D]``."""
    gain = forward.final_gain(model)
    routed, shared = [], []
    for lay in model.layers():
        lag0 = lay.sconv_mlp.weight[0].astype(jnp.float32) * gain
        up = lay.mlp.w_latent_up.astype(jnp.float32)
        dirs = jnp.einsum("eir,rd->eid", lay.mlp.expert_mlp.w_down.astype(jnp.float32), up) * lag0
        routed.append(jnp.swapaxes(dirs, 1, 2))
        shared.append(lay.shared[0].w_down.astype(jnp.float32) * lag0)
    return jnp.stack(routed), jnp.stack(shared)


@jax.jit
def top_promoted(model) -> tuple[jax.Array, jax.Array]:
    """Each unit's most promoted token by its centred lag-0 unembedding: routed ``[L, E, I]``, shared ``[L, SI]``."""
    m = forward.replicate(model)
    w_out = m.output_proj.astype(jnp.float32)
    routed, shared = unit_directions(m)

    def one(dirs):  # [N, D]
        return jnp.argmax(dirs @ w_out, axis=-1).astype(jnp.int32)  # centring does not move the argmax

    r = jax.lax.map(lambda d: one(d.T), routed.reshape(L * E, D, I))
    return r.reshape(L, E, I), jax.vmap(one)(shared)


def _ragged(lhs, rhs, counts):
    return jax.shard_map(
        ragged_dot,
        mesh=get_abstract_mesh(),
        in_specs=(P(None, None), P(None, None, None), P(None)),
        out_specs=P(None, None),
        check_vma=False,
    )(lhs, rhs, counts)


def token_outcomes(tr: dict, valid: jax.Array, w_c: jax.Array) -> dict:
    """Flat ``[T]`` outcomes of a replicated trace."""
    logp = -tr["loss"].reshape(-1)
    p = jnp.exp(logp)
    top1 = tr["top_ids"][..., 0].reshape(-1) == tr["labels"].reshape(-1)
    ptop = jnp.exp(tr["top_logp"][..., 0].reshape(-1))
    hidden = tr["hidden"].reshape(-1, D).astype(jnp.float32)
    out = dict(
        logp=logp,
        top1=top1,
        cr70=top1 & (p > 0.7),
        cr90=top1 & (p > 0.9),
        cr99=top1 & (p > 0.99),
        cw50=~top1 & (ptop > 0.5),
        cw90=~top1 & (ptop > 0.9),
        tlogit=jnp.sum(hidden * w_c, axis=-1),
        valid=valid,
    )
    return out


def _token_matrix(o: dict) -> jax.Array:
    """``[T, len(TOKEN_STATS)]`` per-token stat values (zero where not valid)."""
    vf = o["valid"].astype(jnp.float32)
    cols = [jnp.ones_like(vf)] + [o[k].astype(jnp.float32) for k in TOKEN_STATS[1:]]
    return jnp.stack(cols, axis=-1) * vf[:, None]


def outcome_code(o: dict) -> jax.Array:
    code = jnp.zeros_like(o["logp"], jnp.int32)
    for j, k in enumerate(OUTCOME_BITS):
        code = code | (o[k].astype(jnp.int32) << j)
    return code.astype(jnp.uint8)


def _hash(gtok: jax.Array, unit: jax.Array) -> jax.Array:
    h = gtok.astype(jnp.uint32) * HASH_A + unit.astype(jnp.uint32) * HASH_B
    h = h ^ (h >> 15)
    return (h >> 8).astype(jnp.int32)


def make_step(with_registry: bool):
    def step(state, model, dirs, s_dirs, promoted, s_promoted, ref, s_ref, tokens, seg, valid, offset, reg):
        model = forward.replicate(model)
        tr = forward.replicate(forward.trace_forward(model, tokens, seg, forward.no_ablation()))
        t = tokens.size
        labels = tr["labels"].reshape(-1)
        w_out = model.output_proj.astype(jnp.float32)
        w_c = (w_out.T[labels] - jnp.mean(w_out, axis=1)[None]).astype(jnp.float32)  # [T, D]
        o = token_outcomes(tr, valid.reshape(-1), w_c)
        tokm = _token_matrix(o)
        scale = (tr["inv_rms"].reshape(-1)).astype(jnp.float32)
        new = dict(state)
        hists, s_hists = [], []
        extras = {"votes": [], "best_key": [], "best_tok": [], "best_act": []}
        gtok = offset + jnp.arange(t, dtype=jnp.int32)
        for li, lay in enumerate(tr["layers"]):
            act = lay["expert_act"]  # [TK, I]
            ex, ts, slot = lay["expert_sorted"], lay["token_sorted"], lay["slot_sorted"]
            wf = tr["w_final"][li + 1].reshape(-1).astype(jnp.float32) * scale  # [T]
            used = lay["used_combine"][ts, slot]
            proj = _ragged(w_c[ts], dirs[li], lay["expert_counts"])  # [TK, I]
            dla = act * (used * wf[ts])[:, None] * proj
            vf = o["valid"][ts].astype(jnp.float32)[:, None]
            unit_vals = jnp.stack([act, dla, (promoted[li][ex] == labels[ts][:, None]).astype(jnp.float32)], -1)
            vals = jnp.concatenate(
                [jnp.broadcast_to(tokm[ts][:, None, :], (*act.shape, len(TOKEN_STATS))), unit_vals * vf[..., None]],
                axis=-1,
            )
            idx = (ex[:, None] * I + jnp.arange(I)[None, :]) * NB + unit_bin(act, ref[li][ex])
            hists.append(state["hist"][li].at[idx.reshape(-1)].add(vals.reshape(-1, len(STATS))))

            s_act = lay["shared_act"]  # [T, SI]
            s_proj = w_c @ s_dirs[li].T  # [T, SI]
            s_dla = s_act * wf[:, None] * s_proj
            vft = o["valid"].astype(jnp.float32)[:, None]
            s_unit = jnp.stack([s_act, s_dla, (s_promoted[li][None, :] == labels[:, None]).astype(jnp.float32)], -1)
            s_vals = jnp.concatenate(
                [jnp.broadcast_to(tokm[:, None, :], (*s_act.shape, len(TOKEN_STATS))), s_unit * vft[..., None]], -1
            )
            s_idx = jnp.arange(SI)[None, :] * NB + unit_bin(s_act, s_ref[li][None, :])
            s_hists.append(state["s_hist"][li].at[s_idx.reshape(-1)].add(s_vals.reshape(-1, len(STATS))))

            if with_registry:
                strong = (act >= reg["thr"][li][ex]) & (act > 0) & (vf > 0)
                bits = reg["bits"][li][ex]  # [TK, I] int32
                s_strong = (s_act >= reg["s_thr"][li][None, :]) & (s_act > 0) & (vft > 0)
                s_bits = jnp.broadcast_to(reg["s_bits"][li][None, :], s_act.shape)
                votes = []
                for j in range(NUM_SETS):
                    row = jnp.sum(strong & (((bits >> j) & 1) > 0), axis=-1)
                    v = jax.ops.segment_sum(row, ts, num_segments=t)
                    votes.append(v + jnp.sum(s_strong & (((s_bits >> j) & 1) > 0), axis=-1))
                extras["votes"].append(jnp.stack(votes))  # [NUM_SETS, T]
                ridx = reg["idx"][li][ex]  # [TK, I]
                key = jnp.where(strong & (ridx >= 0), _hash(gtok[ts][:, None], jnp.arange(I)[None, :]), -1)
                seg_id = jnp.where(ridx >= 0, ridx, NUM_REGISTRY)
                s_ridx = jnp.broadcast_to(reg["s_idx"][li][None, :], s_act.shape)
                s_key = jnp.where(s_strong & (s_ridx >= 0), _hash(gtok[:, None], 1000 + jnp.arange(SI)[None, :]), -1)
                s_seg = jnp.where(s_ridx >= 0, s_ridx, NUM_REGISTRY)
                all_key = jnp.concatenate([key.reshape(-1), s_key.reshape(-1)])
                all_seg = jnp.concatenate([seg_id.reshape(-1), s_seg.reshape(-1)])
                all_tok = jnp.concatenate(
                    [jnp.broadcast_to(gtok[ts][:, None], act.shape).reshape(-1), jnp.repeat(gtok, SI)]
                )
                all_act = jnp.concatenate([act.reshape(-1), s_act.reshape(-1)])
                best = jnp.full((NUM_REGISTRY + 1,), -1, jnp.int32).at[all_seg].max(all_key)
                hit = (all_key == best[all_seg]) & (all_key >= 0)
                extras["best_key"].append(best[:NUM_REGISTRY])
                extras["best_tok"].append(
                    jnp.full((NUM_REGISTRY + 1,), -1, jnp.int32).at[all_seg].max(jnp.where(hit, all_tok, -1))[:-1]
                )
                extras["best_act"].append(
                    jnp.zeros((NUM_REGISTRY + 1,), jnp.float32).at[all_seg].max(jnp.where(hit, all_act, 0.0))[:-1]
                )
        new["hist"] = jnp.stack(hists)
        new["s_hist"] = jnp.stack(s_hists)
        src = reg["source"]
        new["source"] = state["source"].at[src].add(jnp.sum(tokm, axis=0))
        per_token = dict(code=outcome_code(o), logp=o["logp"].astype(jnp.float16))
        if with_registry:
            votes = jnp.stack(extras["votes"])  # [L, NUM_SETS, T]
            fired = votes > 0
            first = jnp.where(jnp.any(fired, axis=0), jnp.argmax(fired, axis=0), L).astype(jnp.uint8)
            per_token.update(
                first=first,
                early_votes=jnp.minimum(jnp.sum(votes[:4], axis=0), 255).astype(jnp.uint8),
                votes=jnp.minimum(jnp.sum(votes, axis=0), 255).astype(jnp.uint8),
            )
            bk = jnp.stack(extras["best_key"])  # [L, R]; each unit lives in one layer
            layer = jnp.argmax(bk, axis=0)
            per_token["reg_tok"] = jnp.take_along_axis(jnp.stack(extras["best_tok"]), layer[None], 0)[0]
            per_token["reg_act"] = jnp.take_along_axis(jnp.stack(extras["best_act"]), layer[None], 0)[0]
        return new, per_token

    return jax.jit(step, donate_argnums=0)


def init_state() -> dict:
    return dict(
        hist=jnp.zeros((L, E * I * NB, len(STATS)), jnp.float32),
        s_hist=jnp.zeros((L, SI * NB, len(STATS)), jnp.float32),
        source=jnp.zeros((len(SOURCES), len(TOKEN_STATS)), jnp.float32),
    )


@common.off_mesh
def paloma_range(start: int, count: int) -> Iterator[common.TokenBatch]:
    """Paloma validation sequences ``[start, start + count)`` of every set, in batches of ``BATCH``."""
    sets = common.data_config().validation_sets(Axis("position", S))
    for name in common.PALOMA_SETS:
        ds = sets[f"paloma/{name}"].as_sync_dataset()
        stop = min(len(ds), start + count)
        for lo in range(start, stop, BATCH):
            batch = common.examples_to_batch(name, ds.get_batch(list(range(lo, min(stop, lo + BATCH)))))
            yield common._pad_batch(batch, BATCH)


def scored_mask(batch: common.TokenBatch) -> np.ndarray:
    same_doc = np.pad(batch.segment_ids[:, 1:] == batch.segment_ids[:, :-1], ((0, 0), (0, 1)))
    return (batch.loss_weight > 0) & same_doc


def levels(hist: np.ndarray, cuts: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
    """``[N, NB, len(STATS)]`` histograms -> cumulative stats ``[N, len(LEVEL_NAMES), len(STATS)]`` and the cut bins
    ``[N, len(LEVEL_NAMES)]`` (computed from ``hist`` when ``cuts`` is None)."""
    cum = np.cumsum(hist[:, ::-1].astype(np.float64), axis=1)[:, ::-1]  # [N, NB, S]: stats of bins >= b
    if cuts is None:
        n_fire = cum[:, 1, 0]
        cuts = np.zeros((hist.shape[0], len(LEVEL_NAMES)), np.int16)
        cuts[:, 1] = 1
        for j, f in enumerate(LEVEL_FRACTIONS):
            # The highest bin whose tail holds at least f of the fires (so each level holds >= f of them).
            ok = cum[:, 1:, 0] >= f * n_fire[:, None]
            cuts[:, 2 + j] = 1 + (ok.shape[1] - 1 - np.argmax(ok[:, ::-1], axis=1))
            cuts[n_fire == 0, 2 + j] = NB - 1
    out = np.take_along_axis(cum, cuts[:, :, None].astype(np.int64), axis=1)
    return out.astype(np.float32), cuts


def load_npz(tag: str, name: str) -> dict[str, np.ndarray]:
    with fsspec.open(f"{common.OUTPUT_ROOT}/{tag}/{name}.npz", "rb") as f:
        z = np.load(f)
        return {k: z[k] for k in z.files}


def load_registry(tag: str, name: str) -> dict:
    with fsspec.open(f"{common.OUTPUT_ROOT}/{tag}/{name}.json") as f:
        return json.load(f)


def registry_tables(reg: dict) -> dict[str, np.ndarray]:
    """Device tables of a registry: strong thresholds, set bits and registry indices per routed and shared unit."""
    thr = np.full((L, E, I), np.inf, np.float32)
    bits = np.zeros((L, E, I), np.int32)
    idx = np.full((L, E, I), -1, np.int32)
    s_thr = np.full((L, SI), np.inf, np.float32)
    s_bits = np.zeros((L, SI), np.int32)
    s_idx = np.full((L, SI), -1, np.int32)
    for u in reg["units"]:
        lay, ex, un = u["layer"], u["expert"], u["unit"]
        if ex >= 0:
            thr[lay, ex, un], bits[lay, ex, un] = u["thr"], u["bits"]
            if u.get("reg", -1) >= 0:
                idx[lay, ex, un] = u["reg"]
        else:
            s_thr[lay, un], s_bits[lay, un] = u["thr"], u["bits"]
            if u.get("reg", -1) >= 0:
                s_idx[lay, un] = u["reg"]
    return dict(thr=thr, bits=bits, idx=idx, s_thr=s_thr, s_bits=s_bits, s_idx=s_idx)


@dataclasses.dataclass
class HostTokens:
    """The pass's token arrays, for examples, target types and the ablation batches."""

    tokens: list[np.ndarray] = dataclasses.field(default_factory=list)
    segs: list[np.ndarray] = dataclasses.field(default_factory=list)
    scored: list[np.ndarray] = dataclasses.field(default_factory=list)
    sources: list[int] = dataclasses.field(default_factory=list)

    def add(self, batch: common.TokenBatch) -> None:
        self.tokens.append(batch.tokens.astype(np.uint16))
        self.segs.append(batch.segment_ids.astype(np.int32))
        self.scored.append(scored_mask(batch))
        self.sources += [SOURCES.index(batch.source)] * BATCH

    def stacked(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return (
            np.concatenate(self.tokens),
            np.concatenate(self.segs),
            np.concatenate(self.scored),
            np.asarray(self.sources, np.int8),
        )


def target_types(
    tokens: np.ndarray, segs: np.ndarray, sources: np.ndarray, vocab: list[str], freq_rank: np.ndarray
) -> np.ndarray:
    """``[N, S]`` uint16 bit set of each position's target type (``TYPE_NAMES``): the attribution categories, plus
    ``ngram3`` (``p(c | a, b) >= NGRAM_P`` over these tokens, ``(a, b)`` seen ``>= NGRAM_MIN`` times) and ``bigram``
    (the same for ``p(c | b)``)."""
    feats = categories.token_features(vocab)
    tok = tokens.astype(np.int64)
    out = np.zeros(tokens.shape, np.uint16)
    for r in range(tokens.shape[0]):
        m = categories.category_masks(
            tokens[r : r + 1].astype(np.int64),
            segs[r : r + 1],
            np.ones((1, tokens.shape[1]), np.float32),
            SOURCES[sources[r]],
            feats,
            freq_rank,
        )[1:]
        for j in range(m.shape[0]):
            out[r] |= (m[j, 0] > 0).astype(np.uint16) << j
    c = np.pad(tok[:, 1:], ((0, 0), (0, 1)))
    a = np.pad(tok[:, :-1], ((0, 0), (1, 0)))
    same = np.pad(segs[:, 1:] == segs[:, :-1], ((0, 0), (0, 1))) & np.pad(segs[:, 1:] == segs[:, :-1], ((0, 0), (1, 0)))
    for bit, ctx in ((len(TYPE_NAMES) - 2, a * V + tok), (len(TYPE_NAMES) - 1, tok)):
        key_ctx = ctx[same]
        key_full = key_ctx * V + c[same]
        uc, cc = np.unique(key_ctx, return_counts=True)
        uf, cf = np.unique(key_full, return_counts=True)
        n_ctx = cc[np.searchsorted(uc, key_ctx)]
        n_full = cf[np.searchsorted(uf, key_full)]
        det = (n_ctx >= NGRAM_MIN) & (n_full >= NGRAM_P * n_ctx)
        flag = np.zeros(tokens.shape, bool)
        flag[same] = det
        out |= flag.astype(np.uint16) << bit
    return out


def coverage_tables(per_token: dict, scored: np.ndarray, types: np.ndarray, sources: np.ndarray, reg: dict) -> dict:
    """Coverage of the registry's sets on this pass's scored tokens: for each set and layer cutoff (a member firing
    strongly in layers ``<= c``), the covered tokens' outcome counts (precision) and the share of each outcome's
    tokens covered (recall), overall, by source group and by target type; and precision by vote count."""
    ok = scored.reshape(-1)
    code = per_token["code"][ok]
    out_flags = {k: ((code >> j) & 1) > 0 for j, k in enumerate(OUTCOME_BITS)}
    first = per_token["first"][:, ok]  # [NUM_SETS, N]
    ty = types.reshape(-1)[ok]
    src = np.repeat(sources, scored.shape[1])[ok]
    groups = {"all": np.ones(ok.sum(), bool), "train": src == 0, "paloma": src > 0}
    for j, name in enumerate(TYPE_NAMES):
        groups[f"type:{name}"] = ((ty >> j) & 1) > 0
    totals = {g: {k: int((m & f).sum()) for k, f in out_flags.items()} | {"n": int(m.sum())} for g, m in groups.items()}
    res = {"sets": reg["sets"], "totals": totals, "cutoff": []}
    for j in range(len(reg["sets"])):
        rows = []
        for c in range(L):
            cov = first[j] <= c
            row = {}
            for g, m in groups.items():
                cm = cov & m
                row[g] = {k: int((cm & f).sum()) for k, f in out_flags.items()} | {"n": int(cm.sum())}
            rows.append(row)
        res["cutoff"].append(rows)
    buckets = (0, 1, 2, 3, 5, 9, 17, 256)
    res["vote_buckets"] = list(buckets)
    for key in ("early_votes", "votes"):
        v = per_token[key][:, ok]
        tab = []
        for j in range(len(reg["sets"])):
            b = np.searchsorted(np.asarray(buckets), v[j], side="right") - 1
            tab.append(
                [
                    {k: int((f & (b == i)).sum()) for k, f in out_flags.items()} | {"n": int((b == i).sum())}
                    for i in range(len(buckets) - 1)
                ]
            )
        res[key] = tab
    return res


def registry_examples(per_token: dict, tok: np.ndarray, types: np.ndarray, code: np.ndarray, reg: dict) -> dict:
    """Per registry unit: its recorded strong-fire positions' target types and outcomes, and context windows."""
    flat_tok, flat_types = tok.reshape(-1), types.reshape(-1)
    seq_len = tok.shape[1]
    positions = per_token["reg_tok"]  # [batches, R]
    acts = per_token["reg_act"]
    n_reg = len([u for u in reg["units"] if u.get("reg", -1) >= 0])
    type_counts = np.zeros((n_reg, len(TYPE_NAMES)), np.int32)
    out_counts = np.zeros((n_reg, len(OUTCOME_BITS)), np.int32)
    n_pos = np.zeros(n_reg, np.int32)
    windows = np.full((n_reg, 8, 33), -1, np.int32)
    win_act = np.zeros((n_reg, 8), np.float32)
    win_code = np.zeros((n_reg, 8), np.uint8)
    for r in range(n_reg):
        pos = positions[:, r]
        a = acts[:, r]
        keep = pos >= 0
        pos, a = pos[keep], a[keep]
        n_pos[r] = len(pos)
        if len(pos) == 0:
            continue
        for j in range(len(TYPE_NAMES)):
            type_counts[r, j] = int((((flat_types[pos] >> j) & 1) > 0).sum())
        for j in range(len(OUTCOME_BITS)):
            out_counts[r, j] = int((((code[pos] >> j) & 1) > 0).sum())
        for w, p in enumerate(pos[:8]):  # random order (hash keys)
            lo = max(p - 31, (p // seq_len) * seq_len)
            seq = flat_tok[lo : p + 2]
            windows[r, w, 33 - len(seq) :] = seq
            win_act[r, w] = a[w]
            win_code[r, w] = code[p]
    return dict(
        type_counts=type_counts,
        outcome_counts=out_counts,
        n_positions=n_pos,
        windows=windows,
        window_act=win_act,
        window_code=win_code,
    )


def make_ablate():
    @jax.jit
    def ablate(model, tokens, seg, valid, unit, ctrl):
        """``unit`` / ``ctrl``: ``[layer, expert (-1: shared), unit, thr, mean]`` float arrays. The decider's strong
        fires are found in the base forward; then the decider, and separately the control unit, are mean-ablated at
        exactly those positions."""
        model = forward.replicate(model)
        base = forward.replicate(forward.trace_forward(model, tokens, seg, forward.no_ablation()))
        t = tokens.size
        layer, expert, un = (unit[j].astype(jnp.int32) for j in range(3))
        is_shared = expert < 0
        tok_acts = []
        for lay in base["layers"]:
            ex, ts = lay["expert_sorted"], lay["token_sorted"]
            col = jnp.take(lay["expert_act"], un, axis=1)
            r = jnp.zeros((t,), jnp.float32).at[ts].max(jnp.where(ex == expert, col, 0.0))
            s = jnp.take(lay["shared_act"], un, axis=1)
            tok_acts.append(jnp.where(is_shared, s, r))
        tok_act = jnp.stack(tok_acts)[layer]
        v = valid.reshape(-1)
        strong = (tok_act >= unit[3]) & (tok_act > 0) & v
        selected = jnp.stack([lay["selected"] for lay in base["layers"]])  # [L, T, K]
        lsel = (jnp.arange(L) == layer)[:, None, None]

        def edited(spec):
            sl, se, su, smean = spec[0].astype(jnp.int32), spec[1].astype(jnp.int32), spec[2].astype(jnp.int32), spec[4]
            hit_tk = lsel & (selected == se) & strong[None, :, None] & (se >= 0)  # [L, T, K]
            routed_keep = ~(hit_tk[..., None] & (jnp.arange(I) == su)[None, None, None, :])
            s_hit = (jnp.arange(L) == sl)[:, None] & strong[None, :] & (se < 0)
            shared_keep = ~(s_hit[..., None] & (jnp.arange(SI) == su)[None, None, :])
            abl = forward.no_ablation()
            abl = abl._replace(
                neuron_mean=abl.neuron_mean.at[sl, jnp.maximum(se, 0), su].set(jnp.where(se >= 0, smean, 0.0)),
                shared_mean=abl.shared_mean.at[sl, su].set(jnp.where(se < 0, smean, 0.0)),
            )
            edit = forward.UnitEdit(routed_keep=routed_keep, shared_keep=shared_keep)
            return forward.replicate(forward.trace_forward(model, tokens, seg, abl, edit=edit))

        abl_tr = edited(unit)
        ctl_tr = edited(ctrl)
        w_out = model.output_proj.astype(jnp.float32)
        labels = base["labels"].reshape(-1)
        w_c = (w_out.T[labels] - jnp.mean(w_out, axis=1)[None]).astype(jnp.float32)
        res = {}
        others = v & ~strong
        for name, tr in (("base", base), ("abl", abl_tr), ("ctl", ctl_tr)):
            o = token_outcomes(tr, v, w_c)
            sf = strong.astype(jnp.float32)
            res[f"{name}_logp"] = jnp.sum(o["logp"] * sf)
            for k in ("cr70", "cr90", "cr99", "top1"):
                res[f"{name}_{k}"] = jnp.sum(o[k] * sf)
            res[f"{name}_other_logp"] = jnp.sum(o["logp"] * others)
        res["n"] = jnp.sum(strong)
        res["n_other"] = jnp.sum(others)
        return res

    return ablate


def run_pass(args, model, mesh, ref, s_ref, reg_tables) -> tuple[dict, dict, HostTokens]:
    with set_mesh(mesh):
        dirs, s_dirs = jax.jit(lambda m: unit_directions(forward.replicate(m)))(model)
        promoted, s_promoted = top_promoted(model)
        step = make_step(reg_tables is not None)
        state = init_state()
        dev_ref, dev_s_ref = jnp.asarray(ref), jnp.asarray(s_ref)
        reg = {k: jnp.asarray(v) for k, v in (reg_tables or {}).items()}
        host = HostTokens()
        per_token: dict[str, list[np.ndarray]] = {}
        t0 = time.time()
        batches = [
            *paloma_range(args.paloma_start, args.paloma_sequences),
            *common.train_batches(BATCH, args.train_sequences, seed=args.seed),
        ]
        logger.info("%d batches loaded in %.0fs", len(batches), time.time() - t0)
        for n, batch in enumerate(batches):
            tokens, seg, _ = common.device_batch(batch, mesh)
            valid = jnp.asarray(scored_mask(batch))
            reg["source"] = jnp.asarray(SOURCES.index(batch.source), jnp.int32)
            offset = jnp.asarray(n * BATCH * S, jnp.int32)
            state, pt = step(
                state, model, dirs, s_dirs, promoted, s_promoted, dev_ref, dev_s_ref, tokens, seg, valid, offset, reg
            )
            for k, val in jax.device_get(pt).items():
                per_token.setdefault(k, []).append(np.asarray(val))
            host.add(batch)
            if n % 100 == 0:
                logger.info("batch %d/%d (%s) %.0fs", n, len(batches), batch.source, time.time() - t0)
        hist = np.asarray(state["hist"]).reshape(L * E * I, NB, len(STATS))
        s_hist = np.asarray(state["s_hist"]).reshape(L * SI, NB, len(STATS))
        source = np.asarray(state["source"])
        weights = dict(promoted=np.asarray(promoted, np.int16), s_promoted=np.asarray(s_promoted, np.int16))
    logger.info("pass done in %.0fs", time.time() - t0)
    tables = dict(hist=hist, s_hist=s_hist, source=source, **weights)
    stacked = {}
    for k, vals in per_token.items():
        if k in ("reg_tok", "reg_act"):
            stacked[k] = np.stack(vals)
        elif vals[0].ndim == 2:
            stacked[k] = np.concatenate(vals, axis=1)
        else:
            stacked[k] = np.concatenate(vals)
    return tables, stacked, host


def reduce_tables(tables: dict, cuts: np.ndarray | None, s_cuts: np.ndarray | None, ref, s_ref) -> dict:
    lv, cuts = levels(tables["hist"], cuts)
    s_lv, s_cuts = levels(tables["s_hist"], s_cuts)
    fires = tables["hist"][:, 1:, 0].sum(-1)
    return dict(
        levels=lv.reshape(L, E, I, len(LEVEL_NAMES), len(STATS)),
        cuts=cuts.reshape(L, E, I, -1),
        s_levels=s_lv.reshape(L, SI, len(LEVEL_NAMES), len(STATS)),
        s_cuts=s_cuts.reshape(L, SI, -1),
        thr=(bin_lower_edge(cuts.reshape(L, E, I, -1)) * ref[..., None]).astype(np.float32),
        s_thr=(bin_lower_edge(s_cuts.reshape(L, SI, -1)) * s_ref[..., None]).astype(np.float32),
        ref=ref,
        s_ref=s_ref,
        fires=fires.reshape(L, E, I).astype(np.float32),
        source=tables["source"],
        promoted=tables["promoted"],
        s_promoted=tables["s_promoted"],
        stats=np.asarray(STATS),
        level_names=np.asarray(LEVEL_NAMES),
        sources=np.asarray(SOURCES),
    )


def run_ablations(model, mesh, reg: dict, positions: np.ndarray, host: HostTokens) -> list[dict]:
    tok, segs, scored, _ = host.stacked()
    ablate = make_ablate()
    results = []
    reg_units = {u["reg"]: u for u in reg["units"] if u.get("reg", -1) >= 0}
    t0 = time.time()
    with set_mesh(mesh):
        for r, u in sorted(reg_units.items()):
            if not u.get("ablate"):
                continue
            pos = positions[:, r]
            seqs = list(dict.fromkeys((pos[pos >= 0] // tok.shape[1]).tolist()))[:ABLATE_SEQUENCES]
            spec = jnp.asarray([u["layer"], u["expert"], u["unit"], u["thr"], u["mean"]], jnp.float32)
            ctrl = jnp.asarray([u["layer"], u["expert"], u["ctrl"], u["thr"], u["ctrl_mean"]], jnp.float32)
            acc: dict[str, float] = {}
            for lo in range(0, len(seqs), BATCH):
                rows = seqs[lo : lo + BATCH]
                rows = rows + [rows[0]] * (BATCH - len(rows))
                valid = scored[rows].copy()
                valid[len(seqs[lo : lo + BATCH]) :] = False
                out = jax.device_get(
                    ablate(
                        model,
                        jnp.asarray(tok[rows].astype(np.int32)),
                        jnp.asarray(segs[rows]),
                        jnp.asarray(valid),
                        spec,
                        ctrl,
                    )
                )
                for k, val in out.items():
                    acc[k] = acc.get(k, 0.0) + float(val)
            results.append({"reg": r, "sequences": len(seqs), **acc})
            logger.info("ablated %s: n=%d %.0fs", u, acc.get("n", 0), time.time() - t0)
    return results


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True)
    parser.add_argument("--phase", choices=("discover", "replicate"), required=True)
    parser.add_argument("--train-sequences", type=int, default=4096)
    parser.add_argument("--paloma-start", type=int, default=0)
    parser.add_argument("--paloma-sequences", type=int, default=128, help="per Paloma set")
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--discover-name", default="decider_discover", help="replicate: the discover output")
    parser.add_argument("--registry", default="decider_registry", help="replicate: the registry json")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    cfg = common.model_config(args.tag)
    forward.check_config(cfg)
    mesh = common.one_gpu_mesh()
    params = common.load_params(cfg, mesh, common.CHECKPOINTS[args.tag])
    ref, s_ref = stats_ref("tn-d512-ckpt1-rno" if args.tag.startswith("tn-") else args.tag)
    with set_mesh(mesh):
        model = common.compute_model(params)
    reg = reg_tables = cuts = s_cuts = None
    if args.phase == "replicate":
        disc = load_npz(args.tag, args.discover_name)
        cuts = disc["cuts"].reshape(L * E * I, -1)
        s_cuts = disc["s_cuts"].reshape(L * SI, -1)
        ref, s_ref = disc["ref"], disc["s_ref"]
        reg = load_registry(args.tag, args.registry)
        reg_tables = registry_tables(reg)
    tables, per_token, host = run_pass(args, model, mesh, ref, s_ref, reg_tables)
    out = reduce_tables(tables, cuts, s_cuts, ref, s_ref)
    if reg is None:
        common.save_npz(args.tag, args.out, **out)
        return
    tok, segs, scored, sources = host.stacked()
    t0 = time.time()
    with fsspec.open(f"{common.OUTPUT_ROOT}/tn-d512-ckpt1-rno/stats.npz", "rb") as f:
        counts = np.load(f)["token_counts"]
    freq_rank = np.empty(V, np.int64)
    freq_rank[np.argsort(-counts)] = np.arange(V)
    types = target_types(tok, segs, sources, common.vocab_strings(), freq_rank)
    logger.info("target types in %.0fs", time.time() - t0)
    cov = coverage_tables(per_token, scored, types, sources, reg)
    code = per_token["code"]
    ex = registry_examples(per_token, tok, types, code, reg)
    rng = np.random.default_rng(0)
    flat_ok = np.flatnonzero(scored.reshape(-1))
    sub = np.sort(rng.choice(flat_ok, size=min(SUBSAMPLE_TOKENS, len(flat_ok)), replace=False))
    common.save_npz(
        args.tag,
        args.out,
        **out,
        **{f"ex_{k}": v for k, v in ex.items()},
        sub_index=sub,
        sub_code=code[sub],
        sub_logp=per_token["logp"][sub],
        sub_first=per_token["first"][:, sub],
        sub_early_votes=per_token["early_votes"][:, sub],
        sub_votes=per_token["votes"][:, sub],
        sub_types=types.reshape(-1)[sub],
        sub_source=np.repeat(sources, S)[sub],
        reg_tok=per_token["reg_tok"],
    )
    common.save_json(args.tag, f"{args.out}_coverage", cov)
    abl = run_ablations(model, mesh, reg, per_token["reg_tok"], host)
    common.save_json(args.tag, f"{args.out}_ablation", abl)


if __name__ == "__main__":
    main()
