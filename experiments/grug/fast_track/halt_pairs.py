# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Halting-gate case study at the query positions of the interpretability study's ``critics.py`` (" crit" -> "ics",
" crit" -> other, " wasn" -> "'t" / "\u2019t" / other, and the contrast pairs " produ" -> "cer" and " Ex" -> "ception").

For each model (``RUNS``: the matched control without halting, the reward-trained dampen run, the frozen-trunk
probe), at each query:

- the final next-token distribution: log p of the pair's target and of the label, the top-5;
- the decision after L3: the dampening logit ``z + b`` and the keep ``G``;
- the exit head (final norm and lm_head on its own AttnRes readout of L0-L3): log p_exit of the target, the top-5, the
  max probability, and the reward label ``y`` = exit top-1 is the label and its probability > ``halt_confidence``;
- depth traces through ``LayerSkip`` (every token drops the listed layers' attention and MLP outputs; with strict
  first-half K/V and no branch SConv, skipping L4-L5 changes only each token's own prediction): the truncated depth
  (layers ``j..5`` skipped) and leave-one-layer-out;
- the routed votes per voting layer and each query's top voting units (``Transformer.halt_vote_dump``).

The dampen model is first checked against its own halt dump (G and the next-token loss on the dump's sequences).
Writes ``<OUTPUT>/halt_pairs.npz``.
"""

import argparse
import dataclasses
import io
import itertools
import logging
import time

import fsspec
import jax
import jax.numpy as jnp
import jmp
import numpy as np
from haliax.partitioning import set_mesh
from jax.sharding import PartitionSpec as P
from jax.sharding import reshard
from levanter.checkpoint import load_checkpoint
from levanter.grug.attention import AttentionMask
from levanter.grug.sharding import compact_grug_mesh
from levanter.tokenizers import load_tokenizer

from experiments.grug.fast_track import belay_ckpt1, launch
from experiments.grug.fast_track.model import (
    _HALT_EXITS,
    _HALT_LOGITS,
    GrugModelConfig,
    LayerSkip,
    Transformer,
    _halt_keep,
)
from experiments.grug.fast_track.train import (
    _apply_qb_betas,
    _to_dropless_local,
    dropless_moe_implementation,
    restore_template_from,
)

logger = logging.getLogger(__name__)

ROOT = "s3://marin-us-east-02a/marin/grug"
INTERP_ROOT = "s3://marin-us-east-02a/tmp/ttl=30d/interp-belay-d512/tn-d512-ckpt1-rno"
NOCONV = {"sconv_sites": ""}
KV2Q = {
    "attn_res_kv_layers": "4,5",
    "attn_res_kv_source_layers": "4",
    "attn_res_kv_strict": "true",
    "attn_res_kv_strict_query": "true",
}
HALT = {"halt_points": "3", "halt_mode": "dampen", "halt_write": "unit", "halt_reward_weight": "0.1"}


@dataclasses.dataclass(frozen=True)
class Run:
    name: str
    checkpoint: str
    settings: dict
    dump: str | None  # the run's own halt dump to check the forward against


RUNS = (
    Run("ctl", f"{ROOT}/lc1-halt-kv2q-noconv-s0/2026.09.25/checkpoints/step-2760", {**NOCONV, **KV2Q}, None),
    Run(
        "dampen",
        f"{ROOT}/lc1-halt-ev-v3-dampen-unit-b0.1-fin/2026.09.25/checkpoints/step-2760",
        {**NOCONV, **KV2Q, **HALT},
        f"{ROOT}/lc1-halt-ev-v3-dampen-unit-b0.1-fin/2026.09.25/halt_dump/step-2760.npz",
    ),
    Run(
        "probe", f"{ROOT}/lc1-halt-v3-probe-b0.1-s0v3/2026.09.25/checkpoints/step-600", {**NOCONV, **KV2Q, **HALT}, None
    ),
)
NUM_LAYERS = 6
# Layer sets every token skips: truncated depth (layers j.. dropped) and leave-one-layer-out.
SKIPS = (
    *(tuple(range(j, NUM_LAYERS)) for j in range(NUM_LAYERS)),
    *((j,) for j in range(NUM_LAYERS - 1)),
)
BATCH = 8
Q = 64
TOP = 5
VOTE_TOP_UNITS = 16
VOTE_CHUNK = 64
MP = jmp.get_policy("params=float32,compute=bfloat16,output=bfloat16")
PAIR_TARGETS = ("ics", "cer", "ception", "'t", "\u2019t")
# Per critics.py category: the index of its pair target in PAIR_TARGETS.
CAT_TARGET = (0, 0, 0, 1, 2, 3, 4, 3)


def model_config(settings: dict) -> GrugModelConfig:
    """The Ckpt 1 d512 model (as ``build_h100_ladder_run`` builds it from ``belay_ckpt1.recipe_args``) with the run's
    extra ``--model-set`` settings."""
    merged = {}
    for flag, value in itertools.pairwise(belay_ckpt1.recipe_args("d512")):
        if flag == "--model-set":
            key, _, val = value.partition("=")
            merged[key] = val
    merged.update(settings)
    rung = launch._h100_ladder_rung("d512")
    model = dataclasses.replace(launch._h100_ladder_model(rung), vocab_size=launch.V16384_VOCAB)
    return launch._apply_settings(launch._kma_model(model), merged)


def load_model(cfg: GrugModelConfig, mesh, path: str) -> Transformer:
    """The checkpoint's compute-dtype model with its pending QB betas applied, on the dropless local MoE backend of
    the run's evals."""
    with set_mesh(mesh):
        template = jax.jit(lambda key: MP.cast_to_param(Transformer.init(cfg, key=key)))(jax.random.PRNGKey(0))
        template = restore_template_from(template)
        betas = jax.ShapeDtypeStruct((cfg.num_layers, cfg.qb_num_experts), jnp.float32)
        loaded = load_checkpoint(
            {"params": template, "pending_qb_betas": betas}, path, axis_mapping=None, mesh=mesh, allow_partial=False
        )
        model = MP.cast_to_compute(_apply_qb_betas(loaded["params"], loaded["pending_qb_betas"]))
        return _to_dropless_local(model, implementation=dropless_moe_implementation(cfg))


def _rep(x: jax.Array) -> jax.Array:
    return reshard(x, P(*([None] * x.ndim)))


def _head(model: Transformer, h: jax.Array) -> jax.Array:
    """Capped log-probs ``[Q, V]`` of normed hidden states ``[Q, D]``."""
    logits = jnp.einsum("qd,dv->qv", h.astype(jnp.float32), _rep(model.output_proj).astype(jnp.float32))
    cap = model.config.logit_soft_cap
    logits = cap * jnp.tanh(logits / cap)
    return jax.nn.log_softmax(logits, axis=-1)


def _dist(logp: jax.Array, y: jax.Array, label: jax.Array) -> dict[str, jax.Array]:
    top = jax.lax.top_k(logp, TOP)
    return dict(
        logp_y=jnp.take_along_axis(logp, y[:, None], axis=1)[:, 0],
        logp_label=jnp.take_along_axis(logp, label[:, None], axis=1)[:, 0],
        top_ids=top[1],
        top_logp=top[0],
        entropy=-jnp.sum(jnp.exp(logp) * logp, axis=-1),
    )


@jax.jit
def analyse(model: Transformer, tokens, seg, qflat, qy):
    """Everything per query of one batch (see the module docstring)."""
    cfg = model.config
    mask = AttentionMask.causal().with_segment_ids(seg)
    labels = _rep(jnp.pad(tokens[:, 1:], ((0, 0), (0, 1)))).reshape(-1)[qflat]
    hidden, stats = model(tokens, mask=mask)
    d = hidden.shape[-1]
    out = {f"final/{k}": v for k, v in _dist(_head(model, _rep(hidden).reshape(-1, d)[qflat]), qy, labels).items()}
    for skip in SKIPS:
        keep = jnp.zeros(tokens.shape, jnp.bool_)
        h_s, _ = model(tokens, mask=mask, layer_skip=LayerSkip(skip, keep))
        lp = _head(model, _rep(h_s).reshape(-1, d)[qflat])
        name = "-".join(str(j) for j in skip)
        out[f"skip/{name}/logp_y"] = jnp.take_along_axis(lp, qy[:, None], axis=1)[:, 0]
        out[f"skip/{name}/logp_label"] = jnp.take_along_axis(lp, labels[:, None], axis=1)[:, 0]
        out[f"skip/{name}/top1"] = jnp.argmax(lp, axis=-1)
    if cfg.halt_points:
        logits = _rep(stats[_HALT_LOGITS])  # [P, B, S]
        g, _ = _halt_keep(logits, None)
        out["halt/logit"] = logits.reshape(logits.shape[0], -1)[:, qflat]
        out["halt/g"] = g.reshape(g.shape[0], -1)[:, qflat]
        exit_h = model.final_norm(_rep(stats[_HALT_EXITS][0]))
        lp_exit = _head(model, exit_h.reshape(-1, d)[qflat])
        out.update({f"exit/{k}": v for k, v in _dist(lp_exit, qy, labels).items()})
        maxp = jnp.exp(jnp.max(lp_exit, axis=-1))
        correct = jnp.argmax(lp_exit, axis=-1) == labels
        out["exit/y"] = correct & (maxp > cfg.halt_confidence)
        out["exit/maxp"] = maxp
        votes = model.halt_vote_dump(tokens, mask, VOTE_TOP_UNITS, VOTE_CHUNK)
        out["votes/routed"] = votes["routed_vote"][:, qflat]
        out["votes/top_ids"] = votes["top_ids"][:, qflat]
        out["votes/top_vals"] = votes["top_vals"][:, qflat]
        out["votes/gates"] = votes["gates"][:, qflat]
    return out


@jax.jit
def dump_check(model: Transformer, tokens, seg):
    """Per-token G after the first decision point and the next-token loss, as the halt dump writes them."""
    mask = AttentionMask.causal().with_segment_ids(seg)
    weight = jnp.ones(tokens.shape, jnp.float32)
    loss = model.next_token_loss(tokens, weight, mask=mask, reduction="none")
    _, stats = model(tokens, mask=mask)
    g, _ = _halt_keep(stats[_HALT_LOGITS], None)
    return loss, g[0]


def batches(tokens: np.ndarray, seg: np.ndarray, queries: np.ndarray):
    """``(rows, row tokens, row segments, query ids, flat positions)`` per batch of ``BATCH`` rows."""
    rows_of = queries[:, 0]
    for start in range(0, len(tokens), BATCH):
        rows = np.arange(start, min(start + BATCH, len(tokens)))
        tok = np.zeros((BATCH, tokens.shape[1]), np.int32)
        sg = np.zeros((BATCH, tokens.shape[1]), np.int32)
        tok[: len(rows)] = tokens[rows]
        sg[: len(rows)] = seg[rows]
        qids = np.flatnonzero((rows_of >= start) & (rows_of < start + BATCH))
        for c in range(0, max(len(qids), 1), Q):
            part = qids[c : c + Q]
            flat = np.zeros(Q, np.int32)
            flat[: len(part)] = (rows_of[part] - start) * tokens.shape[1] + queries[part, 1]
            yield tok, sg, part, flat


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", default=f"{INTERP_ROOT}/critics_rows.npz")
    parser.add_argument("--out", default=f"{INTERP_ROOT}/halt_pairs.npz")
    parser.add_argument("--runs", default=",".join(r.name for r in RUNS))
    args = parser.parse_args()
    t0 = time.time()
    with fsspec.open(args.rows, "rb") as f:
        z = np.load(io.BytesIO(f.read()))
        tokens, seg, queries = z["tokens"], z["segment_ids"], z["queries"]
    tok = load_tokenizer(launch.V16384_TOKENIZER)
    strings = [tok.decode([i]) for i in range(launch.V16384_VOCAB)]
    targets = np.asarray([strings.index(s) for s in PAIR_TARGETS], np.int32)
    qy_all = targets[np.asarray(CAT_TARGET)[queries[:, 2]]]
    logger.info("%d rows, %d queries, targets %s (%.0fs)", len(tokens), len(queries), targets, time.time() - t0)
    mesh = compact_grug_mesh(expert_axis_size=1, replica_axis_size=1)
    results: dict[str, np.ndarray] = {"queries": queries, "targets": targets}
    for run in RUNS:
        if run.name not in args.runs.split(","):
            continue
        cfg = model_config(run.settings)
        model = load_model(cfg, mesh, run.checkpoint)
        logger.info("loaded %s (%.0fs)", run.name, time.time() - t0)
        if run.dump is not None:
            with fsspec.open(run.dump, "rb") as f:
                d = np.load(io.BytesIO(f.read()))
            errs_g, errs_l = [], []
            for s in range(0, d["tokens"].shape[0], BATCH):
                with set_mesh(mesh):
                    loss, g = dump_check(
                        model, jnp.asarray(d["tokens"][s : s + BATCH]), jnp.asarray(d["segment_ids"][s : s + BATCH])
                    )
                    loss, g = jax.device_get((loss, g))
                errs_g.append(np.abs(np.asarray(g) - d["halt_g_P3"][s : s + BATCH].astype(np.float32)).mean())
                errs_l.append(
                    np.abs(np.asarray(loss) - d["next_token_loss"][s : s + BATCH].astype(np.float32))[:, :-1].mean()
                )
            results[f"{run.name}/dump_err_g"] = np.asarray(errs_g)
            results[f"{run.name}/dump_err_loss"] = np.asarray(errs_l)
            logger.info("%s dump check: mean |dG| %.4g, mean |dloss| %.4g", run.name, np.mean(errs_g), np.mean(errs_l))
        acc: dict[str, list] = {}
        for tok_b, seg_b, part, flat in batches(tokens, seg, queries):
            if len(part) == 0:
                continue
            qy = np.zeros(Q, np.int32)
            qy[: len(part)] = qy_all[part]
            with set_mesh(mesh):
                out = jax.device_get(
                    analyse(model, jnp.asarray(tok_b), jnp.asarray(seg_b), jnp.asarray(flat), jnp.asarray(qy))
                )
            for k, v in out.items():
                # halt/ and votes/ arrays lead with the decision-point or voting-layer axis.
                lead = k.startswith(("halt/", "votes/"))
                v = np.asarray(v)[:, : len(part)] if lead else np.asarray(v)[: len(part)]
                key = f"{run.name}/{k}"
                acc.setdefault(key, []).append((lead, v))
            acc.setdefault(f"{run.name}/query", []).append((False, part))
        for key, parts in acc.items():
            results[key] = np.concatenate([v for _, v in parts], axis=1 if parts[0][0] else 0)
        if cfg.halt_points:
            results[f"{run.name}/halt_bias"] = np.asarray(
                [float(lay.halt_bias) for lay in model.layers() if lay.halt_bias is not None]
            )
        logger.info("%s done (%.0fs)", run.name, time.time() - t0)
        del model
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **results)
    with fsspec.open(args.out, "wb") as f:
        f.write(buffer.getvalue())
    logger.info("wrote %s (%.1f MB, %.0fs)", args.out, buffer.tell() / 1e6, time.time() - t0)


if __name__ == "__main__":
    main()
