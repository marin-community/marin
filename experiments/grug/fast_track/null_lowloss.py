# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Routed-expert headroom and low-loss tokens of one Belay Ckpt 1 checkpoint (eval only, one GPU).

On the held-out sequences of a token dump, scores the checkpoint at its native top-k and with fewer / more routed
experts: globally (k-2, k-4, k-6, k+2, top-2, top-1), in one layer at a time (k-2, k-4, k-6, top-2, top-1), and with
one layer's routed MoE output dropped for a token subset (the low-loss tokens, a matched random subset, or all
tokens). Captures every layer's router input and residual stream at a fixed position subsample and fits cross-fitted
ridge probes for the value of 2 experts and for the low-loss class. Writes the per-token losses to ``--output``.

    python -m experiments.grug.fast_track.null_lowloss --size d1024 --run lc1-l171-d1024 --step 32323 \\
        --output s3://.../lowloss_l171.npz
"""

import argparse
import dataclasses
import io
import itertools
import json
import logging
import time
from typing import ClassVar

import fsspec
import jax
import jax.numpy as jnp
import jmp
import numpy as np
from haliax.partitioning import set_mesh
from levanter.checkpoint import load_checkpoint
from levanter.grug.sharding import compact_grug_mesh

from experiments.grug.fast_track import belay_ckpt1, launch
from experiments.grug.fast_track import model as M
from experiments.grug.fast_track.train import _apply_qb_betas, _to_dropless_local, restore_template_from

logger = logging.getLogger(__name__)

ROOT = "s3://marin-us-east-02a/marin/grug"
VERSION = "2026.09.25"
MP = jmp.get_policy("params=float32,compute=bfloat16,output=bfloat16")
LOW_LOSS = 0.05
PROBE_STRIDE = 8  # probe features at every PROBE_STRIDE-th position (offset PROBE_OFFSET)
PROBE_OFFSET = 3
RIDGE_ALPHAS = (1.0, 10.0, 100.0, 1000.0, 10000.0)


def model_config(size: str, overrides: dict[str, str]) -> M.GrugModelConfig:
    """The Ckpt 1 model of ``size`` as ``belay_ckpt1`` builds it, plus ``overrides``; evaluated with the plain
    autodiff AttnRes layer (same forward)."""
    settings = {}
    for flag, value in itertools.pairwise(belay_ckpt1.recipe_args(size)):
        if flag == "--model-set":
            key, _, val = value.partition("=")
            settings[key] = val
    settings.update(overrides)
    cfg = dataclasses.replace(launch._h100_ladder_model(launch._h100_ladder_rung(size)), vocab_size=launch.V16384_VOCAB)
    cfg = launch._apply_settings(launch._kma_model(cfg), settings)
    return dataclasses.replace(cfg, attn_res_layer_backward=M.AttnResLayerBackward.SAVE, attn_res_remat_attention=False)


def load_model(cfg: M.GrugModelConfig, path: str, mesh) -> M.Transformer:
    with set_mesh(mesh):
        params = jax.jit(lambda key: MP.cast_to_param(M.Transformer.init(cfg, key=key)))(jax.random.PRNGKey(0))
        loaded = load_checkpoint(
            {"params": restore_template_from(params)}, path, axis_mapping=None, mesh=mesh, allow_partial=False
        )["params"]
        betas = load_checkpoint(
            {"pending_qb_betas": jax.ShapeDtypeStruct((cfg.num_layers, cfg.router_columns), jnp.float32)},
            path,
            axis_mapping=None,
            mesh=mesh,
            allow_partial=True,
        )["pending_qb_betas"]
        model = _apply_qb_betas(loaded, betas)
        return _to_dropless_local(MP.cast_to_compute(model), implementation="sonic")


@dataclasses.dataclass(frozen=True)
class Variant:
    """One scored configuration: per-layer top-k (None: native) and which layers drop their routed MoE output for the
    masked tokens."""

    name: str
    layer_k: tuple[tuple[int, int], ...] = ()
    drop_layers: tuple[int, ...] = ()
    capture: bool = False


class _Hooks:
    """Per-trace state of the patched MoE / AttnRes calls (set while one variant's forward is traced)."""

    variant: Variant | None = None
    mask: jax.Array | None = None
    moe_calls = 0
    router_inputs: ClassVar[list[jax.Array]] = []
    residuals: ClassVar[list[jax.Array]] = []


_ORIG_MOE = M.MoEMLP.__call__
_ORIG_LAYER = M._attn_res_layer_passthrough


def _probe_rows(x: jax.Array) -> jax.Array:
    return x[:, PROBE_OFFSET::PROBE_STRIDE]


def _moe_call(self, x, projected=None):
    layer = _Hooks.moe_calls
    _Hooks.moe_calls += 1
    v = _Hooks.variant
    module = self
    for target, k in v.layer_k:
        if target == layer:
            module = dataclasses.replace(self, cfg=dataclasses.replace(self.cfg, num_experts_per_token=k))
    if v.capture:
        _Hooks.router_inputs.append(_probe_rows(x))
    routed, stats = _ORIG_MOE(module, x, projected)
    if layer in v.drop_layers:
        routed = routed * (1.0 - _Hooks.mask)[..., None].astype(routed.dtype)
    return routed, stats


def _layer_call(diff_args, mask, token_ids, use_long, layer_index, eps):
    partial, blocks, block_logits, stats = _ORIG_LAYER(diff_args, mask, token_ids, use_long, layer_index, eps)
    if _Hooks.variant is not None and _Hooks.variant.capture:
        _Hooks.residuals.append(_probe_rows(M._residual_stream(blocks, partial)))
    return partial, blocks, block_logits, stats


def score(model: M.Transformer, variant: Variant, tokens, segments, token_mask, batch: int):
    """Per-token losses ``[N, S]`` of ``variant`` (plus the captured probe features when ``variant.capture``)."""

    def fwd(m, tok, seg, msk):
        _Hooks.variant, _Hooks.mask, _Hooks.moe_calls = variant, msk, 0
        _Hooks.router_inputs, _Hooks.residuals = [], []
        attn_mask = M.AttentionMask.causal().with_segment_ids(seg)
        loss = m.next_token_loss(tok, jnp.ones(tok.shape, jnp.float32), mask=attn_mask, reduction="none")
        feats = (tuple(_Hooks.router_inputs), tuple(_Hooks.residuals)) if variant.capture else ((), ())
        _Hooks.variant, _Hooks.mask = None, None
        return loss, feats

    run = jax.jit(fwd)  # a fresh jit per variant: the hooks are trace-time state
    losses, router, resid = [], [], []
    for start in range(0, tokens.shape[0], batch):
        sl = slice(start, start + batch)
        loss, (ri, rs) = run(model, jnp.asarray(tokens[sl]), jnp.asarray(segments[sl]), jnp.asarray(token_mask[sl]))
        losses.append(np.asarray(loss, np.float32))
        if variant.capture:
            router.append(np.stack([np.asarray(x, np.float16) for x in ri], 1))  # [B, L, S', D]
            resid.append(np.stack([np.asarray(x, np.float16) for x in rs], 1))
    out = np.concatenate(losses)
    if variant.capture:
        return out, np.concatenate(router), np.concatenate(resid)
    return out, None, None


def ridge_scores(x: np.ndarray, y: np.ndarray, fold: np.ndarray) -> np.ndarray:
    """Out-of-fold ridge predictions (two folds by sequence; alpha on an inner split of the training fold)."""
    pred = np.zeros(y.shape, np.float32)
    xd = jnp.asarray(x, jnp.float32)
    xd = (xd - xd.mean(0)) / (xd.std(0) + 1e-3)
    xd = jnp.concatenate([xd, jnp.ones((xd.shape[0], 1), jnp.float32)], axis=1)
    yd = jnp.asarray(y, jnp.float32)

    def solve(rows, alpha):
        a = xd[rows]
        return jnp.linalg.solve(a.T @ a + alpha * jnp.eye(a.shape[1]), a.T @ yd[rows])

    for k in (0, 1):
        tr, te = np.flatnonzero(fold != k), np.flatnonzero(fold == k)
        inner = tr % 5 == 0
        errs = [float(jnp.mean((xd[tr[inner]] @ solve(tr[~inner], a) - yd[tr[inner]]) ** 2)) for a in RIDGE_ALPHAS]
        w = solve(tr, RIDGE_ALPHAS[int(np.argmin(errs))])
        pred[te] = np.asarray(xd[te] @ w)
    return pred


def auc(score_: np.ndarray, label: np.ndarray) -> float:
    order = np.argsort(score_)
    ranks = np.empty(order.size)
    ranks[order] = np.arange(1, order.size + 1)
    pos = label.astype(bool)
    n1, n0 = pos.sum(), (~pos).sum()
    return float((ranks[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--version", default=VERSION)
    parser.add_argument("--set", action="append", default=[], help="model config override KEY=VALUE")
    parser.add_argument("--dump", default=f"{ROOT}/nx-ctl/{VERSION}/token_dump.npz")
    parser.add_argument("--sequences", type=int, default=512)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--phases", default="global,layers,drop,probes")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    phases = set(args.phases.split(","))
    with fsspec.open(args.dump, "rb") as f:
        dump = np.load(io.BytesIO(f.read()))
        tokens, segments = dump["tokens"][: args.sequences], dump["segment_ids"][: args.sequences]
    cfg = model_config(args.size, dict(s.split("=", 1) for s in args.set))
    k, layers = cfg.num_experts_per_token, cfg.num_layers
    mesh = compact_grug_mesh(expert_axis_size=1, replica_axis_size=1)
    M.MoEMLP.__call__ = _moe_call
    M._attn_res_layer_passthrough = _layer_call
    out: dict[str, np.ndarray] = {"tokens": tokens, "segment_ids": segments}
    meta: dict = {"run": args.run, "size": args.size, "native_k": k, "layers": layers}
    no_mask = np.zeros(tokens.shape, np.float32)
    with set_mesh(mesh):
        t0 = time.time()
        model = load_model(cfg, f"{ROOT}/{args.run}/{args.version}/checkpoints/step-{args.step}", mesh)
        logger.info("loaded %s in %.0f s", args.run, time.time() - t0)

        def run(variant: Variant, mask=no_mask):
            t = time.time()
            loss, router, resid = score(model, variant, tokens, segments, mask, args.batch)
            logger.info("%s: mean loss %.5f (%.0f s)", variant.name, float(loss[:, :-1].mean()), time.time() - t)
            return loss, router, resid

        native, router, resid = run(Variant("native", capture="probes" in phases))
        out["native"] = native
        if "global" in phases:
            for kk in sorted({k - 2, k - 4, k - 6, k + 2, 2, 1}):
                if kk == k or kk < 1:
                    continue
                out[f"k{kk}"] = run(Variant(f"k{kk}", layer_k=tuple((i, kk) for i in range(layers))))[0]
        if "layers" in phases:
            for i in range(layers):
                for kk in (k - 2, k - 4, k - 6, 2, 1):
                    loss = run(Variant(f"L{i}_k{kk}", layer_k=((i, kk),)))[0]
                    out[f"L{i}_k{kk}_delta"] = (loss - native).astype(np.float16)
        if "drop" in phases:
            low = (native < LOW_LOSS).astype(np.float32)
            rng = np.random.default_rng(0)
            rand = (rng.random(native.shape) < low.mean()).astype(np.float32)
            meta["low_frac"], meta["rand_frac"] = float(low.mean()), float(rand.mean())
            out["low_mask"], out["rand_mask"] = low.astype(np.int8), rand.astype(np.int8)
            for name, mask in (("low", low), ("rand", rand)):
                loss = run(Variant(f"drop_all_{name}", drop_layers=tuple(range(layers))), mask)[0]
                out[f"drop_all_{name}_delta"] = (loss - native).astype(np.float32)
                for i in range(layers):
                    loss = run(Variant(f"drop_L{i}_{name}", drop_layers=(i,)), mask)[0]
                    out[f"drop_L{i}_{name}_delta"] = (loss - native).astype(np.float16)
            for i in range(layers):
                loss = run(Variant(f"drop_L{i}_every", drop_layers=(i,)), np.ones(tokens.shape, np.float32))[0]
                out[f"drop_L{i}_every_delta"] = (loss - native).astype(np.float16)
    if "probes" in phases:
        assert router is not None and resid is not None
        pos = np.arange(tokens.shape[1])[PROBE_OFFSET::PROBE_STRIDE]
        valid = np.ones(tokens.shape, bool)
        valid[:, -1] = False
        valid &= np.concatenate([segments[:, 1:] == segments[:, :-1], np.zeros((tokens.shape[0], 1), bool)], 1)
        vsub = valid[:, pos]
        fold = np.broadcast_to((np.arange(tokens.shape[0]) % 2)[:, None], vsub.shape)[vsub]
        targets = {"low_loss": (native[:, pos] < LOW_LOSS).astype(np.float32)[vsub]}
        if f"k{k - 2}" in out:
            targets["gain2"] = np.clip(out[f"k{k - 2}"][:, pos] - native[:, pos], -5, 5)[vsub]
        meta["probes"] = {}
        out["probe_positions"] = pos
        for kind, feats in (("router", router), ("resid", resid)):
            for i in range(feats.shape[1]):
                x = feats[:, i][vsub]
                for tname, y in targets.items():
                    pred = ridge_scores(x, y, fold)
                    full = np.zeros(vsub.shape, np.float16)
                    full[vsub] = pred
                    out[f"probe_{kind}_L{i}_{tname}"] = full
                    if tname == "low_loss":
                        res = {"auc": auc(pred, y)}
                    else:
                        res = {"r2": float(1 - np.mean((pred - y) ** 2) / np.var(y))}
                    meta["probes"][f"{kind}_L{i}_{tname}"] = res
                    logger.info("probe %s L%d %s %s", kind, i, tname, res)
    out["meta"] = np.frombuffer(json.dumps(meta).encode(), np.uint8)
    buffer = io.BytesIO()
    np.savez(buffer, **out)
    with fsspec.open(args.output, "wb") as f:
        f.write(buffer.getvalue())
    logger.info("wrote %s", args.output)


if __name__ == "__main__":
    main()
