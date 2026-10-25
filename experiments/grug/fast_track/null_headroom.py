# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Dynamic-compute headroom of the d512 Belay Ckpt 1 runs (one H100, no training).

Scores the held-out sequences of a token dump under several routed-expert counts per checkpoint (dropless sonic
backend), captures MoE router inputs of the top-8 model, and fits cross-fitted probes (ridge, small MLP) that predict
the per-token value of extra experts from them. Writes ``[N, S]`` per-token losses and the probes' out-of-fold
predictions to ``--output``.

    python -m experiments.grug.fast_track.null_headroom --output s3://.../headroom.npz
"""

import argparse
import dataclasses
import io
import itertools
import logging

import equinox as eqx
import fsspec
import jax
import jax.numpy as jnp
import jmp
import numpy as np
import optax
from haliax.partitioning import set_mesh
from levanter.checkpoint import load_checkpoint
from levanter.grug.sharding import compact_grug_mesh

from experiments.grug.fast_track import belay_ckpt1, launch
from experiments.grug.fast_track import model as M
from experiments.grug.fast_track.train import _apply_qb_betas, _to_dropless_local, restore_template_from

logger = logging.getLogger(__name__)

ROOT = "s3://marin-us-east-02a/marin/grug"
VERSION = "2026.09.25"
STEP = 8821
MP = jmp.get_policy("params=float32,compute=bfloat16,output=bfloat16")
# (name, run, trained top-k, evaluated top-k, routing_renorm_sum)
EVALS = (
    ("top8_k8", "nx-top8", 8, 8, 2.5),
    ("top8_k6", "nx-top8", 8, 6, 2.5),
    ("top8_k6_keepw", "nx-top8", 8, 6, 2.5 * 6 / 8),
    ("top8_k10", "nx-top8", 8, 10, 2.5),
    ("ctl_k10", "nx-ctl", 10, 10, 2.5),
    ("ctl_k8", "nx-ctl", 10, 8, 2.5),
    ("ctl_k8_keepw", "nx-ctl", 10, 8, 2.5 * 8 / 10),
    ("ctl_k6", "nx-ctl", 10, 6, 2.5),
)
FEATURE_LAYERS = (0, 2)
BATCH = 16


def model_config(top_k: int, renorm_sum: float) -> M.GrugModelConfig:
    """The Ckpt 1 d512 model as ``belay_ckpt1`` builds it, at ``top_k`` routed experts."""
    settings = {}
    for flag, value in itertools.pairwise(belay_ckpt1.recipe_args("d512")):
        if flag == "--model-set":
            key, _, val = value.partition("=")
            settings[key] = val
    rung = launch._resolved_rung("d512", None, None)
    cfg = dataclasses.replace(launch._h100_ladder_model(rung), vocab_size=launch.V16384_VOCAB)
    cfg = launch._apply_settings(launch._kma_model(cfg), settings)
    return dataclasses.replace(cfg, num_experts_per_token=top_k, routing_renorm_sum=renorm_sum)


def load_state(cfg: M.GrugModelConfig, path: str, mesh) -> M.Transformer:
    """The checkpoint's params with its pending QB betas applied (the weights its token dump scored)."""
    with set_mesh(mesh):
        params = jax.jit(lambda key: MP.cast_to_param(M.Transformer.init(cfg, key=key)))(jax.random.PRNGKey(0))
        template = {
            "params": restore_template_from(params),
            "pending_qb_betas": jax.ShapeDtypeStruct((cfg.num_layers, cfg.router_columns), jnp.float32),
        }
        loaded = load_checkpoint(template, path, axis_mapping=None, mesh=mesh, allow_partial=True)
        return _apply_qb_betas(loaded["params"], loaded["pending_qb_betas"])


def with_routing(model: M.Transformer, top_k: int, renorm_sum: float, implementation: str) -> M.Transformer:
    """``model``'s weights under a config with ``top_k`` experts (the router does not depend on K; the config is static
    tree structure, so the leaves are re-attached to the new config's tree)."""
    cfg = dataclasses.replace(model.config, num_experts_per_token=top_k, routing_renorm_sum=renorm_sum)
    abstract = eqx.filter_eval_shape(M.Transformer.init, cfg, key=jax.random.PRNGKey(0))
    abstract = _to_dropless_local(abstract, implementation=implementation)
    leaves = jax.tree.leaves(model)
    treedef = jax.tree.structure(abstract)
    assert treedef.num_leaves == len(leaves), (treedef.num_leaves, len(leaves))
    return jax.tree.unflatten(treedef, leaves)


_STASH: list[jax.Array] = []
_ORIG_MOE_CALL = M.MoEMLP.__call__


def _stashing_moe_call(self, x, projected=None):
    _STASH.append(x)
    return _ORIG_MOE_CALL(self, x, projected)


def per_token_losses(model, tokens, segments, *, features: bool):
    @jax.jit
    def run(m, tok, seg):
        _STASH.clear()
        mask = M.AttentionMask.causal().with_segment_ids(seg)
        loss = m.next_token_loss(tok, jnp.ones(tok.shape, jnp.float32), mask=mask, reduction="none")
        feats = tuple(_STASH[i] for i in FEATURE_LAYERS) if features else ()
        _STASH.clear()
        return loss, feats

    losses, feats = [], [[] for _ in FEATURE_LAYERS]
    for start in range(0, tokens.shape[0], BATCH):
        loss, f = run(model, jnp.asarray(tokens[start : start + BATCH]), jnp.asarray(segments[start : start + BATCH]))
        losses.append(np.asarray(loss, np.float32))
        for i, x in enumerate(f):
            feats[i].append(np.asarray(x, np.float16))
    return np.concatenate(losses), [np.concatenate(f) for f in feats] if features else []


def ridge_oof(x: np.ndarray, y: np.ndarray, fold: np.ndarray, alphas=(1.0, 10.0, 100.0, 1000.0)) -> np.ndarray:
    """Out-of-fold ridge predictions (two folds; alpha chosen on an inner split of the training fold)."""
    pred = np.zeros_like(y)
    xd = jnp.asarray(x, jnp.float32)
    xd = jnp.concatenate([xd, jnp.ones((xd.shape[0], 1), jnp.float32)], axis=1)
    for k in (0, 1):
        tr, te = fold != k, fold == k
        idx = np.flatnonzero(tr)
        inner = idx % 5 == 0
        best, best_err = None, np.inf
        for a in alphas:
            w = _ridge(xd[idx[~inner]], jnp.asarray(y[idx[~inner]]), a)
            err = float(jnp.mean((xd[idx[inner]] @ w - y[idx[inner]]) ** 2))
            if err < best_err:
                best, best_err = a, err
        w = _ridge(xd[tr], jnp.asarray(y[tr]), best)
        pred[te] = np.asarray(xd[te] @ w)
        logger.info("ridge fold %d alpha %s", k, best)
    return pred


def _ridge(x, y, alpha):
    gram = x.T @ x + alpha * jnp.eye(x.shape[1], dtype=x.dtype)
    return jnp.linalg.solve(gram, x.T @ y)


def mlp_oof(x: np.ndarray, y: np.ndarray, fold: np.ndarray, *, hidden=256, steps=6000, batch=8192) -> np.ndarray:
    """Out-of-fold predictions of a one-hidden-layer GELU MLP (Adam, Huber loss) trained per fold."""
    pred = np.zeros_like(y)
    tx = optax.adamw(optax.cosine_decay_schedule(1e-3, steps), weight_decay=1e-4)
    xd = jnp.asarray(x, jnp.bfloat16)
    mu, sd = jnp.mean(xd.astype(jnp.float32), 0), jnp.std(xd.astype(jnp.float32), 0) + 1e-3
    for k in (0, 1):
        tr = np.flatnonzero(fold != k)
        key = jax.random.PRNGKey(k)
        k1, k2 = jax.random.split(key)
        params = {
            "w1": jax.random.normal(k1, (x.shape[1], hidden)) / np.sqrt(x.shape[1]),
            "b1": jnp.zeros(hidden),
            "w2": jax.random.normal(k2, (hidden,)) / np.sqrt(hidden),
            "b2": jnp.zeros(()),
        }
        opt = tx.init(params)

        def f(p, xb):
            h = jax.nn.gelu(((xb.astype(jnp.float32) - mu) / sd) @ p["w1"] + p["b1"])
            return h @ p["w2"] + p["b2"]

        @jax.jit
        def step(p, o, idx):
            def loss(p):
                return jnp.mean(optax.huber_loss(f(p, xd[idx]), jnp.asarray(y)[idx], delta=1.0))

            g = jax.grad(loss)(p)
            u, o = tx.update(g, o, p)
            return optax.apply_updates(p, u), o

        rng = np.random.default_rng(k)
        for _ in range(steps):
            params, opt = step(params, opt, jnp.asarray(rng.choice(tr, batch)))
        te = np.flatnonzero(fold == k)
        for s in range(0, te.size, 1 << 18):
            sl = te[s : s + (1 << 18)]
            pred[sl] = np.asarray(jax.jit(f)(params, xd[sl]))
    return pred


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump", default=f"{ROOT}/nx-ctl/{VERSION}/token_dump.npz")
    parser.add_argument("--output", required=True)
    parser.add_argument("--sequences", type=int, default=512)
    args = parser.parse_args()
    with fsspec.open(args.dump, "rb") as f:
        dump = np.load(io.BytesIO(f.read()))
        tokens, segments = dump["tokens"][: args.sequences], dump["segment_ids"][: args.sequences]
    mesh = compact_grug_mesh(expert_axis_size=1, replica_axis_size=1)
    M.MoEMLP.__call__ = _stashing_moe_call
    out = {"tokens": tokens, "segment_ids": segments}
    feats = None
    loaded = {}
    with set_mesh(mesh):
        for name, run, trained_k, k, renorm in EVALS:
            if run not in loaded:
                params = load_state(
                    model_config(trained_k, 2.5), f"{ROOT}/{run}/{VERSION}/checkpoints/step-{STEP}", mesh
                )
                loaded[run] = _to_dropless_local(MP.cast_to_compute(params), implementation="sonic")
            model = with_routing(loaded[run], k, renorm, "sonic")
            want = feats is None and name == "top8_k8"
            loss, f = per_token_losses(model, tokens, segments, features=want)
            if want:
                feats = f
            out[name] = loss.astype(np.float32)
            logger.info("%s mean loss %.5f", name, float(loss[:, :-1].mean()))
    valid = np.ones(tokens.shape, bool)
    valid[:, -1] = False
    valid &= np.concatenate([segments[:, 1:] == segments[:, :-1], np.zeros((tokens.shape[0], 1), bool)], 1)
    fold = np.broadcast_to((np.arange(tokens.shape[0]) % 2)[:, None], tokens.shape)[valid]
    targets = {
        "gain_across": out["top8_k8"] - out["ctl_k10"],
        "gain_within": out["ctl_k8"] - out["ctl_k10"],
        "cost_top8": out["top8_k6"] - out["top8_k8"],
    }
    assert feats is not None
    for li, layer in enumerate(FEATURE_LAYERS):
        x = feats[li].reshape(-1, feats[li].shape[-1])[valid.reshape(-1)]
        for tname, t in targets.items():
            y = np.clip(t[valid], -5.0, 5.0).astype(np.float32)
            for kind, fn in (("ridge", ridge_oof), ("mlp", mlp_oof)):
                pred = fn(x, y, fold)
                full = np.zeros(tokens.shape, np.float32)
                full[valid] = pred
                out[f"pred_{kind}_L{layer}_{tname}"] = full
                r2 = 1 - np.mean((pred - y) ** 2) / np.var(y)
                logger.info("probe %s L%d %s out-of-fold R2 %.4f", kind, layer, tname, r2)
    buffer = io.BytesIO()
    np.savez(buffer, **out)
    with fsspec.open(args.output, "wb") as f:
        f.write(buffer.getvalue())
    logger.info("wrote %s", args.output)


if __name__ == "__main__":
    main()
