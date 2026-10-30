# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compare norm-matched directions from a complete reference TrainerState checkpoint."""

import argparse
import json
import time
from dataclasses import asdict, dataclass
from functools import partial
from itertools import islice
from pathlib import Path

import draccus
import equinox as eqx
import haliax as hax
import jax
import jax.numpy as jnp
import numpy as np
import optax
from levanter.checkpoint import load_checkpoint
from levanter.data.loader import DataLoader
from levanter.main.train_lm import TrainLmConfig
from levanter.optim.ema_gram import (
    NEWTON_SCHULZ,
    SVD,
    EmaGramConfig,
    EmaGramState,
    GramState,
    gram_direction,
    relative_gram_power,
)
from levanter.optim.muon import MuonConfig
from levanter.optim.util import flatten_linear_layers, unflatten_linear_layers
from levanter.trainer_state import TrainerState
from levanter.utils.tree_utils import inference_mode

from experiments.ema_gram.plan import ALPHAS, RIDGES


@dataclass(frozen=True)
class LossProbe:
    alpha: float
    ridge: float
    step_size: float
    baseline_losses: list[float]
    losses: list[float]
    delta_mean: float
    delta_std: float
    delta_sem: float


@dataclass(frozen=True)
class DecompositionDiagnostic:
    category: str
    path: str
    shape: list[int]
    seconds: dict[str, float]
    alpha: float
    ridge: float
    cosine: float | None
    relative_error: float | None


def matrix_state(model, opt_state):
    states = [
        s
        for s in jax.tree.leaves(opt_state, is_leaf=lambda x: isinstance(x, EmaGramState))
        if isinstance(s, EmaGramState)
    ]
    if len(states) != 1:
        raise ValueError("A complete EMA-Gram optimizer checkpoint is required")
    labels = MuonConfig().create_mask(model, use_kimi_scaling=True)
    masked = jax.tree.map(lambda label, p: p if label == "muon" else optax.MaskedNode(), labels, model)
    flattened = flatten_linear_layers(masked)
    leaves, treedef = jax.tree.flatten(flattened)
    if len(leaves) != len(states[0].matrices):
        raise ValueError("Checkpoint matrix layout does not match the model")
    return masked, treedef, states[0]


def norm_matched_direction(model, opt_state, *, alpha: float, ridge: float, polar_steps: int):
    """Hold nonmatrices fixed and match each matrix to the alpha=0 direction norm."""
    masked, treedef, state = matrix_state(model, opt_state)
    directions = []
    for s in state.matrices:
        reference = gram_direction(s, alpha=0, ridge=ridge, polar_steps=polar_steps)
        candidate = gram_direction(s, alpha=alpha, ridge=ridge, polar_steps=polar_steps)
        target = jnp.linalg.norm(reference, axis=(-2, -1), keepdims=True)
        norm = jnp.linalg.norm(candidate, axis=(-2, -1), keepdims=True)
        directions.append(candidate * target / jnp.where(norm > 0, norm, 1))
    direction = unflatten_linear_layers(masked, treedef.unflatten(directions))
    return jax.tree.map(
        lambda d, p: jax.tree.map(jnp.zeros_like, p) if isinstance(d, optax.MaskedNode) else d,
        direction,
        model,
        is_leaf=lambda x: isinstance(x, optax.MaskedNode),
    )


def loss_probes(state, loss_fn, batches, step_sizes: list[float], *, polar_steps: int):
    """Reuse independent held-out batches across candidates and retain paired loss deltas."""
    if len(batches) < 2 or not step_sizes or any(s <= 0 for s in step_sizes):
        raise ValueError("Use at least two independent batches and positive common step sizes")
    baseline = np.array([float(loss_fn(state.model, batch)) for batch in batches])
    rows = []
    for alpha in ALPHAS:
        for ridge in ((RIDGES[0],) if alpha == 0 else RIDGES):
            direction = norm_matched_direction(
                state.model, state.opt_state, alpha=alpha, ridge=ridge, polar_steps=polar_steps
            )
            for size in step_sizes:
                candidate = eqx.apply_updates(state.model, jax.tree.map(lambda d, size=size: -size * d, direction))
                losses = np.array([float(loss_fn(candidate, batch)) for batch in batches])
                delta = losses - baseline
                rows.append(
                    LossProbe(
                        alpha=alpha,
                        ridge=ridge,
                        step_size=size,
                        baseline_losses=baseline.tolist(),
                        losses=losses.tolist(),
                        delta_mean=float(delta.mean()),
                        delta_std=float(delta.std(ddof=1)),
                        delta_sem=float(delta.std(ddof=1) / np.sqrt(len(delta))),
                    )
                )
    return rows


def decomposition_diagnostics(model, opt_state, *, polar_steps: int):
    """Offline SVD checks on one attention and one MLP matrix, with synchronized timings."""
    masked, _, state = matrix_state(model, opt_state)
    paths = jax.tree_util.tree_flatten_with_path(flatten_linear_layers(masked))[0]
    selected = {}
    for (path, _), s in zip(paths, state.matrices, strict=True):
        name = jax.tree_util.keystr(path)
        category = "attention" if "attn" in name else "mlp" if "mlp" in name else None
        if category is None or category in selected:
            continue
        selected[category] = (name, GramState(*(x.reshape((-1, *x.shape[-2:]))[0] for x in s)))
    rows = []
    for category, (name, s) in selected.items():
        for alpha in ALPHAS:
            for ridge in ((RIDGES[0],) if alpha == 0 else RIDGES):
                directions = {}
                timings = {}
                for method in (NEWTON_SCHULZ, SVD):
                    fn = jax.jit(
                        partial(gram_direction, alpha=alpha, ridge=ridge, polar_steps=polar_steps, polar_method=method)
                    )
                    jax.block_until_ready(fn(s))
                    start = time.perf_counter()
                    directions[method] = jax.block_until_ready(fn(s))
                    timings[method] = time.perf_counter() - start
                approx, exact = (np.asarray(directions[k]) for k in (NEWTON_SCHULZ, SVD))
                denom = np.linalg.norm(approx) * np.linalg.norm(exact)
                power = jax.jit(partial(relative_gram_power, alpha=alpha, ridge=ridge))
                jax.block_until_ready((power(s.left), power(s.right)))
                start = time.perf_counter()
                jax.block_until_ready((power(s.left), power(s.right)))
                timings["gram_powers"] = time.perf_counter() - start
                rows.append(
                    DecompositionDiagnostic(
                        category=category,
                        path=name,
                        shape=list(s.momentum.shape),
                        alpha=alpha,
                        ridge=ridge,
                        seconds=timings,
                        cosine=float(np.vdot(approx, exact) / denom) if denom else None,
                        relative_error=(
                            float(np.linalg.norm(approx - exact) / np.linalg.norm(exact))
                            if np.linalg.norm(exact)
                            else None
                        ),
                    )
                )
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs", type=Path, required=True)
    parser.add_argument("--config-index", type=int, default=0)
    parser.add_argument("--checkpoint", required=True, help="Concrete checkpoint directory, including optimizer state")
    parser.add_argument("--dataset", default="uncheatable_eval/wikipedia_english")
    parser.add_argument("--batch-offset", type=int, default=0)
    parser.add_argument("--batches", type=int, default=8)
    parser.add_argument("--step-sizes", nargs="+", type=float, default=[1e-5, 3e-5, 1e-4])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cfg = draccus.decode(TrainLmConfig, json.loads(args.configs.read_text())[args.config_index]["train_config"])
    if not isinstance(cfg.optimizer, EmaGramConfig):
        raise ValueError("Select a reference EMA-Gram config")
    with cfg.trainer.use_device_mesh():
        vocab = hax.partitioning.round_axis_for_partitioning(
            hax.Axis("vocab", len(cfg.data.the_tokenizer)), cfg.trainer.compute_axis_mapping
        )
        tx = cfg.optimizer.build(cfg.trainer.num_train_steps)
        template = eqx.filter_eval_shape(
            lambda: TrainerState.init(
                tx, cfg.model.build(vocab, key=jax.random.PRNGKey(0)), key=jax.random.PRNGKey(0), mp=cfg.trainer.mp
            )
        )
        state = load_checkpoint(
            template, args.checkpoint, axis_mapping=cfg.trainer.parameter_axis_mapping, allow_partial=False
        )
        dataset = cfg.data.validation_sets(cfg.model.max_Pos)[args.dataset]
        loader = DataLoader(dataset, batch_size=cfg.trainer.EvalBatch, axis_resources=cfg.trainer.compute_axis_mapping)
        batches = list(islice(loader, args.batch_offset, args.batch_offset + args.batches))
        if len(batches) != args.batches:
            raise ValueError("Not enough independent probe batches")

        @hax.named_jit(axis_resources=cfg.trainer.compute_axis_mapping)
        def loss_fn(model, batch):
            model = cfg.trainer.mp.cast_to_compute(inference_mode(model, True))
            return model.compute_next_token_loss(batch, key=None).scalar()

        report = {
            "checkpoint": args.checkpoint,
            "step": int(state.step),
            "dataset": args.dataset,
            "batch_offset": args.batch_offset,
            "batches": args.batches,
            "loss_probes": [
                asdict(r)
                for r in loss_probes(state, loss_fn, batches, args.step_sizes, polar_steps=cfg.optimizer.polar_steps)
            ],
            "decompositions": [
                asdict(r)
                for r in decomposition_diagnostics(state.model, state.opt_state, polar_steps=cfg.optimizer.polar_steps)
            ],
            "device_memory_stats": [d.memory_stats() for d in jax.local_devices()],
        }
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
