# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Full-width identity restriction and matched-state continuation on native Belay."""

import argparse
import hashlib
import json
import logging
import time
from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import AxisType, Mesh
from rigging.filesystem.s3_compat import configure_coreweave_s3

from experiments.grug.fast_track.model import Transformer
from experiments.grug.fast_track.train import build_train_loader
from experiments.grug.noise_belay.candidate11 import REFERENCE, ROOT, load_model_config
from experiments.grug.noise_belay.candidate11_control_probe import native_control
from experiments.grug.noise_belay.candidate11_data import native_train_dataset
from experiments.grug.tied_drafter.calibration import MODEL_SOURCE_SHA256
from experiments.grug.tied_drafter.coordinates import coordinate_map, restrict_model
from experiments.grug.tied_drafter.runtime import sequence_losses
from experiments.grug.tied_drafter.transactions import snapshot_state

ANCHOR_STEP = 256
CONTINUATION_STEPS = 16
CHECKPOINTS = (0, 1, 4, 16)
VERIFICATION_START = 50000


def tree_difference(first, second):
    """Reduce differences on device so full optimizer states need not reach the host."""
    if jax.tree.structure(first) != jax.tree.structure(second):
        raise ValueError("Compared state structures differ")
    left, right = jax.tree.leaves(first), jax.tree.leaves(second)
    difference = [a.astype(jnp.float32) - b.astype(jnp.float32) for a, b in zip(left, right, strict=True)]
    total = float(sum(x.size for x in left))
    square = jnp.stack([jnp.sum(x**2) for x in difference]).sum()
    reference = jnp.stack([jnp.sum(x.astype(jnp.float32) ** 2) for x in left]).sum()
    changed = jnp.stack([jnp.count_nonzero(x).astype(jnp.float32) for x in difference]).sum()
    return {
        "exact_equal": jnp.stack([jnp.array_equal(a, b) for a, b in zip(left, right, strict=True)]).all(),
        "max_absolute": jnp.stack([jnp.max(jnp.abs(x)) for x in difference]).max(),
        "rms": jnp.sqrt(square / total),
        "relative_rms": jnp.sqrt(square / jnp.maximum(reference, 1e-30)),
        "different_fraction": changed / total,
    }


def predictions(params, batch, mp):
    model = mp.cast_to_compute(params)
    # The native prediction helper uses the training head and its soft-cap.
    positions = jnp.asarray([(row, position) for row in range(0, 128, 8) for position in (0, 31, 255, 1023, 2047, 4094)])
    return model.position_predictions(batch.tokens, positions, mask=batch.attn_mask, k=5)


def host_metrics(metrics):
    return {k: np.asarray(v).item() for k, v in jax.device_get(metrics).items()}


def output_difference(first, second):
    a, b = np.asarray(first), np.asarray(second)
    delta = a.astype(np.float64) - b.astype(np.float64)
    return {
        "exact_equal": bool(np.array_equal(a, b)),
        "max_absolute": float(np.max(np.abs(delta))),
        "rms": float(np.sqrt(np.mean(delta**2))),
        "different_fraction": float(np.mean(a != b)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    configure_coreweave_s3()
    record = json.loads(REFERENCE.read_text())["config"]
    for key, value in record["trainer"]["trainer"]["jax_config"].items():
        jax.config.update(key, value)
    cfg = load_model_config(source_sha256=MODEL_SOURCE_SHA256)
    devices = jax.devices("gpu")
    if len(devices) != 8:
        raise ValueError("Use the same eight-H100 mesh as the native hero")
    mesh = Mesh(
        np.asarray(devices).reshape(1, 1, 8, 1),
        ("replica_dcn", "data", "expert", "model"),
        axis_types=(AxisType.Explicit,) * 4,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    result = {
        "status": "initializing",
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "model_source_sha256": MODEL_SOURCE_SHA256,
        "anchor_step": ANCHOR_STEP,
        "continuation_steps": CONTINUATION_STEPS,
        "verification_start": VERIFICATION_START,
        "seed": 0,
        "hardware": "8 H100, same mesh and executable; sequential replica comparison",
        "scope": "Full-width coordinate restriction plus detached complete-state copy; no narrowing or HTTP transfer",
        "threshold_policy": (
            "Require exact copied state; report raw forward/update divergence and repeated-forward variation."
        ),
        "training": [],
        "checks": [],
    }

    def save():
        (args.output / "replica.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")

    save()
    with jax.set_mesh(mesh), ExitStack() as cleanup:
        dataset, schedule, _ = native_train_dataset(ROOT / "tokenizers/hero-bpe-v16384")
        loader = build_train_loader(dataset, batch_schedule=schedule, mesh=mesh, max_buffered_batches=4)
        train = iter(loader.iter_from_step(0))
        # The pinned loader exposes explicit shutdown on its prefetch iterator.
        cleanup.callback(train._batches.stop)
        first_batch = next(train)
        verification_iterator = iter(loader.iter_from_step(VERIFICATION_START))
        verification = [next(verification_iterator) for _ in range(2)]
        verification_iterator._batches.stop()
        del verification_iterator
        key = jax.random.split(jax.random.PRNGKey(0), 2)[1]
        leaves, structure, update, mp = native_control(
            cfg, record=record, model_key=key, schedule_steps=record["trainer"]["trainer"]["num_train_steps"]
        )
        state = jax.tree.unflatten(structure, leaves)
        began = time.perf_counter()
        executable = update.lower(leaves, first_batch, loop_active=None, router_tie_active=None).compile()
        forward = eqx.filter_jit(sequence_losses).lower(state.params, first_batch, mp).compile()
        prediction = eqx.filter_jit(predictions).lower(state.params, first_batch, mp).compile()
        compare = eqx.filter_jit(tree_difference)
        state_compare = compare.lower(state, state).compile()
        parameter_compare = compare.lower(state.params, state.params).compile()
        template = eqx.filter_eval_shape(Transformer.init, cfg, key=key)
        mapping = coordinate_map(state.params, template)
        restrict = eqx.filter_jit(restrict_model).lower(state.params, template, mapping).compile()
        result["compile_seconds"] = time.perf_counter() - began
        del state
        began = time.perf_counter()
        for step in range(1, ANCHOR_STEP + 1):
            batch = first_batch if step == 1 else next(train)
            leaves, metrics, watches = executable(leaves, batch)
            jax.block_until_ready((leaves, metrics, watches))
            loss = float(metrics["train/loss"])
            if not np.isfinite(loss):
                raise FloatingPointError("Nonfinite native loss")
            result["training"].append({"step": step, "loss": loss})
            if step % 64 == 0:
                print(json.dumps({"event": "native", "step": step, "loss": loss}), flush=True)
                save()
        result["native_training_seconds"] = time.perf_counter() - began
        hero = jax.tree.unflatten(structure, leaves)
        # The same map used by the drafter now has identical source/destination shapes.
        selected = restrict(hero.params, template, mapping)
        replica = snapshot_state(replace(hero, params=selected))
        del selected, leaves
        result["initial_state_difference"] = host_metrics(state_compare(hero, replica))
        if not result["initial_state_difference"]["exact_equal"]:
            raise AssertionError("Full-width restriction/copy changed the state")
        result["parameter_count"] = sum(x.size for x in jax.tree.leaves(hero.params))
        result["status"] = "comparing"
        save()

        def evaluate(step, hero, replica):
            row = {"continuation_updates": step, "batches": []}
            for batch in verification:
                before = jax.device_get(forward(hero.params, batch, mp))
                repeated = jax.device_get(forward(hero.params, batch, mp))
                after = jax.device_get(forward(replica.params, batch, mp))
                hp = jax.device_get(prediction(hero.params, batch, mp))
                rp = jax.device_get(prediction(replica.params, batch, mp))
                row["batches"].append(
                    {
                        "hero_loss": float(np.average(before[0], weights=before[1])),
                        "replica_loss": float(np.average(after[0], weights=after[1])),
                        "sequence_loss_difference": output_difference(before[0], after[0]),
                        "repeat_forward_difference": output_difference(before[0], repeated[0]),
                        "sampled_target_loss_difference": output_difference(hp[0], rp[0]),
                        "top5_ids_agreement": float(np.mean(hp[1] == rp[1])),
                        "top5_probability_difference": output_difference(hp[2], rp[2]),
                    }
                )
            row["parameters"] = host_metrics(parameter_compare(hero.params, replica.params))
            row["complete_state"] = host_metrics(state_compare(hero, replica))
            result["checks"].append(row)
            print(json.dumps({"event": "replica_check", "step": step, "parameters": row["parameters"]}), flush=True)
            save()

        evaluate(0, hero, replica)
        for step in range(1, CONTINUATION_STEPS + 1):
            batch = next(train)
            left, _ = jax.tree.flatten(hero)
            right, _ = jax.tree.flatten(replica)
            # Both branches use the exact same executable, optimizer state and examples.
            left, hm, hw = executable(left, batch)
            jax.block_until_ready((left, hm, hw))
            right, rm, rw = executable(right, batch)
            jax.block_until_ready((right, rm, rw))
            hero = jax.tree.unflatten(structure, left)
            replica = jax.tree.unflatten(structure, right)
            if not np.isfinite(float(hm["train/loss"])) or not np.isfinite(float(rm["train/loss"])):
                raise FloatingPointError("Nonfinite continuation loss")
            result.setdefault("continuation", []).append(
                {"step": step, "hero_train_loss": float(hm["train/loss"]), "replica_train_loss": float(rm["train/loss"])}
            )
            if step in CHECKPOINTS:
                evaluate(step, hero, replica)
        result["status"] = "complete"
        save()


if __name__ == "__main__":
    main()
