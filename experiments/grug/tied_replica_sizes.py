# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Prediction fidelity of tied half/quarter-compute restrictions of trained Belay."""

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
import numpy as np
from jax.sharding import AxisType, Mesh
from rigging.filesystem.s3_compat import configure_coreweave_s3

from experiments.grug.fast_track.model import Transformer
from experiments.grug.fast_track.train import build_train_loader
from experiments.grug.noise_belay.candidate11 import REFERENCE, ROOT, load_model_config
from experiments.grug.noise_belay.candidate11_control_probe import native_control
from experiments.grug.noise_belay.candidate11_data import native_train_dataset
from experiments.grug.tied_drafter.audit import linear_weight_equivalents
from experiments.grug.tied_drafter.calibration import MODEL_SOURCE_SHA256
from experiments.grug.tied_drafter.coordinates import coordinate_map, restrict_model
from experiments.grug.tied_drafter.runtime import sequence_losses
from experiments.grug.tied_replica_sanity import output_difference, predictions

ANCHOR_STEP = 256
VERIFICATION_START = 50000


def size_config(cfg, *, width: int, expert_width: int):
    return replace(
        cfg,
        hidden_dim=width,
        num_heads=width // 128,
        global_local_heads=width // 128,
        latent_dim=width // 2,
        mla_kv_latent_dim=width,
        intermediate_dim=expert_width,
        shared_expert_intermediate_dim=2 * expert_width,
    )


def comparison(before, after, hero_prediction, replica_prediction):
    hero_loss, counts = before
    replica_loss, replica_counts = after
    if not np.array_equal(counts, replica_counts):
        raise AssertionError("Prediction comparisons must score identical tokens")
    _, hero_ids, hero_probs = hero_prediction
    _, replica_ids, replica_probs = replica_prediction
    overlap = (hero_ids[:, :, None] == replica_ids[:, None, :]).any(axis=2).mean()
    return {
        "hero_loss": float(np.average(hero_loss, weights=counts)),
        "replica_loss": float(np.average(replica_loss, weights=counts)),
        "scored_tokens": float(counts.sum()),
        "hero_sequence_losses": hero_loss.tolist(),
        "replica_sequence_losses": replica_loss.tolist(),
        "sequence_token_counts": counts.tolist(),
        "sequence_loss_difference": output_difference(hero_loss, replica_loss),
        "sequence_loss_correlation": float(np.corrcoef(hero_loss, replica_loss)[0, 1]),
        "sampled_top1_agreement": float(np.mean(hero_ids[:, 0] == replica_ids[:, 0])),
        "sampled_top5_set_overlap": float(overlap),
        "sampled_top5_rank_agreement": float(np.mean(hero_ids == replica_ids)),
        "sampled_target_loss_difference": output_difference(hero_prediction[0], replica_prediction[0]),
        "hero_top5_ids": hero_ids.tolist(),
        "replica_top5_ids": replica_ids.tolist(),
        "hero_top5_probabilities": hero_probs.tolist(),
        "replica_top5_probabilities": replica_probs.tolist(),
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
        raise ValueError("Use the native eight-H100 mesh")
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
        "helper_source_sha256": (
            hashlib.sha256(Path(__file__).with_name("tied_replica_sanity.py").read_bytes()).hexdigest()
        ),
        "seed": 0,
        "anchor_step": ANCHOR_STEP,
        "verification_batches": [VERIFICATION_START, VERIFICATION_START + 1],
        "hardware": "8 H100; sequential comparisons of one shared hero checkpoint",
        "size_definition": "Leading linear-weight FLOP estimate per equal-token batch; full arithmetic not measured",
        "scope": (
            "Immediate prediction fidelity after prefix restriction; no replica optimization or proposal acceptance"
        ),
        "training": [],
        "replicas": [],
    }

    def save():
        (args.output / "replica-sizes.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")

    save()
    with jax.set_mesh(mesh), ExitStack() as cleanup:
        dataset, schedule, _ = native_train_dataset(ROOT / "tokenizers/hero-bpe-v16384")
        loader = build_train_loader(dataset, batch_schedule=schedule, mesh=mesh, max_buffered_batches=4)
        train = iter(loader.iter_from_step(0))
        cleanup.callback(train._batches.stop)
        first_batch = next(train)
        verification_iterator = iter(loader.iter_from_step(VERIFICATION_START))
        batches = [next(verification_iterator) for _ in range(2)]
        verification_iterator._batches.stop()
        del verification_iterator
        key = jax.random.split(jax.random.PRNGKey(0), 2)[1]
        leaves, structure, update, mp = native_control(
            cfg, record=record, model_key=key, schedule_steps=record["trainer"]["trainer"]["num_train_steps"]
        )
        began = time.perf_counter()
        executable = update.lower(leaves, first_batch, loop_active=None, router_tie_active=None).compile()
        result["native_update_compile_seconds"] = time.perf_counter() - began
        for step in range(1, ANCHOR_STEP + 1):
            batch = first_batch if step == 1 else next(train)
            began = time.perf_counter()
            leaves, metrics, watches = executable(leaves, batch)
            jax.block_until_ready((leaves, metrics, watches))
            loss = float(metrics["train/loss"])
            if not np.isfinite(loss):
                raise FloatingPointError("Nonfinite hero loss")
            result["training"].append({"step": step, "loss": loss, "update_seconds": time.perf_counter() - began})
            if step % 64 == 0:
                print(json.dumps({"event": "native", "step": step, "loss": loss}), flush=True)
                save()
        state = jax.tree.unflatten(structure, leaves)
        hero = state.params
        # Fidelity needs only weights; release optimizer state before compiling replicas.
        del state, leaves, executable, update
        forward = eqx.filter_jit(sequence_losses)
        prediction = eqx.filter_jit(predictions)
        hero_forward = forward.lower(hero, first_batch, mp).compile()
        hero_prediction = prediction.lower(hero, first_batch, mp).compile()
        reference = []
        for batch in batches:
            loss = jax.device_get(hero_forward(hero, batch, mp))
            repeated = jax.device_get(hero_forward(hero, batch, mp))
            reference.append((loss, jax.device_get(hero_prediction(hero, batch, mp))))
            result.setdefault("hero_repeatability", []).append(output_difference(loss[0], repeated[0]))
        full_cost = linear_weight_equivalents(hero)
        full_count = sum(x.size for x in jax.tree.leaves(hero))
        result["hero_parameters"] = full_count
        result["hero_linear_weight_equivalents"] = full_cost
        result["status"] = "comparing"
        save()
        for name, config in (
            ("full_identity", cfg),
            ("half_compute", size_config(cfg, width=384, expert_width=128)),
            ("quarter_compute", size_config(cfg, width=256, expert_width=48)),
        ):
            template = eqx.filter_eval_shape(Transformer.init, config, key=key)
            mapping = coordinate_map(hero, template)
            selected = eqx.filter_jit(restrict_model)(hero, template, mapping)
            selected = jax.tree.map(lambda x: jax.device_put(x, x.sharding, may_alias=False), selected)
            jax.block_until_ready(selected)
            count = sum(x.size for x in jax.tree.leaves(selected))
            row = {
                "name": name,
                "hidden_width": config.hidden_dim,
                "head_width": config.inferred_head_dim,
                "heads": config.num_heads,
                "routed_expert_width": config.intermediate_dim,
                "shared_expert_width": config.shared_expert_intermediate_dim,
                "parameters": count,
                "parameter_fraction": count / full_count,
                "linear_compute_fraction": linear_weight_equivalents(selected) / full_cost,
                "batches": [],
            }
            began = time.perf_counter()
            replica_forward = forward.lower(selected, first_batch, mp).compile()
            replica_prediction = prediction.lower(selected, first_batch, mp).compile()
            row["compile_seconds"] = time.perf_counter() - began
            for batch, (before, before_prediction) in zip(batches, reference, strict=True):
                after = jax.device_get(replica_forward(selected, batch, mp))
                after_prediction = jax.device_get(replica_prediction(selected, batch, mp))
                row["batches"].append(comparison(before, after, before_prediction, after_prediction))
            times = []
            for _ in range(5):
                began = time.perf_counter()
                out = replica_forward(selected, batches[0], mp)
                jax.block_until_ready(out)
                times.append(time.perf_counter() - began)
            row["forward_seconds"] = times
            if name == "full_identity" and any(
                not b["sequence_loss_difference"]["exact_equal"] or b["sampled_top1_agreement"] != 1
                for b in row["batches"]
            ):
                raise AssertionError("Full-width identity control failed")
            result["replicas"].append(row)
            print(
                json.dumps({"event": "replica", "name": name, "losses": [b["replica_loss"] for b in row["batches"]]}),
                flush=True,
            )
            save()
            del selected, replica_forward, replica_prediction
        result["status"] = "complete"
        save()


if __name__ == "__main__":
    main()
