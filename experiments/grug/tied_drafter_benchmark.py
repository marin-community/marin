# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Matched native/one-proposal time-to-quality measurement, without diagnostic forks."""

import argparse
import hashlib
import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from dataclasses import asdict
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from iris.client.client import iris_ctx
from iris.cluster.client import get_job_info
from jax.sharding import AxisType, Mesh
from rigging.filesystem.s3_compat import configure_coreweave_s3

from experiments.grug.fast_track.model import Transformer
from experiments.grug.fast_track.train import build_train_loader
from experiments.grug.noise_belay.candidate11 import REFERENCE, ROOT, load_model_config
from experiments.grug.noise_belay.candidate11_control_probe import native_control
from experiments.grug.noise_belay.candidate11_data import native_train_dataset
from experiments.grug.tied_drafter.acceptance import (
    ProgressForecast,
    assess_candidate,
    grouped_loss,
    local_progress_forecast,
)
from experiments.grug.tied_drafter.calibration import MODEL_SOURCE_SHA256
from experiments.grug.tied_drafter.coordinates import (
    DraftRecipe,
    coordinate_map,
    draft_config,
    lift_displacement,
    restrict_model,
)
from experiments.grug.tied_drafter.exchange import Exchange, decode_packet, encode_packet, serve_exchange
from experiments.grug.tied_drafter.runtime import sequence_losses
from experiments.grug.tied_drafter.transactions import commit_parameters


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("native", "draft"), required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--benchmark-protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    configure_coreweave_s3()
    protocol_bytes = args.protocol.read_bytes()
    protocol = json.loads(protocol_bytes)
    benchmark = json.loads(args.benchmark_protocol.read_text())
    if hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != benchmark["implementation_sha256"]:
        raise ValueError("Benchmark source differs from its preregistration")
    for name, digest in protocol["implementation_sha256"].items():
        if hashlib.sha256((args.protocol.parent / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Frozen proposal source changed: {name}")
    protocol_hash = hashlib.sha256(protocol_bytes).hexdigest()
    record = json.loads(REFERENCE.read_text())["config"]
    for key, value in record["trainer"]["trainer"]["jax_config"].items():
        jax.config.update(key, value)
    cfg = load_model_config(source_sha256=MODEL_SOURCE_SHA256)
    devices = jax.devices("gpu")
    if len(devices) != 8:
        raise ValueError("Benchmark requires eight hero H100s")
    mesh = Mesh(
        np.asarray(devices).reshape(1, 1, 8, 1),
        ("replica_dcn", "data", "expert", "model"),
        axis_types=(AxisType.Explicit,) * 4,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    result = {
        "status": "initializing",
        "mode": args.mode,
        "benchmark_protocol": benchmark,
        "proposal_protocol_sha256": protocol_hash,
        "training": [],
        "evaluation": [],
        "candidates": [],
    }

    def save():
        (args.output / "benchmark.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")

    exchange = Exchange()
    context = iris_ctx()
    info = get_job_info()
    save()
    with (
        jax.set_mesh(mesh),
        serve_exchange(exchange, host="0.0.0.0", port=0, token=os.environ["TIED_DRAFTER_TOKEN"]) as port,
    ):
        with (
            context.registry.registered("tied-drafter", f"http://{info.advertise_host}:{port}"),
            ThreadPoolExecutor(max_workers=1) as sender,
            ExitStack() as loader_cleanup,
        ):
            dataset, schedule, _ = native_train_dataset(ROOT / "tokenizers/hero-bpe-v16384")
            loader = build_train_loader(dataset, batch_schedule=schedule, mesh=mesh, max_buffered_batches=4)
            train = iter(loader.iter_from_step(0))
            # This pinned loader exposes shutdown only on its prefetch iterator.
            # Join it before Python tears down JAX and its CPU worker threads.
            loader_cleanup.callback(train._batches.stop)
            first = next(train)
            target_iterator = iter(loader.iter_from_step(20000))
            targets = [next(target_iterator) for _ in range(4)]
            target_iterator._batches.stop()
            del target_iterator
            verification_iterator = iter(loader.iter_from_step(30000 + benchmark["horizon"] * 8))
            verification = next(verification_iterator)
            verification_iterator._batches.stop()
            del verification_iterator
            key = jax.random.split(jax.random.PRNGKey(protocol["seed"]), 2)[1]
            leaves, structure, update, mp = native_control(
                cfg, record=record, model_key=key, schedule_steps=record["trainer"]["trainer"]["num_train_steps"]
            )
            state = jax.tree.unflatten(structure, leaves)
            began = time.perf_counter()
            executable = update.lower(leaves, first, loop_active=None, router_tie_active=None).compile()
            forward = eqx.filter_jit(sequence_losses).lower(state.params, first, mp).compile()
            template = eqx.filter_eval_shape(
                Transformer.init, draft_config(cfg, DraftRecipe(protocol["recipe"])), key=key
            )
            mapping = coordinate_map(state.params, template)
            restrict = eqx.filter_jit(restrict_model).lower(state.params, template, mapping).compile()
            selected = restrict(state.params, template, mapping)
            selected_leaves, selected_tree = jax.tree.flatten(selected)
            shardings = [x.sharding for x in selected_leaves]
            shapes = [x.shape for x in selected_leaves]
            zero = jax.tree.map(jnp.zeros_like, selected)
            lift = (
                eqx.filter_jit(lift_displacement)
                .lower(state.params, zero, selected, mapping, scale=jnp.array(1.0, dtype=jnp.float32))
                .compile()
            )
            # Both arms warm the same forward and application executables before
            # the training clock starts. No optimizer update is discarded.
            warmed = forward(state.params, first, mp)
            unchanged = lift(state.params, zero, zero, mapping, scale=jnp.array(1.0, dtype=jnp.float32))
            jax.block_until_ready((warmed, unchanged))
            del warmed, unchanged, selected, selected_leaves, state
            result["compile_and_forward_warmup_seconds"] = time.perf_counter() - began

            def evaluate(params, batches):
                values = [jax.device_get(forward(params, b, mp)) for b in batches]
                return grouped_loss(
                    np.concatenate([v[0] for v in values]),
                    np.concatenate([v[1] for v in values]),
                    sequences_per_group=protocol["sequences_per_group"],
                )

            result["status"] = "training"
            started = time.perf_counter()
            snapshot_step = None
            offered = None
            proposal_done = False
            evaluation_overlap = 0.0
            for step in range(1, benchmark["last_step"] + 1):
                began = time.perf_counter()
                batch = first if step == 1 else next(train)
                loaded = time.perf_counter()
                leaves, metrics, watches = executable(leaves, batch)
                jax.block_until_ready((leaves, metrics, watches))
                if not np.isfinite(float(metrics["train/loss"])):
                    raise FloatingPointError("Nonfinite native benchmark loss")
                result["training"].append(
                    {
                        "step": step,
                        "loss": float(metrics["train/loss"]),
                        "data_seconds": loaded - began,
                        "update_seconds": time.perf_counter() - loaded,
                        "wall_seconds": time.perf_counter() - started,
                    }
                )
                state = jax.tree.unflatten(structure, leaves)
                if (
                    args.mode == "draft"
                    and snapshot_step is None
                    and step >= protocol["anchor_step"]
                    and exchange.ready.is_set()
                ):
                    began = time.perf_counter()
                    selected = restrict(state.params, template, mapping)
                    copied = jax.tree.map(lambda x: jax.device_put(x, x.sharding, may_alias=False), selected)
                    jax.block_until_ready(copied)
                    result["snapshot_pause_seconds"] = time.perf_counter() - began
                    snapshot_step = step
                    metadata = {
                        "generation": 0,
                        "snapshot_step": step,
                        "horizon": benchmark["horizon"],
                        "protocol_sha256": protocol_hash,
                    }

                    def send(metadata=metadata, copied=copied):
                        exchange.offer(
                            0, encode_packet(metadata, [np.array(x, copy=True) for x in jax.tree.leaves(copied)])
                        )

                    offered = sender.submit(send)
                    offer_started = time.perf_counter()
                    del send, copied, selected
                if offered is not None and not proposal_done and offered.done():
                    offered.result()
                    packet = exchange.poll()
                    if packet is not None:
                        overlap = time.perf_counter() - offer_started
                        verification_started = time.perf_counter()
                        metadata, arrays = decode_packet(packet)
                        del packet
                        if metadata["protocol_sha256"] != protocol_hash or metadata["snapshot_step"] != snapshot_step:
                            raise ValueError("Proposal identity mismatch")
                        if [a.shape for a in arrays] != shapes or any(
                            a.dtype != np.float32 or not np.all(np.isfinite(a)) for a in arrays
                        ):
                            raise ValueError("Invalid displacement schema")
                        displacement = jax.tree.unflatten(
                            selected_tree, [jax.device_put(a, s) for a, s in zip(arrays, shardings, strict=True)]
                        )
                        before = evaluate(state.params, [verification])
                        forecast = local_progress_forecast(
                            [ProgressForecast(**part) for part in protocol["forecast_curve"]],
                            step=step,
                            lookahead=protocol["forecast_lookahead"],
                        )
                        staleness = step - snapshot_step
                        excess = max(0.0, overlap - evaluation_overlap - staleness * protocol["native_step_seconds"])
                        accepted = False
                        for scale in protocol["scales"]:
                            candidate = lift(
                                state.params, zero, displacement, mapping, scale=jnp.array(scale, dtype=jnp.float32)
                            )
                            after = evaluate(candidate, [verification])
                            overhead = (
                                result["snapshot_pause_seconds"] + time.perf_counter() - verification_started + excess
                            )
                            decision = assess_candidate(
                                before,
                                after,
                                forecast,
                                overhead_equivalent_steps=overhead / protocol["native_step_seconds"],
                                assumed_acceptance=protocol["assumed_acceptance"],
                                repeatability_floor=protocol["repeatability_floor"],
                                confidence=protocol["candidate_confidence"],
                                candidate_scales=len(protocol["scales"]),
                            )
                            result["candidates"].append({"scale": scale, "decision": asdict(decision)})
                            accepted = decision.accepted and staleness <= protocol["maximum_staleness_steps"]
                            if accepted:
                                state = commit_parameters(state, candidate)
                                leaves, _ = jax.tree.flatten(state)
                                break
                        del candidate, displacement, arrays
                        result.update(
                            accepted=accepted,
                            staleness_steps=staleness,
                            snapshot_step=snapshot_step,
                            proposal_ready_step=step,
                            overlap_seconds=overlap,
                            verification_pause_seconds=time.perf_counter() - verification_started,
                        )
                        proposal_done = True
                        exchange.closed.set()
                if step in benchmark["evaluation_steps"]:
                    began = time.perf_counter()
                    groups = evaluate(state.params, targets)
                    elapsed = time.perf_counter() - began
                    if snapshot_step is not None and not proposal_done:
                        evaluation_overlap += elapsed
                    result["evaluation"].append(
                        {
                            "step": step,
                            "loss": float(groups.mean()),
                            "groups": groups.tolist(),
                            "wall_seconds": time.perf_counter() - started,
                            "evaluation_seconds": elapsed,
                        }
                    )
                    save()
                del state
                if step % 64 == 0:
                    print(json.dumps({"event": "benchmark", "step": step, "proposal_done": proposal_done}), flush=True)
            result["training_wall_seconds"] = time.perf_counter() - started
            if args.mode == "draft" and not proposal_done:
                raise RuntimeError("No proposal returned before benchmark end")
            if args.mode == "draft" and not exchange.finished.wait(30):
                raise TimeoutError("Draft did not acknowledge completion")
            result["target_attainment"] = []
            for target in protocol["loss_targets"]:
                attained = next((row for row in result["evaluation"] if row["loss"] <= target["loss"]), None)
                result["target_attainment"].append(
                    {
                        "target": target["loss"],
                        "first_observed_step": attained["step"] if attained else None,
                        "first_observed_wall_seconds": attained["wall_seconds"] if attained else None,
                    }
                )
            result["status"] = "complete"
            save()


if __name__ == "__main__":
    main()
