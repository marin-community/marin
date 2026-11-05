# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""One asynchronous proposal followed by matched 64-update persistence branches.

The persistence forks are research diagnostics, not an end-to-end accelerated
training benchmark. Their costs are recorded separately and still charged to
the resource ledger.
"""

import argparse
import hashlib
import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
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
from experiments.grug.tied_drafter.transactions import commit_parameters, snapshot_state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--horizon", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    configure_coreweave_s3()
    protocol_bytes = args.protocol.read_bytes()
    protocol = json.loads(protocol_bytes)
    for name, digest in protocol["implementation_sha256"].items():
        if hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Frozen experiment source changed: {name}")
    if args.horizon not in protocol["horizons"]:
        raise ValueError("Horizon was not preregistered")
    protocol_hash = hashlib.sha256(protocol_bytes).hexdigest()
    record = json.loads(REFERENCE.read_text())["config"]
    for key, value in record["trainer"]["trainer"]["jax_config"].items():
        jax.config.update(key, value)
    cfg = load_model_config(source_sha256=MODEL_SOURCE_SHA256)
    devices = jax.devices("gpu")
    if len(devices) != 8:
        raise ValueError("The initial hero requires eight H100s")
    mesh = Mesh(
        np.asarray(devices).reshape(1, 1, 8, 1),
        ("replica_dcn", "data", "expert", "model"),
        axis_types=(AxisType.Explicit,) * 4,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    result = {
        "status": "initializing",
        "protocol": protocol,
        "protocol_sha256": protocol_hash,
        "horizon": args.horizon,
        "gpu_allocator": os.environ.get("XLA_PYTHON_CLIENT_ALLOCATOR", "default"),
        "training": [],
        "candidates": [],
        "persistence": {},
    }

    def save():
        (args.output / "trial.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")

    save()
    exchange = Exchange()
    context = iris_ctx()
    info = get_job_info()
    with (
        jax.set_mesh(mesh),
        serve_exchange(exchange, host="0.0.0.0", port=0, token=os.environ["TIED_DRAFTER_TOKEN"]) as port,
    ):
        address = f"http://{info.advertise_host}:{port}"
        with context.registry.registered("tied-drafter", address), ThreadPoolExecutor(max_workers=1) as sender:
            dataset, schedule, _ = native_train_dataset(ROOT / "tokenizers/hero-bpe-v16384")
            loader = build_train_loader(dataset, batch_schedule=schedule, mesh=mesh, max_buffered_batches=4)
            iterator = iter(loader.iter_from_step(0))
            first = next(iterator)
            verification_iterator = iter(loader.iter_from_step(30000 + args.horizon * 8))
            verification = [next(verification_iterator) for _ in range(protocol["verification_batches"])]
            del verification_iterator
            persistence_iterator = iter(loader.iter_from_step(40000 + args.horizon * 8))
            persistence = [next(persistence_iterator) for _ in range(4)]
            del persistence_iterator
            target_iterator = iter(loader.iter_from_step(20000))
            targets = [next(target_iterator) for _ in range(4)]
            del target_iterator
            result["verification_token_sha256"] = [
                hashlib.sha256(np.asarray(b.tokens).tobytes()).hexdigest() for b in verification
            ]
            result["persistence_token_sha256"] = [
                hashlib.sha256(np.asarray(b.tokens).tobytes()).hexdigest() for b in persistence
            ]
            result["target_token_sha256"] = [hashlib.sha256(np.asarray(b.tokens).tobytes()).hexdigest() for b in targets]
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
            selected_shardings = [x.sharding for x in selected_leaves]
            selected_shapes = [x.shape for x in selected_leaves]
            zeros = jax.tree.map(jnp.zeros_like, selected)
            scale_value = jnp.array(1.0, dtype=jnp.float32)
            lift = (
                eqx.filter_jit(lift_displacement)
                .lower(state.params, zeros, selected, mapping, scale=scale_value)
                .compile()
            )
            full_mapping = coordinate_map(state.params, state.params)
            extrapolate = (
                eqx.filter_jit(lift_displacement)
                .lower(state.params, state.params, state.params, full_mapping, scale=scale_value)
                .compile()
            )
            del selected, selected_leaves, state
            result["compile_seconds"] = time.perf_counter() - began
            result["warmup_policy"] = "First real update included in the recorded hero trajectory"

            def evaluate(params, batches):
                values = [jax.device_get(forward(params, b, mp)) for b in batches]
                return grouped_loss(
                    np.concatenate([v[0] for v in values]),
                    np.concatenate([v[1] for v in values]),
                    sequences_per_group=protocol["sequences_per_group"],
                )

            result["status"] = "ordinary_training"
            started = time.perf_counter()
            step = 0
            offered = None
            proposal = None
            production_snapshot_seconds = 0.0
            snapshot_step = None
            snapshot_parameters = None
            offer_started = None
            while proposal is None:
                began = time.perf_counter()
                batch = first if step == 0 else next(iterator)
                loaded = time.perf_counter()
                leaves, metrics, watches = executable(leaves, batch)
                jax.block_until_ready((leaves, metrics, watches))
                step += 1
                loss = float(metrics["train/loss"])
                if not np.isfinite(loss):
                    raise FloatingPointError("Nonfinite hero training loss")
                result["training"].append(
                    {
                        "step": step,
                        "loss": loss,
                        "update_seconds": time.perf_counter() - loaded,
                        "data_seconds": loaded - began,
                        "elapsed_seconds": time.perf_counter() - started,
                    }
                )
                if offered is None and step >= protocol["anchor_step"] and exchange.ready.is_set():
                    state = jax.tree.unflatten(structure, leaves)
                    # This extra full snapshot belongs only to the cheap-control diagnostic.
                    began = time.perf_counter()
                    snapshot_parameters = jax.tree.map(
                        lambda x: jax.device_put(x, x.sharding, may_alias=False), state.params
                    )
                    jax.block_until_ready(snapshot_parameters)
                    result["research_control_snapshot_seconds"] = time.perf_counter() - began
                    began = time.perf_counter()
                    draft = restrict(state.params, template, mapping)
                    draft_snapshot = jax.tree.map(lambda x: jax.device_put(x, x.sharding, may_alias=False), draft)
                    jax.block_until_ready(draft_snapshot)
                    production_snapshot_seconds = time.perf_counter() - began
                    del draft, state
                    snapshot_step = step
                    metadata = {
                        "generation": 0,
                        "snapshot_step": step,
                        "horizon": args.horizon,
                        "protocol_sha256": protocol_hash,
                    }

                    def send_snapshot(metadata=metadata, draft_snapshot=draft_snapshot):
                        began = time.perf_counter()
                        arrays = [np.array(x, copy=True) for x in jax.tree.leaves(draft_snapshot)]
                        downloaded = time.perf_counter()
                        payload = encode_packet(metadata, arrays)
                        serialized = time.perf_counter()
                        exchange.offer(0, payload)
                        return {
                            "background_download_seconds": downloaded - began,
                            "background_serialization_seconds": serialized - downloaded,
                            "payload_bytes": len(payload),
                        }

                    offer_started = time.perf_counter()
                    offered = sender.submit(send_snapshot)
                    del send_snapshot, draft_snapshot
                if offered is not None and offered.done():
                    result["snapshot_transfer"] = offered.result()
                    proposal = exchange.poll()
                if step % 64 == 0:
                    save()
                    print(json.dumps({"event": "hero", "step": step, "snapshot_step": snapshot_step}), flush=True)
                if step >= protocol["anchor_step"] + 512:
                    raise TimeoutError("No draft proposal within 512 ordinary hero steps of the anchor")
            result["proposal_ready_step"] = step
            result["snapshot_step"] = snapshot_step
            result["staleness_steps"] = step - snapshot_step
            result["draft_overlap_seconds"] = time.perf_counter() - offer_started
            result["production_snapshot_seconds"] = production_snapshot_seconds
            result["status"] = "verifying"
            save()
            verification_started = time.perf_counter()
            metadata, arrays = decode_packet(proposal)
            del proposal
            if (
                metadata["protocol_sha256"] != protocol_hash
                or metadata["snapshot_step"] != snapshot_step
                or metadata["horizon"] != args.horizon
            ):
                raise ValueError("Received proposal identity differs from the offered snapshot")
            if [a.shape for a in arrays] != selected_shapes or any(
                a.dtype != np.float32 or not np.all(np.isfinite(a)) for a in arrays
            ):
                raise ValueError("Proposal arrays do not match finite FP32 draft coordinates")
            displacement = jax.tree.unflatten(
                selected_tree, [jax.device_put(a, s) for a, s in zip(arrays, selected_shardings, strict=True)]
            )
            jax.block_until_ready(displacement)
            del arrays
            result["draft_metadata"] = metadata
            state = jax.tree.unflatten(structure, leaves)
            began = time.perf_counter()
            baseline_losses = evaluate(state.params, verification)
            result["baseline_verification_seconds"] = time.perf_counter() - began
            result["observed_recent_step_seconds"] = float(
                np.median([x["update_seconds"] + x["data_seconds"] for x in result["training"][-64:]])
            )
            native_step_seconds = protocol["native_step_seconds"]
            overlap_excess = max(0.0, result["draft_overlap_seconds"] - result["staleness_steps"] * native_step_seconds)
            result["estimated_overlap_excess_seconds"] = overlap_excess
            forecast = local_progress_forecast(
                [ProgressForecast(**part) for part in protocol["forecast_curve"]],
                step=step,
                lookahead=protocol["forecast_lookahead"],
            )
            result["local_forecast"] = asdict(forecast)
            selected_candidate = None
            accepted = False
            for scale in protocol["scales"]:
                candidate = lift(state.params, zeros, displacement, mapping, scale=jnp.array(scale, dtype=jnp.float32))
                after = evaluate(candidate, verification)
                overhead = production_snapshot_seconds + time.perf_counter() - verification_started + overlap_excess
                decision = assess_candidate(
                    baseline_losses,
                    after,
                    forecast,
                    overhead_equivalent_steps=overhead / native_step_seconds,
                    assumed_acceptance=protocol["assumed_acceptance"],
                    repeatability_floor=protocol["repeatability_floor"],
                    confidence=protocol["candidate_confidence"],
                    candidate_scales=len(protocol["scales"]),
                )
                eligible = result["staleness_steps"] <= protocol["maximum_staleness_steps"]
                result["candidates"].append(
                    {
                        "scale": scale,
                        "decision": asdict(decision),
                        "within_staleness_limit": eligible,
                        "before_groups": baseline_losses.tolist(),
                        "after_groups": after.tolist(),
                        "overhead_seconds_so_far": overhead,
                    }
                )
                selected_candidate = candidate
                accepted = decision.accepted and eligible
                if accepted:
                    break
            result["accepted"] = accepted
            result["verification_pause_seconds"] = time.perf_counter() - verification_started
            result["native_step_seconds"] = native_step_seconds
            result["selected_scale"] = result["candidates"][-1]["scale"]
            # Evaluate the last predefined scale even when rejected, to test whether
            # an immediate negative verdict hides a later benefit.
            cheap_started = time.perf_counter()
            cheap = extrapolate(
                state.params,
                snapshot_parameters,
                state.params,
                full_mapping,
                scale=jnp.array(
                    protocol["extrapolation_equivalent_steps"] / max(1, result["staleness_steps"]), dtype=jnp.float32
                ),
            )
            cheap_losses = evaluate(cheap, verification)
            cheap_overhead = (
                result["research_control_snapshot_seconds"]
                + time.perf_counter()
                - cheap_started
                + result["baseline_verification_seconds"]
            )
            result["extrapolation_decision"] = asdict(
                assess_candidate(
                    baseline_losses,
                    cheap_losses,
                    forecast,
                    overhead_equivalent_steps=cheap_overhead / native_step_seconds,
                    assumed_acceptance=protocol["assumed_acceptance"],
                    repeatability_floor=protocol["repeatability_floor"],
                    confidence=protocol["candidate_confidence"],
                    candidate_scales=1,
                )
            )
            result["extrapolation_overhead_seconds"] = cheap_overhead
            del snapshot_parameters, displacement, zeros
            save()
            exchange.closed.set()
            # Allow the idle drafter to acknowledge shutdown before closing its endpoint.
            if not exchange.finished.wait(30):
                raise TimeoutError("Drafter did not acknowledge completion")
            result["status"] = "persistence_diagnostics"
            save()
            diagnostic_started = time.perf_counter()
            continuations = [next(iterator) for _ in range(64)]
            jax.block_until_ready(continuations)
            selected_branch = "draft" if accepted else "native"
            outcome = None
            branches = [("native", state.params), ("draft", selected_candidate), ("extrapolation", cheap)]
            # Keep the committed branch last so its full final state does not
            # occupy GPU memory while another diagnostic branch trains.
            branches.sort(key=lambda item: item[0] == selected_branch)
            for name, parameters in branches:
                branch = snapshot_state(commit_parameters(state, parameters))
                branch_leaves, branch_structure = jax.tree.flatten(branch)
                if branch_structure != structure:
                    raise ValueError("Candidate changed optimizer tree structure")
                rows = [
                    {
                        "continuation_updates": 0,
                        "groups": evaluate(branch.params, persistence).tolist(),
                        "target_groups": evaluate(branch.params, targets).tolist(),
                        "update_seconds": 0.0,
                    }
                ]
                del branch
                update_seconds = 0.0
                for index, batch in enumerate(continuations, start=1):
                    began = time.perf_counter()
                    branch_leaves, metrics, watches = executable(branch_leaves, batch)
                    jax.block_until_ready((branch_leaves, metrics, watches))
                    update_seconds += time.perf_counter() - began
                    if not np.isfinite(float(metrics["train/loss"])):
                        raise FloatingPointError(f"Nonfinite {name} persistence loss at {index}")
                    if index in (1, 4, 16, 64):
                        branch = jax.tree.unflatten(structure, branch_leaves)
                        rows.append(
                            {
                                "continuation_updates": index,
                                "groups": evaluate(branch.params, persistence).tolist(),
                                "target_groups": evaluate(branch.params, targets).tolist(),
                                "update_seconds": update_seconds,
                            }
                        )
                        del branch
                if name == selected_branch:
                    outcome = jax.tree.unflatten(structure, branch_leaves)
                del branch_leaves
                result["persistence"][name] = rows
                save()
            result["diagnostic_seconds"] = time.perf_counter() - diagnostic_started
            result["committed_outcome"] = {
                "branch": selected_branch,
                "ordinary_step": int(outcome.step),
                "target_loss": float(np.mean(result["persistence"][selected_branch][-1]["target_groups"])),
            }
            result["status"] = "complete"
            result["total_trial_seconds_after_compilation"] = time.perf_counter() - started
            save()
            print(
                json.dumps({"event": "complete", "accepted": accepted, "staleness": result["staleness_steps"]}),
                flush=True,
            )


if __name__ == "__main__":
    main()
