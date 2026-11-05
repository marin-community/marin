# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Measure long-block four-expert learning efficiency against native training.

Frozen frontier driver adapted only to permit a declared native learning-rate
floor tail and record its schedule. The original experiment source is preserved."""

import argparse
import hashlib
import json
import logging
import time
from dataclasses import replace
from pathlib import Path

import draccus
import equinox as eqx
import jax
import numpy as np
import wandb
from jax.sharding import AxisType, Mesh
from levanter.callbacks.watch import WatchConfig
from rigging.filesystem.buckets import filesystem_for
from rigging.filesystem.s3_compat import configure_coreweave_s3

from experiments.grug.fast_track.optimizer import GrugMoeMuonHConfig
from experiments.grug.fast_track.train import (
    _apply_qb_betas,
    _flat_train_step,
    _make_train_step,
    build_train_loader,
    initial_state,
)
from experiments.grug.noise_belay.candidate11 import REFERENCE, ROOT, load_model_config
from experiments.grug.noise_belay.candidate11_control_probe import native_control
from experiments.grug.noise_belay.candidate11_data import native_train_dataset
from experiments.grug.tied_drafter.calibration import MODEL_SOURCE_SHA256
from experiments.grug.tied_drafter.runtime import sequence_losses
from experiments.grug.tied_drafter_endpoints import (
    check_state_coordinates,
    device_snapshot,
    endpoint_accepted,
    identical_arrays,
    load_batches,
    mean_loss,
)
from experiments.grug.tied_drafter_latency import restored_leaf


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    configure_coreweave_s3()
    protocol_bytes = args.protocol.read_bytes()
    protocol = json.loads(protocol_bytes)
    if args.seed not in protocol["seeds"]:
        raise ValueError("Seed outside frozen protocol")
    for name, expected in protocol["implementation_sha256"].items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Frozen source changed: {name}")
    horizons = protocol["horizons"]
    variants = protocol["active_experts"]
    if horizons != sorted(set(horizons)) or horizons[0] < 1:
        raise ValueError("Horizons must be unique increasing positive integers")
    if len(variants) != len(set(variants)) or any(k < 1 or k > 8 for k in variants):
        raise ValueError("Invalid active-expert configurations")
    record = json.loads(REFERENCE.read_text())["config"]
    record["trainer"]["grad_accum_microbatches"] = protocol["microbatches"]
    for name, value in record["trainer"]["trainer"]["jax_config"].items():
        jax.config.update(name, value)
    schedule_steps = record["trainer"]["trainer"]["num_train_steps"]
    if protocol["schedule_steps"] != schedule_steps or protocol["learning_rate_tail"] != "native_floor":
        raise ValueError("This continuation preserves the native schedule and its existing floor tail")
    if protocol["checkpoint_step"] + max(horizons) != protocol["final_optimizer_step"]:
        raise ValueError("Continuation clock differs from the declared endpoint")
    cfg = load_model_config(source_sha256=MODEL_SOURCE_SHA256)
    devices = jax.devices("gpu")
    if len(devices) != 8:
        raise ValueError("Use eight H100s on the native expert mesh")
    mesh = Mesh(
        np.asarray(devices).reshape(1, 1, 8, 1),
        ("replica_dcn", "data", "expert", "model"),
        axis_types=(AxisType.Explicit,) * 4,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    result = {
        "status": "initializing",
        "schedule_steps": schedule_steps,
        "learning_rate_tail": protocol["learning_rate_tail"],
        "seed": args.seed,
        "protocol": protocol,
        "protocol_sha256": hashlib.sha256(protocol_bytes).hexdigest(),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "trials": [],
        "endpoints": [],
        "cost_seconds": {},
    }

    def save():
        (args.output / "frontier.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")

    def charge(category, began):
        elapsed = time.perf_counter() - began
        result["cost_seconds"][category] = result["cost_seconds"].get(category, 0.0) + elapsed
        return elapsed

    save()
    with wandb.init(
        project=protocol["wandb_project"], name=args.run_name, config={"protocol": protocol, "seed": args.seed}
    ) as run:
        result["wandb_url"] = run.url
        with jax.set_mesh(mesh), (args.output / "updates.jsonl").open("w") as updates:
            began = time.perf_counter()
            key = jax.random.split(jax.random.PRNGKey(args.seed), 2)[1]
            leaves, hero_tree, native_update, mp = native_control(
                cfg, record=record, model_key=key, schedule_steps=schedule_steps
            )
            leaves = jax.block_until_ready(leaves)
            template = jax.tree.unflatten(
                hero_tree, [jax.ShapeDtypeStruct(x.shape, x.dtype, sharding=x.sharding) for x in leaves]
            )
            del leaves
            checkpoint = protocol["checkpoints"][str(args.seed)]
            fs, path = filesystem_for(checkpoint["uri"])
            metadata = fs.info(path)
            if metadata["size"] != checkpoint["bytes"] or metadata["ETag"] != checkpoint["etag"]:
                raise ValueError("Checkpoint object differs from the frozen source")
            with fs.open(path, "rb") as stream:
                state = eqx.tree_deserialise_leaves(
                    stream, template, filter_spec=lambda stream, leaf: restored_leaf(stream, leaf, mesh)
                )
                if stream.read(1):
                    raise ValueError("Checkpoint contains unexpected trailing data")
            state = jax.block_until_ready(state)
            if int(state.step) != protocol["checkpoint_step"]:
                raise ValueError("Restored optimizer clock differs from protocol")
            start_leaves = jax.tree.leaves(state)
            step_index = next(
                i
                for i, (p, _) in enumerate(jax.tree_util.tree_leaves_with_path(state))
                if jax.tree_util.keystr(p) == ".step"
            )
            held_snapshot = device_snapshot(start_leaves)
            result["checkpoint"] = {**checkpoint, "restored_step": int(state.step), "leaves": len(start_leaves)}
            charge("restore_and_initialize", began)
            result["status"] = "restored"
            save()
            optimizer = draccus.decode(GrugMoeMuonHConfig, record["optimizer"]).build(schedule_steps)
            update_functions = {}
            for active in variants:
                if active == 8:
                    update_functions[active] = native_update
                    continue
                draft_shape = eqx.filter_eval_shape(
                    initial_state,
                    replace(cfg, num_experts_per_token=active),
                    optimizer=optimizer,
                    mp=mp,
                    key=key,
                    ema_beta=record["trainer"]["ema_beta"],
                )
                check_state_coordinates(state, draft_shape)
                draft_step = _make_train_step(
                    optimizer,
                    mp,
                    z_loss_weight=record["trainer"]["z_loss_weight"],
                    ema_beta=record["trainer"]["ema_beta"],
                    ema_start_step=max(0, schedule_steps - record["trainer"]["ema_last_steps"]),
                    head_replay_period=record["trainer"]["head_replay_period"],
                    head_replay_scale=record["trainer"]["head_replay_scale"],
                    grad_accum_microbatches=protocol["microbatches"],
                    watch_config=draccus.decode(WatchConfig, record["trainer"]["trainer"]["watch"]),
                )
                update_functions[active] = _flat_train_step(draft_step, jax.tree.structure(draft_shape))
            del state, template
            data, schedule, _ = native_train_dataset(ROOT / "tokenizers/hero-bpe-v16384")
            loader = build_train_loader(data, batch_schedule=schedule, mesh=mesh, max_buffered_batches=2)
            first = load_batches(loader, 0, 1)[0]
            expected = json.loads(Path(__file__).with_name("tied-large-first-batch.json").read_text())
            if hashlib.sha256(np.asarray(first.tokens).tobytes()).hexdigest() != expected["array_sha256"]["tokens"]:
                raise ValueError("Native data pipeline fingerprint changed")

            @jax.jit
            def evaluate_arrays(arrays, batch):
                current = jax.tree.unflatten(hero_tree, arrays)
                params = _apply_qb_betas(current.params, current.pending_qb_betas)
                return sequence_losses(params, batch, mp)

            began = time.perf_counter()
            result["status"] = "compiling"
            save()
            executables = {
                k: fn.lower(start_leaves, first, loop_active=None, router_tie_active=None).compile()
                for k, fn in update_functions.items()
            }
            forward = evaluate_arrays.lower(start_leaves, first).compile()
            charge("compile", began)
            del first, update_functions

            def evaluate(arrays, batches, category):
                began = time.perf_counter()
                values = [jax.device_get(forward(arrays, batch)) for batch in batches]
                elapsed = charge(category, began)
                losses = np.concatenate([x[0] for x in values]).astype(float)
                counts = np.concatenate([x[1] for x in values]).astype(float)
                if not np.all(np.isfinite(losses)):
                    return None, elapsed
                return {"losses": losses.tolist(), "counts": counts.tolist()}, elapsed

            result["status"] = "screening"
            save()
            for trial in protocol["trial_ids"]:
                began = time.perf_counter()
                data_start = protocol["training_batch_start"] + trial * protocol["training_window_stride"]
                check_start = protocol["check_batch_start"] + trial
                fresh_start = protocol["fresh_batch_start"] + trial * protocol["monitor_batches"]
                batches = load_batches(loader, data_start, max(horizons))
                check = load_batches(loader, check_start, 1)
                fresh = load_batches(loader, fresh_start, protocol["monitor_batches"])
                data_seconds = charge("data", began)
                before, baseline_seconds = evaluate(start_leaves, check, "baseline_check")
                before_fresh, _ = evaluate(start_leaves, fresh, "monitor_forward")
                if before is None or before_fresh is None:
                    raise FloatingPointError("Nonfinite restored native checkpoint loss")
                repeats = [mean_loss(before)]
                for _ in range(2):
                    repeated, _ = evaluate(start_leaves, check, "calibration_forward")
                    if repeated is None:
                        raise FloatingPointError("Nonfinite calibration loss")
                    repeats.append(mean_loss(repeated))
                floor = max(protocol["minimum_loss_gain"], 2 * (max(repeats) - min(repeats)))
                offset = trial % len(variants)
                order = variants[offset:] + variants[:offset]
                trial_record = {
                    "trial": trial,
                    "training_batch_start": data_start,
                    "check_batch": check_start,
                    "fresh_batch_start": fresh_start,
                    "variant_order": order,
                    "first_training_token_sha256": hashlib.sha256(np.asarray(batches[0].tokens).tobytes()).hexdigest(),
                    "last_training_token_sha256": hashlib.sha256(np.asarray(batches[-1].tokens).tobytes()).hexdigest(),
                    "prefix_last_token_sha256": {
                        str(h): hashlib.sha256(np.asarray(batches[h - 1].tokens).tobytes()).hexdigest() for h in horizons
                    },
                    "check_token_sha256": hashlib.sha256(np.asarray(check[0].tokens).tobytes()).hexdigest(),
                    "fresh_token_sha256": [hashlib.sha256(np.asarray(b.tokens).tobytes()).hexdigest() for b in fresh],
                    "starting_check": before,
                    "starting_fresh": before_fresh,
                    "repeated_loss": repeats,
                    "acceptance_floor": floor,
                    "data_seconds": data_seconds,
                    "baseline_check_seconds": baseline_seconds,
                }
                result["trials"].append(trial_record)
                for active in order:
                    began = time.perf_counter()
                    arrays = device_snapshot(start_leaves)
                    copy_seconds = charge("state_copy", began)
                    if not bool(identical_arrays(arrays, start_leaves)):
                        raise AssertionError("Proposal copy differs from starting state")
                    update_seconds = 0.0
                    check_seconds = 0.0
                    failure_step = None
                    for i in range(1, max(horizons) + 1):
                        began = time.perf_counter()
                        arrays, metrics, watches = executables[active](arrays, batches[i - 1])
                        jax.block_until_ready((arrays, metrics, watches))
                        elapsed = charge(f"k{active}_updates", began)
                        update_seconds += elapsed
                        train_loss = float(metrics["train/loss"])
                        if int(arrays[step_index]) != protocol["checkpoint_step"] + i:
                            raise ValueError("Optimizer clock incorrectly follows the new data-window index")
                        finite = bool(np.isfinite(train_loss))
                        updates.write(
                            json.dumps(
                                {
                                    "trial": trial,
                                    "active_experts": active,
                                    "proposal_step": i,
                                    "optimizer_step": int(arrays[step_index]),
                                    "data_step": data_start + i - 1,
                                    "loss": train_loss if finite else None,
                                    "finite": finite,
                                    "seconds": elapsed,
                                },
                                allow_nan=False,
                            )
                            + "\n"
                        )
                        if not finite:
                            if active == 8:
                                raise FloatingPointError("Native control diverged")
                            failure_step = i
                        if i in horizons or failure_step is not None:
                            pending = [h for h in horizons if h >= i] if failure_step is not None else [i]
                            for horizon in pending:
                                after, checked_seconds = (
                                    (None, 0.0)
                                    if failure_step is not None
                                    else evaluate(arrays, check, "endpoint_check")
                                )
                                check_seconds += checked_seconds
                                monitor, _ = (
                                    (None, 0.0)
                                    if failure_step is not None
                                    else evaluate(arrays, fresh, "monitor_forward")
                                )
                                if active == 8 and (after is None or monitor is None):
                                    raise FloatingPointError("Nonfinite native endpoint evaluation")
                                accepted = after is not None and endpoint_accepted(
                                    mean_loss(before), mean_loss(after), floor
                                )
                                row = {
                                    "trial": trial,
                                    "active_experts": active,
                                    "horizon": horizon,
                                    "optimizer_step": int(arrays[step_index]),
                                    "failure_step": failure_step,
                                    "accepted": accepted,
                                    "check": after,
                                    "fresh": monitor,
                                    "check_gain": None if after is None else mean_loss(before) - mean_loss(after),
                                    "fresh_gain": (
                                        None if monitor is None else mean_loss(before_fresh) - mean_loss(monitor)
                                    ),
                                    "update_seconds": update_seconds,
                                    "initial_copy_seconds": copy_seconds,
                                    "candidate_check_seconds": checked_seconds,
                                    "all_prefix_check_seconds": check_seconds,
                                }
                                result["endpoints"].append(row)
                                run.log(
                                    {
                                        "trial": trial,
                                        "active_experts": active,
                                        "horizon": horizon,
                                        "accepted": int(accepted),
                                        "check_gain": row["check_gain"],
                                        "fresh_gain": row["fresh_gain"],
                                        "update_seconds": update_seconds,
                                    }
                                )
                                print(
                                    json.dumps(
                                        {
                                            "event": "endpoint",
                                            "trial": trial,
                                            "active_experts": active,
                                            "horizon": horizon,
                                            "accepted": accepted,
                                            "check_gain": row["check_gain"],
                                            "fresh_gain": row["fresh_gain"],
                                        }
                                    ),
                                    flush=True,
                                )
                            updates.flush()
                            save()
                        if failure_step is not None:
                            break
                    del arrays
                    if not bool(identical_arrays(start_leaves, held_snapshot)):
                        raise AssertionError("Donated proposal updates changed the retained checkpoint")
                del batches, check, fresh
                result["completed_trials"] = len(result["trials"])
                save()
            result["status"] = "complete"
            result["snapshot_isolation"] = "passed"
            save()
            run.summary["completed"] = True
            run.summary["endpoints"] = len(result["endpoints"])
            run.summary["cost_seconds"] = result["cost_seconds"]


if __name__ == "__main__":
    main()
