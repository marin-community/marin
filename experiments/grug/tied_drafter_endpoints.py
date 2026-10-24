# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Sequential cheap-update blocks, full-model endpoint checks and paired continuation."""

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
import jax.numpy as jnp
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

PREFIXES = (8, 16, 32, 64)


def check_state_coordinates(hero, draft):
    """Require complete state arrays to share paths, shapes and dtypes before transfer."""
    a, _ = jax.tree_util.tree_flatten_with_path(hero)
    b, _ = jax.tree_util.tree_flatten_with_path(draft)
    if len(a) != len(b):
        raise ValueError("State leaf counts differ")
    for (pa, xa), (pb, xb) in zip(a, b, strict=True):
        if jax.tree_util.keystr(pa) != jax.tree_util.keystr(pb) or xa.shape != xb.shape or xa.dtype != xb.dtype:
            raise ValueError(f"State coordinate mismatch: {pa}, {pb}")


def endpoint_accepted(before, after, floor):
    """Accept a finite mean loss reduction above the frozen repeatability floor."""
    return bool(np.isfinite(after) and before - after > floor)


def device_snapshot(leaves):
    return jax.block_until_ready([jax.device_put(x, x.sharding, may_alias=False) for x in leaves])


def restore_snapshot(snapshot, shardings):
    return jax.block_until_ready(
        [jax.device_put(x, s, may_alias=False) for x, s in zip(snapshot, shardings, strict=True)]
    )


@jax.jit
def identical_arrays(left, right):
    return jnp.all(jnp.stack([jnp.array_equal(a, b) for a, b in zip(left, right, strict=True)]))


def load_batches(loader, start, count):
    iterator = iter(loader.iter_from_step(start))
    try:
        return [next(iterator) for _ in range(count)]
    finally:
        iterator._batches.stop()


def mean_loss(values):
    return float(np.average(values["losses"], weights=values["counts"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint-prefix", required=True)
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
    record = json.loads(REFERENCE.read_text())["config"]
    record["trainer"]["grad_accum_microbatches"] = protocol["microbatches"]
    for name, value in record["trainer"]["trainer"]["jax_config"].items():
        jax.config.update(name, value)
    cfg = load_model_config(source_sha256=MODEL_SOURCE_SHA256)
    draft_cfg = replace(cfg, num_experts_per_token=protocol["draft_active_experts"])
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
        "protocol": protocol,
        "protocol_sha256": hashlib.sha256(protocol_bytes).hexdigest(),
        "seed": args.seed,
        "episodes": [],
        "recovery": [],
        "checkpoints": {},
        "cost_seconds": {},
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "model_source_sha256": MODEL_SOURCE_SHA256,
    }

    def save():
        (args.output / "endpoints.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")

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
            dataset, schedule, _ = native_train_dataset(ROOT / "tokenizers/hero-bpe-v16384")
            loader = build_train_loader(dataset, batch_schedule=schedule, mesh=mesh, max_buffered_batches=2)
            first = load_batches(loader, 0, 1)[0]
            expected = json.loads(Path(__file__).with_name("tied-large-first-batch.json").read_text())
            if hashlib.sha256(np.asarray(first.tokens).tobytes()).hexdigest() != expected["array_sha256"]["tokens"]:
                raise ValueError("Native first-batch tokens changed")
            key = jax.random.split(jax.random.PRNGKey(args.seed), 2)[1]
            schedule_steps = record["trainer"]["trainer"]["num_train_steps"]
            leaves, hero_tree, hero_update, mp = native_control(
                cfg, record=record, model_key=key, schedule_steps=schedule_steps
            )
            leaves = jax.block_until_ready(leaves)
            shardings = [x.sharding for x in leaves]
            hero_state = jax.tree.unflatten(hero_tree, leaves)
            optimizer = draccus.decode(GrugMoeMuonHConfig, record["optimizer"]).build(schedule_steps)
            draft_shape = eqx.filter_eval_shape(
                initial_state, draft_cfg, optimizer=optimizer, mp=mp, key=key, ema_beta=record["trainer"]["ema_beta"]
            )
            check_state_coordinates(hero_state, draft_shape)
            draft_tree = jax.tree.structure(draft_shape)
            result["state_bytes"] = sum(x.size * x.dtype.itemsize for x in leaves)
            result["state_bytes_per_device"] = sum(x.addressable_shards[0].data.size * x.dtype.itemsize for x in leaves)
            result["parameters"] = sum(x.size for x in jax.tree.leaves(hero_state.params))
            result["state_coordinate_check"] = "passed"
            del hero_state, draft_shape
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
            draft_update = _flat_train_step(draft_step, draft_tree)

            @jax.jit
            def evaluate_arrays(arrays, batch):
                state = jax.tree.unflatten(hero_tree, arrays)
                params = _apply_qb_betas(state.params, state.pending_qb_betas)
                return sequence_losses(params, batch, mp)

            began = time.perf_counter()
            native_exec = hero_update.lower(leaves, first, loop_active=None, router_tie_active=None).compile()
            draft_exec = draft_update.lower(leaves, first, loop_active=None, router_tie_active=None).compile()
            forward = evaluate_arrays.lower(leaves, first).compile()
            charge("compile", began)
            result["compiled_memory"] = {}
            for name, executable in (("native", native_exec), ("draft", draft_exec)):
                analysis = executable.memory_analysis()
                result["compiled_memory"][name] = {
                    "argument_bytes": analysis.argument_size_in_bytes,
                    "temporary_bytes": analysis.temp_size_in_bytes,
                    "output_bytes": analysis.output_size_in_bytes,
                    "alias_bytes": analysis.alias_size_in_bytes,
                }
            step_index = jax.tree_util.tree_leaves_with_path(jax.tree.unflatten(hero_tree, leaves))
            step_index = next(i for i, (path, _) in enumerate(step_index) if jax.tree_util.keystr(path) == ".step")

            def step(arrays, batch, executable, branch, data_step):
                began = time.perf_counter()
                arrays, metrics, watches = executable(arrays, batch)
                jax.block_until_ready((arrays, metrics, watches))
                elapsed = charge(branch + "_updates", began)
                loss = float(metrics["train/loss"])
                if not np.isfinite(loss):
                    raise FloatingPointError(f"Nonfinite {branch} loss at {data_step}")
                if int(arrays[step_index]) != data_step + 1:
                    raise ValueError("Optimizer clock and data cursor disagree")
                row = {"branch": branch, "data_step": data_step, "loss": loss, "seconds": elapsed}
                updates.write(json.dumps(row) + "\n")
                return arrays

            def evaluate(arrays, batches, category):
                began = time.perf_counter()
                values = [jax.device_get(forward(arrays, batch)) for batch in batches]
                charge(category, began)
                losses = np.concatenate([x[0] for x in values]).astype(float)
                counts = np.concatenate([x[1] for x in values]).astype(float)
                if not np.all(np.isfinite(losses)):
                    raise FloatingPointError("Nonfinite evaluation")
                return {"losses": losses.tolist(), "counts": counts.tolist()}

            def store_state(label, arrays):
                if protocol["checkpoint_policy"] == "none":
                    return
                began = time.perf_counter()
                uri = args.checkpoint_prefix + "/" + label + ".eqx"
                fs, path = filesystem_for(uri)
                with fs.open(path, "wb") as stream:
                    eqx.tree_serialise_leaves(stream, jax.tree.unflatten(hero_tree, arrays))
                result["checkpoints"][label] = {"uri": uri, "bytes": fs.size(path), "step": int(arrays[step_index])}
                charge("checkpoint_io", began)
                save()

            result["status"] = "warmup"
            save()
            iterator = iter(loader.iter_from_step(0))
            try:
                for i in range(protocol["warmup_updates"]):
                    leaves = step(leaves, next(iterator), native_exec, "warmup", i)
                    if (i + 1) % 100 == 0:
                        print(json.dumps({"event": "warmup", "step": i + 1}), flush=True)
            finally:
                iterator._batches.stop()
            cursor = protocol["warmup_updates"]
            result["status"] = "copy_validation"
            save()
            validation_batch = load_batches(loader, cursor, 1)[0]
            began = time.perf_counter()
            cpu_snapshot = [np.array(x, copy=True) for x in jax.device_get(leaves)]
            cpu_restored = restore_snapshot(cpu_snapshot, shardings)
            device_copied = device_snapshot(leaves)
            held_snapshot = device_snapshot(leaves)
            if not bool(identical_arrays(cpu_restored, device_copied)):
                raise AssertionError("Host and device snapshots differ before update")
            cpu_updated, cpu_metrics, cpu_watches = native_exec(cpu_restored, validation_batch)
            device_updated, device_metrics, device_watches = native_exec(device_copied, validation_batch)
            jax.block_until_ready(
                (cpu_updated, device_updated, cpu_metrics, device_metrics, cpu_watches, device_watches)
            )
            # Native GPU updates vary even between identical device copies. Check
            # copying and donation isolation exactly; record update repeatability
            # separately instead of requiring determinism from the native kernel.
            reference = restore_snapshot(cpu_snapshot, shardings)
            if not bool(identical_arrays(leaves, reference)):
                raise AssertionError("Donated snapshot update changed the source state")
            if not bool(identical_arrays(held_snapshot, reference)):
                raise AssertionError("Donated snapshot update changed a retained snapshot")
            validation_losses = [float(m["train/loss"]) for m in (cpu_metrics, device_metrics)]
            if not np.all(np.isfinite(validation_losses)):
                raise FloatingPointError("Nonfinite native copy-validation loss")
            result["copy_validation"] = {
                "status": "passed",
                "state_leaves": len(leaves),
                "updates_discarded": 2,
                "comparison": "Exact input copying and source/retained-snapshot isolation after donated updates",
                "inputs_exact": True,
                "source_unchanged": True,
                "retained_snapshot_unchanged": True,
                "native_outputs_exact": bool(identical_arrays(cpu_updated, device_updated)),
                "native_losses": validation_losses,
            }
            charge("copy_validation", began)
            del cpu_snapshot, cpu_updated, device_updated, cpu_metrics, device_metrics, cpu_watches, device_watches
            del held_snapshot, reference
            store_state("initial-full-state", leaves)
            began = time.perf_counter()
            treatment = device_snapshot(leaves)
            control = device_snapshot(leaves)
            charge("state_copy", began)
            anchor = load_batches(loader, protocol["anchor_batch_start"], protocol["monitor_batches"])
            result["anchor_token_sha256"] = [hashlib.sha256(np.asarray(b.tokens).tobytes()).hexdigest() for b in anchor]
            initial = evaluate(leaves, anchor, "monitor_forward")
            repeated = [mean_loss(evaluate(leaves, anchor[:1], "calibration_forward")) for _ in range(3)]
            floor = max(protocol["minimum_loss_gain"], 2 * (max(repeated) - min(repeated)))
            result.update(
                initial_validation=initial,
                acceptance_floor=floor,
                repeated_loss=repeated,
                status="proposing",
                initial_cursor=cursor,
            )
            del leaves
            save()
            for episode in range(protocol["episodes"]):
                began = time.perf_counter()
                batches = load_batches(loader, cursor, 65)
                check_start = protocol["check_batch_start"] + episode
                check = load_batches(loader, check_start, 1)
                fresh_start = protocol["fresh_batch_start"] + episode * protocol["monitor_batches"]
                fresh = load_batches(loader, fresh_start, protocol["monitor_batches"])
                charge("data", began)
                began = time.perf_counter()
                leaves = restore_snapshot(treatment, shardings)
                charge("state_copy", began)
                before = evaluate(leaves, check, "acceptance_forward")
                before_fresh = evaluate(leaves, fresh, "monitor_forward")
                prefixes = {}
                for i in range(64):
                    leaves = step(leaves, batches[i], draft_exec, "draft", cursor + i)
                    if i + 1 in PREFIXES:
                        began = time.perf_counter()
                        prefixes[i + 1] = device_snapshot(leaves)
                        charge("state_copy", began)
                del leaves
                trials = []
                accepted = 0
                for k in reversed(PREFIXES):
                    began = time.perf_counter()
                    leaves = restore_snapshot(prefixes[k], shardings)
                    charge("state_copy", began)
                    after = evaluate(leaves, check, "acceptance_forward")
                    good = endpoint_accepted(mean_loss(before), mean_loss(after), floor)
                    trials.append(
                        {
                            "prefix": k,
                            "loss": mean_loss(after),
                            "gain": mean_loss(before) - mean_loss(after),
                            "accepted": good,
                        }
                    )
                    if good:
                        accepted = k
                        break
                    del leaves
                if accepted == 0:
                    began = time.perf_counter()
                    leaves = restore_snapshot(treatment, shardings)
                    charge("state_copy", began)
                endpoint_fresh = evaluate(leaves, fresh, "monitor_forward")
                leaves = step(leaves, batches[accepted], native_exec, "treatment_native", cursor + accepted)
                after_anchor = evaluate(leaves, anchor, "monitor_forward")
                after_fresh = evaluate(leaves, fresh, "monitor_forward")
                began = time.perf_counter()
                treatment = device_snapshot(leaves)
                del leaves, prefixes
                leaves = restore_snapshot(control, shardings)
                charge("state_copy", began)
                for i in range(accepted + 1):
                    leaves = step(leaves, batches[i], native_exec, "control", cursor + i)
                control_anchor = evaluate(leaves, anchor, "monitor_forward")
                control_fresh = evaluate(leaves, fresh, "monitor_forward")
                began = time.perf_counter()
                control = device_snapshot(leaves)
                del leaves
                charge("state_copy", began)
                row = {
                    "episode": episode,
                    "start_cursor": cursor,
                    "end_cursor": cursor + accepted + 1,
                    "check_batch": check_start,
                    "fresh_batch_start": fresh_start,
                    "check_token_sha256": hashlib.sha256(np.asarray(check[0].tokens).tobytes()).hexdigest(),
                    "starting_check_loss": mean_loss(before),
                    "trials": trials,
                    "accepted_prefix": accepted,
                    "starting_fresh": before_fresh,
                    "endpoint_fresh": endpoint_fresh,
                    "treatment_anchor": after_anchor,
                    "control_anchor": control_anchor,
                    "treatment_fresh": after_fresh,
                    "control_fresh": control_fresh,
                    "cost_seconds": dict(result["cost_seconds"]),
                }
                result["episodes"].append(row)
                cursor += accepted + 1
                run.log(
                    {
                        "episode": episode,
                        "cursor": cursor,
                        "accepted_prefix": accepted,
                        "validation/treatment": mean_loss(after_anchor),
                        "validation/control": mean_loss(control_anchor),
                        "validation/gain": mean_loss(control_anchor) - mean_loss(after_anchor),
                        "fresh/gain": mean_loss(control_fresh) - mean_loss(after_fresh),
                    }
                )
                print(
                    json.dumps(
                        {
                            "event": "episode",
                            "episode": episode,
                            "accepted": accepted,
                            "cursor": cursor,
                            "gain": mean_loss(control_anchor) - mean_loss(after_anchor),
                        }
                    ),
                    flush=True,
                )
                updates.flush()
                save()
                del batches, check, fresh
            result.update(status="native_recovery", recovery_start_cursor=cursor)
            fresh = load_batches(loader, protocol["recovery_batch_start"], protocol["monitor_batches"])
            for branch, snapshot in (("treatment", treatment), ("control", control)):
                leaves = restore_snapshot(snapshot, shardings)
                iterator = iter(loader.iter_from_step(cursor))
                try:
                    for i in range(max(protocol["recovery_steps"]) + 1):
                        if i in protocol["recovery_steps"]:
                            result["recovery"].append(
                                {
                                    "branch": branch,
                                    "native_steps": i,
                                    "cursor": cursor + i,
                                    "anchor": evaluate(leaves, anchor, "monitor_forward"),
                                    "fresh": evaluate(leaves, fresh, "monitor_forward"),
                                }
                            )
                            observation = result["recovery"][-1]
                            run.log(
                                {
                                    "recovery/native_steps": i,
                                    "recovery/cursor": cursor + i,
                                    f"recovery/{branch}_anchor": mean_loss(observation["anchor"]),
                                    f"recovery/{branch}_fresh": mean_loss(observation["fresh"]),
                                }
                            )
                            save()
                        if i < max(protocol["recovery_steps"]):
                            leaves = step(leaves, next(iterator), native_exec, branch + "_recovery", cursor + i)
                finally:
                    iterator._batches.stop()
                store_state(branch + "-final-full-state", leaves)
                del leaves
            result.update(status="complete", final_cursor=cursor + max(protocol["recovery_steps"]))
            updates.flush()
            save()
            run.summary["accepted_blocks"] = sum(r["accepted_prefix"] > 0 for r in result["episodes"])
            run.summary["completed"] = True
            run.summary["cost_seconds"] = result["cost_seconds"]
            print(json.dumps({"event": "complete", "accepted_blocks": run.summary["accepted_blocks"]}), flush=True)


if __name__ == "__main__":
    main()
