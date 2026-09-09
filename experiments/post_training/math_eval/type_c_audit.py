# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Audit synchronous reuse experiments without inheriting anchor-only assumptions."""

import collections
import hashlib
import json
import math
import posixpath

import yaml

from experiments.post_training import async_rl_audit as audit
from experiments.post_training import async_rl_optimizer_gate as optimizer


def validate_checkpoint_receipts(envelope, terminal, attempt, receipt, resolved, *, minibatches):
    """Bind a frozen reuse-ladder request to its actual native checkpoint step."""
    if type(minibatches) is not int or minibatches not in (1, 2, 4, 8, 16):
        raise ValueError("Reuse ladder requires a declared integer minibatch count")
    native_step = 96 // minibatches
    recipe = yaml.safe_load(envelope["request"]["config_yaml"])
    trainer = recipe["trainer"]
    if (
        audit.canonical_entrypoint(recipe["entrypoint"]) != "standard"
        or trainer["train_batch_size"] != 64 * minibatches
        or trainer["policy_mini_batch_size"] != 64
        or trainer["max_steps"] != native_step
        or trainer["eval_interval"] != 32 // minibatches
        or trainer["update_epochs_per_batch"] != 1
    ):
        raise ValueError("Frozen request differs from the 96-update synchronous reuse schedule")
    request = envelope["request"]
    output = request["output"]
    if (
        envelope.get("schema_version") != 2
        or terminal.get("schema_version") != 2
        or terminal != attempt
        or terminal["request"] != request
        or terminal["execution"] != envelope["execution"]
        or request["completion_mode"] != "checkpoint"
        or request["seed"] not in (17, 29)
    ):
        raise ValueError("Checkpoint terminal differs from its exact frozen request/attempt")
    response = terminal["response"]
    if (
        response.get("state") != "succeeded"
        or response.get("iris_job_state") != "succeeded"
        or response.get("failure") is not None
        or response.get("runtime") != request["runtime"]
        or response.get("run_id") != request["run_id"]
        or response.get("attempt_id") != request["attempt_id"]
    ):
        raise ValueError("Checkpoint launcher did not finish the exact request successfully")
    receipt_uri = posixpath.join(
        posixpath.dirname(output["terminal_manifest_uri"]), "receipts", request["attempt_id"] + ".json"
    )
    fingerprint = audit.canonical_sha({"schema_version": 2, "request": request, "receipt_uri": receipt_uri})
    expected = {
        "schema_version": 1,
        "run_id": request["run_id"],
        "attempt_id": request["attempt_id"],
        "request_fingerprint": fingerprint,
        "completion_mode": "checkpoint",
        "global_step": native_step,
    }
    if any(receipt.get(key) != value for key, value in expected.items()) or set(receipt) != set(expected) | {
        "checkpoint"
    }:
        raise ValueError("Checkpoint completion receipt has wrong step, identity or fingerprint")
    training = response["training"]
    checkpoint = receipt["checkpoint"]
    if (
        training.get("global_step") != native_step
        or training.get("checkpoint") != checkpoint
        or training.get("receipt_uri") != receipt_uri
        or training.get("resolved_config_uri") != output["resolved_config_uri"]
        or checkpoint.get("global_step") != native_step
        or checkpoint.get("checkpoint_path") != output["checkpoint_root"].rstrip("/") + f"/global_step_{native_step}"
    ):
        raise ValueError("Checkpoint and trainer completion proofs disagree")
    arguments = audit.parse_hydra_args(resolved["hydra_args"])
    completion = {
        key.removeprefix("trainer.completion."): value
        for key, value in arguments.items()
        if key.startswith("trainer.completion.")
    }
    if completion != {
        "mode": "checkpoint",
        "run_id": request["run_id"],
        "attempt_id": request["attempt_id"],
        "request_fingerprint": fingerprint,
        "receipt_uri": receipt_uri,
    }:
        raise ValueError("Actual resolved completion mode or identity changed")
    return {
        "schema": "math_eval_type_c_training_receipts_v1",
        "request_fingerprint": fingerprint,
        "terminal_sha256": audit.canonical_sha(terminal),
        "receipt_sha256": audit.canonical_sha(receipt),
        "resolved_config_sha256": audit.canonical_sha(resolved),
        "checkpoint_sha256": audit.canonical_sha(checkpoint),
        "runtime": request["runtime"],
        "training_seed": request["seed"],
        "global_step": native_step,
        "native_checkpoint_file_validation_required": True,
        "saved_successful_updates_validation_required": True,
        "minibatches": minibatches,
        "expected_successful_updates": 96,
    }


def audit_checkpoint_history(run, request, receipt_proof, *, minibatches):
    """Audit each rollout batch and every successful optimizer update separately."""
    steps = 96 // minibatches
    if run.state != "finished":
        raise ValueError("Checkpoint W&B run is not finished")
    cfg = dict(run.config)
    expected = {
        "trainer.completion.mode": "checkpoint",
        "trainer.completion.run_id": request["run_id"],
        "trainer.completion.attempt_id": request["attempt_id"],
        "trainer.completion.request_fingerprint": receipt_proof["request_fingerprint"],
        "trainer.seed": request["seed"],
        "trainer.max_steps": steps,
        "trainer.eval_interval": 32 // minibatches,
        "trainer.ckpt_interval": steps,
        "data.epoch_seeded_shuffle": True,
        "data.num_workers": 0,
        "trainer.train_batch_size": 64 * minibatches,
        "trainer.policy_mini_batch_size": 64,
        "trainer.update_epochs_per_batch": 1,
        "generator.n_samples_per_prompt": 4,
        "trainer.algorithm.use_kl_in_reward": False,
        "trainer.algorithm.use_kl_loss": False,
        "trainer.algorithm.policy_loss_type": "behavior_clip",
        "trainer.algorithm.use_tis": False,
        "trainer.max_prompt_length": 1024,
        "generator.max_input_length": 1024,
        "generator.sampling_params.max_generate_length": 1024,
        "generator.engine_init_kwargs.max_model_len": 2048,
    }
    mismatches = [key for key, value in expected.items() if audit.lookup(cfg, key) != value]
    if mismatches:
        raise ValueError("Checkpoint W&B configuration differs from frozen reuse controls: " + ", ".join(mismatches))
    required = ["policy/updates_completed", "policy/updates_completed_valid"]
    required.extend(f"policy/by_update/{i}/optimizer_step_succeeded" for i in range(minibatches))
    history, evaluations = audit.summarize_history(run, steps, "standard", required)
    if history["sums"]["consumed/sequences"] != 6144 * 4:
        raise ValueError("Consumed response count differs from the fixed group budget")
    if sorted(evaluations) != [0, 32 // minibatches, 64 // minibatches, steps]:
        raise ValueError("Development evaluation coverage changed")
    ranges = history["ranges"]
    if (
        ranges["policy/updates_completed"]["last"] != 96
        or ranges["policy/updates_completed_valid"]["min"] != 1
        or ranges["policy/updates_completed_valid"]["max"] != 1
        or any(
            ranges[f"policy/by_update/{i}/optimizer_step_succeeded"][key] != 1
            for i in range(minibatches)
            for key in ("min", "max")
        )
    ):
        raise ValueError("Optimizer success counters do not establish 96 successful updates")
    if len(history["consumed_uid_digests"]) != steps:
        raise ValueError("Missing consumed UID-set digest; do not infer source coverage")
    return history, evaluations


def audit_native_capture(capture, *, minibatches, expected_source_uids):
    """Join actual update fields, source vectors, integer hashes and mask-token ages."""
    steps = 96 // minibatches
    rows = capture["results"]
    parsed = []
    terminals = {}
    for row in rows["events"]:
        body, attrs = json.loads(row["body_json"]), json.loads(row["attributes_json"])
        if row["name"] == "terminal":
            role = attrs["role"]
            if (
                role in terminals
                or body["status"] != "completed"
                or any(body[key] != 0 for key in ("export_lost_records", "export_queued_records"))
            ):
                raise ValueError("Native terminal is duplicated, incomplete or lost telemetry")
            terminals[role] = body
        else:
            parsed.append({"name": row["name"], "step": int(attrs["step"]), "body": body})
    if set(terminals) != {"trainer", "driver", "worker", "controller"}:
        raise ValueError("Incomplete native role terminal coverage")
    scalars = {}
    for row in rows["scalars"]:
        key = (row["step"], row["metric"])
        if key in scalars or not math.isfinite(row["value"]):
            raise ValueError("Duplicated or nonfinite native scalar")
        scalars[key] = row["value"]
    history = []
    for step in range(1, steps + 1):
        history.append(
            {
                "global_step": step,
                "policy/policy_update_steps": minibatches,
                **{metric: value for (index, metric), value in scalars.items() if index == step},
            }
        )
        if (
            scalars[step, "policy/updates_completed"] != step * minibatches
            or scalars[step, "policy/updates_completed_valid"] != 1
        ):
            raise ValueError("Native successful optimizer update count differs")
    update_proof = optimizer.audit_policy_update_events(
        history, [row for row in parsed if row["name"] == "policy_update"]
    )
    source = optimizer.audit_source_order_events(
        [row for row in parsed if row["name"] == "consumed_source_order"],
        minibatches=minibatches,
        rollout_batches=steps,
    )
    if source["ordered_uids"] != expected_source_uids or len(expected_source_uids) != 6144:
        raise ValueError("Actual source vector differs from the qualified loader order")
    for step in range(1, steps + 1):
        uids = source["ordered_uids"][(step - 1) * 64 * minibatches : step * 64 * minibatches]
        digest = int(hashlib.sha256(json.dumps(sorted(set(uids))).encode()).hexdigest()[:13], 16)
        value = scalars[step, "consumed/uid_digest_u52"]
        if type(value) not in (int, float) or value != int(value) or value != digest:
            raise ValueError("Native UID hash must equal the exact integer without a numeric tolerance")
    work = {}
    loss_tokens = {}
    for row in rows["work"]:
        step, kind, value = row["step"], row["work_kind"], row["value"]
        if kind == "consumed_loss_token":
            if step in loss_tokens:
                raise ValueError("Duplicated loss-token counter")
            loss_tokens[step] = value
        else:
            key = {"consumed_sample": "sequences", "consumed_response_token": "response_tokens"}[kind]
            if key in work.setdefault(step, {}):
                raise ValueError("Duplicated consumed-work counter")
            work[step][key] = value
    if set(work) != set(range(1, steps + 1)) or set(loss_tokens) != set(work):
        raise ValueError("Native work coverage differs")
    age = optimizer.audit_consumed_age_events(
        [row for row in parsed if row["name"] == "consumed_age"],
        work,
        minibatches=minibatches,
        synchronous=True,
    )
    if any(loss_tokens[step] != values["response_tokens"] for step, values in work.items()):
        raise ValueError("Non-agentic loss and response mask counts differ")
    tokens = sum(row["response_tokens"] for row in work.values())
    return {
        "schema": "math_eval_type_c_native_audit_v1",
        "minibatches": minibatches,
        "optimizer": update_proof,
        "native_successful_updates": 96,
        "source_order_sha256": audit.canonical_sha(source["ordered_uids"]),
        "source_groups": 6144,
        "unique_question_indices": len(set(source["ordered_uids"])),
        "exposure_histogram": dict(collections.Counter(collections.Counter(source["ordered_uids"]).values())),
        "integer_digest_joins": steps,
        "consumed_age": age,
        "terminals": terminals,
        "response_mask_tokens": tokens,
        "loss_mask_tokens": sum(loss_tokens.values()),
        "token_weighted_age_mean": (
            sum(row["response_tokens"] * row["token_weighted_age_mean"] for row in age["by_step"].values()) / tokens
        ),
        "capture_sha256": audit.canonical_sha(capture),
    }


def validate_saved_source(trainer_state, *, seed, minibatches, dataset_sha256):
    """Verify saved real-data sampler contract; consumption vectors are checked separately."""
    steps = 96 // minibatches
    contract = {
        "algorithm": "torch-randperm-seed-plus-epoch-v1",
        "seed": seed,
        "dataset_sha256": dataset_sha256,
        "rows": 1918,
        "prompts_per_step": 64 * minibatches,
        "updates_per_batch": minibatches,
    }
    epoch, offset = divmod(steps, 1918 // (64 * minibatches))
    expected = {
        "contract": contract,
        "loader_batch_size": 64 * minibatches,
        "loader_workers": 0,
        "completed_step": steps,
        "epoch": epoch,
        "step_in_epoch": offset,
    }
    if trainer_state["source_order"] != expected or trainer_state["successful_policy_updates"] != 96:
        raise ValueError("Actual saved source contract or successful progress differs")
    return expected


def audit_native_history(capture, selected_history, *, minibatches):
    """Cross-check the retained native update series against independently read W&B rows."""
    steps = 96 // minibatches
    combined = {}
    for row in selected_history:
        step = row.get("global_step", row.get("trainer/global_step"))
        if step is None:
            continue
        target = combined.setdefault(int(step), {})
        for key, value in row.items():
            if key in target and target[key] != value:
                raise ValueError("Conflicting selected W&B rows")
            target[key] = value
    scalars = {(row["step"], row["metric"]): row["value"] for row in capture["results"]["scalars"]}
    comparisons = 0
    for (step, metric), value in scalars.items():
        if not metric.startswith("policy/by_update/") and metric != "consumed/uid_digest_u52":
            continue
        reference = combined[step][metric]
        if metric == "consumed/uid_digest_u52":
            if type(reference) not in (int, float) or reference != int(reference) or reference != value:
                raise ValueError("W&B and native UID digests must match as exact integers")
        elif not math.isfinite(reference) or not math.isclose(value, reference, rel_tol=1e-9, abs_tol=1e-12):
            raise ValueError("Native scalar differs from its actual W&B update")
        comparisons += 1
    maximum_norm_error = 0.0
    for step in range(1, steps + 1):
        for index in range(minibatches):
            prefix = f"policy/by_update/{index}/"
            for key in (
                "optimizer_step_succeeded",
                "grad_norm_valid",
                "stale/statistics_valid",
                "stale/finite_fraction",
                "stale/quantiles_valid",
                "stale/p999_valid",
            ):
                if scalars[step, prefix + key] != 1:
                    raise ValueError("Invalid successful-update or numerical coverage flag")
            if scalars[step, prefix + "stale/quantiles_overflow"] != 0:
                raise ValueError("Exact quantiles overflowed")
            if scalars[step, prefix + "grad_cosine_valid"] != int(step > 1 or index > 0):
                raise ValueError("Gradient cosine validity differs from consecutive successful updates")
            raw, norm = (scalars[step, prefix + key] for key in ("raw_grad_norm", "grad_norm_reduced"))
            target = min(raw, 1.0)
            if target <= 0:
                raise ValueError("Observed run cannot qualify nonzero gradient coverage")
            error = abs(norm - target) / target
            if error >= 1e-3:
                raise ValueError("Post-clip gradient coverage differs")
            maximum_norm_error = max(maximum_norm_error, error)
    tokens = 0
    for row in capture["results"]["events"]:
        if row["name"] != "consumed_age":
            continue
        body, attrs = json.loads(row["body_json"]), json.loads(row["attributes_json"])
        step = int(attrs["step"])
        if body["response_tokens"] != scalars[step, f"policy/by_update/{body['age']}/stale/selected_tokens"]:
            raise ValueError("Update age token count differs from its actual selected mask population")
        tokens += body["response_tokens"]
    training = [combined[step] for step in range(1, steps + 1)]
    step_walls = [row.get("timing/step") for row in training]
    has_step_walls = all(isinstance(value, (int, float)) and math.isfinite(value) and value > 0 for value in step_walls)
    return {
        "native_wandb_scalar_joins": comparisons,
        "optimizer_updates": 96,
        "maximum_postclip_relative_norm_error": maximum_norm_error,
        "response_mask_tokens": tokens,
        "exact_integer_digest_joins": steps,
        "summed_training_step_wall_seconds": sum(step_walls) if has_step_walls else None,
        "step_wall_scope": (
            "Generation through weight sync; excludes startup, evaluation, checkpoint callbacks and loader iteration."
        ),
        "dedicated_core_seconds_available": any(any(key.endswith("core_seconds") for key in row) for row in training),
        "selected_history_sha256": audit.canonical_sha(selected_history),
    }
