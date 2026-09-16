# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Recover completed TPP10 artifacts and audit corrected BPB on saved checkpoints.

Training outputs and legacy metrics are preserved. Corrected evaluations live
under a separate, content-addressed repair record.
"""

import argparse
import asyncio
import hashlib
import importlib.util
import json
import logging
import math
import sys
import time
from pathlib import Path

import equinox as eqx
import fsspec
import haliax as hax
import jax
import jmp
import numpy as np
import wandb
from haliax.partitioning import ResourceAxis
from iris.client.client import IrisClient, get_iris_ctx
from iris.cluster.client.job_info import get_job_info
from iris.cluster.types import JobName
from iris.resources.state import TERMINAL_JOB_STATES, JobState
from levanter.checkpoint import load_checkpoint
from levanter.data.text.datasets import DatasetComponent, LmDataConfig
from levanter.eval import TaggedEvaluator, eval_model
from levanter.tracker.tracker import NoopConfig
from levanter.trainer import TrainerConfig
from levanter.utils.hf_utils import byte_length_of_token
from levanter.utils.jax_utils import use_cpu_device
from levanter.utils.mesh import MeshConfig
from marin.execution.artifact import ArtifactRecord, read_record, write_record
from marin.execution.fingerprint import fingerprint_hash
from marin.execution.remote import remote
from marin.execution.step_status import STATUS_SUCCESS, StatusFile, StepAlreadyDone, step_lock
from rigging.provenance import Provenance

from experiments.domain_phase_mix import evaluate_starcoder_tpp10_uncheatable as original_eval
from experiments.domain_phase_mix import starcoder_tpp10 as experiment
from experiments.domain_phase_mix.starcoder_epoch_matching import canonical_sha256, file_sha256

logger = logging.getLogger(__name__)
PALOMA = experiment.PRIMARY_METRIC.removeprefix("eval/").removesuffix("/bpb")
TOLERANCE = 5e-5
SURVEY_TAGS = (PALOMA, *(f"uncheatable_eval/{name}" for name in original_eval.COMPONENTS))


def read_json(uri: str) -> dict:
    with fsspec.open(uri, "rt") as handle:
        return json.load(handle)


def write_json(uri: str, value: dict) -> None:
    with fsspec.open(uri, "wt") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def validate_plan(plan: dict) -> None:
    expected = canonical_sha256({k: v for k, v in plan.items() if k != "repair_sha256"})
    if expected != plan["repair_sha256"]:
        raise ValueError("Repair plan hash mismatch")
    for name, digest in plan["code_sha256"].items():
        if file_sha256(original_eval.launcher.REPO / name) != digest:
            raise ValueError(f"Repair source changed: {name}")
    for row in plan["artifacts"]["rows"]:
        if fingerprint_hash(row["record"]["fingerprint_payload"]) != row["request"]["fingerprint"]:
            raise ValueError("Archived artifact payload does not reproduce its fingerprint")
        if row["record"]["fingerprint"] != row["request"]["fingerprint"]:
            raise ValueError("Archived record identity differs from the training plan")


def root(plan: dict) -> str:
    return f"{experiment.PREFIX}/experiments/tpp10_bpb_repair/{plan['repair_sha256']}"


def final_record(request: dict) -> dict:
    """Read all final-step metrics, rejecting conflicting duplicate records."""
    metrics = {}
    with fsspec.open(request["output_path"] + "/checkpoints/eval_metrics.jsonl", "rt") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("step") != request["total_steps"] - 1:
                continue
            for key, value in row.items():
                if not key.startswith("eval/") or key.endswith(("/total_time", "/loading_time")):
                    continue
                value = float(value)
                if not math.isfinite(value) or (key in metrics and metrics[key] != value):
                    raise ValueError(f"Nonfinite or conflicting final metric: {request['run_name']} {key}")
                metrics[key] = value
    tags = SURVEY_TAGS if "domain" in request else (PALOMA,)
    required = {f"eval/{tag}/{metric}" for tag in tags for metric in ("loss", "bpb")}
    if missing := required - metrics.keys():
        raise ValueError(f"Missing final metrics for {request['run_name']}: {sorted(missing)}")
    return metrics


def training_evidence(request: dict, plan: dict) -> dict:
    """Verify permanent checkpoint metadata and the frozen training receipts."""
    checkpoint = request["output_path"] + f"/checkpoints/step-{request['total_steps'] - 1}"
    metadata = read_json(checkpoint + "/metadata.json")
    if metadata["step"] != request["total_steps"] - 1 or metadata["is_temporary"]:
        raise ValueError(f"Not the final permanent checkpoint: {checkpoint}")
    runtime = read_json(request["output_path"] + "/verified_runtime.json")
    expected = {
        "design_sha256": plan["design_sha256"],
        "versions": plan["runtime_versions"],
        "code_sha256": plan["code_sha256"],
    }
    if runtime != expected:
        raise ValueError(f"Training runtime identity changed: {request['run_name']}")
    if "domain" in request:
        domain = read_json(request["output_path"] + "/domain_runtime.json")
        if domain != {"domain": request["domain"], "extension_code_sha256": plan["extension_code_sha256"]}:
            raise ValueError(f"Domain identity changed: {request['run_name']}")
    return {"path": checkpoint, "metadata": metadata, "metrics": final_record(request)}


def recover_records(plan: dict, client: IrisClient) -> dict:
    """Finalize only successful child jobs with expired leases and verified outputs."""
    api = wandb.Api(timeout=60)
    statuses = {}
    parent = JobName.from_wire(plan["source_coordinator"])
    parent_finished = client.job_state(parent) in TERMINAL_JOB_STATES
    for row in plan["artifacts"]["rows"]:
        request = row["request"]
        name = request["run_name"]
        state = client.job_state(JobName.from_wire(row["iris_job"]))
        if state != JobState.SUCCEEDED:
            statuses[name] = {"state": str(state)}
            continue
        evidence = training_evidence(request, plan["artifacts"]["plan"])
        run = api.run(f"marin-community/marin/{name}")
        mismatched = {
            key: {"wandb": repr(run.summary.get(key)), "saved": value}
            for key, value in evidence["metrics"].items()
            if run.summary.get(key) != value
        }
        if run.state != "finished" or mismatched:
            statuses[name] = {"state": "wandb_mismatch", "wandb_state": run.state, "metrics": mismatched}
            continue
        # A successful child has immutable final outputs, but its live parent
        # remains responsible for publishing the native completion record.
        if not parent_finished:
            statuses[name] = {"state": "verified_output", "checkpoint": evidence["path"]}
            continue
        status = StatusFile(request["output_path"], worker_id="tpp10-record-recovery")
        if status.active_lock_holder() is not None:
            statuses[name] = {"state": "active_lease"}
            continue
        record = read_record(request["output_path"])
        if record is not None and record.fingerprint != request["fingerprint"]:
            raise ValueError(f"Existing artifact identity differs: {name}")
        if status.status != STATUS_SUCCESS:
            try:
                with step_lock(request["output_path"], "tpp10-record-recovery") as locked:
                    if client.job_state(parent) not in TERMINAL_JOB_STATES:
                        raise ValueError("Original coordinator is still live")
                    # Read again under the native artifact lease before publishing completion.
                    if training_evidence(request, plan["artifacts"]["plan"]) != evidence:
                        raise ValueError(f"Training outputs changed during recovery: {name}")
                    if client.job_state(JobName.from_wire(row["iris_job"])) != JobState.SUCCEEDED:
                        raise ValueError(f"Training job state changed during recovery: {name}")
                    restored = ArtifactRecord(**row["record"], provenance=Provenance.capture())
                    if restored.fingerprint != request["fingerprint"]:
                        raise ValueError(f"Archived artifact fingerprint does not reproduce: {name}")
                    write_json(
                        request["output_path"] + "/completion_recovery.json",
                        {
                            "repair_sha256": plan["repair_sha256"],
                            "iris_job": row["iris_job"],
                            "checkpoint": evidence["path"],
                            "metadata": evidence["metadata"],
                            "training_plan_sha256": plan["artifacts"]["plan"]["plan_sha256"],
                        },
                    )
                    write_record(restored)
                    locked.write_status(STATUS_SUCCESS)
            except StepAlreadyDone:
                pass
        record = read_record(request["output_path"])
        if record is None or record.fingerprint != request["fingerprint"] or status.status != STATUS_SUCCESS:
            raise ValueError(f"Completion record did not verify: {name}")
        statuses[name] = {"state": "verified", "fingerprint": record.fingerprint, "checkpoint": evidence["path"]}
    write_json(root(plan) + "/artifact_status.json", statuses)
    return statuses


def retire_finished_coordinator(plan: dict, statuses: dict, client: IrisClient) -> None:
    """Stop the broken parent only after every training output verifies."""
    info = get_job_info()
    if info is None:
        raise ValueError("Coordinator retirement requires an Iris job identity")
    parent = JobName.from_wire(plan["source_coordinator"])
    if info.job_id == parent or info.job_id.to_wire().startswith(parent.to_wire() + "/"):
        raise ValueError("Repair must run outside the original coordinator's subtree")
    if any(state["state"] not in {"verified", "verified_output"} for state in statuses.values()):
        return
    if client.job_state(parent) in TERMINAL_JOB_STATES:
        return
    jobs = client.list_jobs(prefix=parent.to_wire() + "/", limit=None)
    training_jobs = {row["iris_job"] for row in plan["artifacts"]["rows"]}
    observed = {job.job_id.to_wire(): job.state for job in jobs}
    if not training_jobs <= observed.keys():
        raise ValueError("Cannot retire coordinator: training child inventory is incomplete")
    if any(observed[name] != JobState.SUCCEEDED for name in training_jobs):
        return
    if any(state not in TERMINAL_JOB_STATES for state in observed.values()):
        return
    write_json(root(plan) + "/coordinator_retirement.json", {"parent": parent.to_wire(), "children": observed})
    client.cancel_job(parent)
    logger.info("Stopped broken coordinator after all 28 training children succeeded")


def checkpoint_snapshot(request: dict) -> dict:
    """Pin a permanent checkpoint without requiring its stranded executor record."""
    path = request["output_path"] + f"/checkpoints/step-{request['total_steps'] - 1}"
    with fsspec.open(path + "/metadata.json", "rb") as handle:
        encoded = handle.read()
    metadata = json.loads(encoded)
    if metadata["step"] != request["total_steps"] - 1 or metadata["is_temporary"]:
        raise ValueError(f"Not the final permanent checkpoint: {path}")
    return {"path": path, "metadata_sha256": hashlib.sha256(encoded).hexdigest(), "metadata": metadata}


def evaluation_data(paths: dict[str, str]) -> LmDataConfig:
    return LmDataConfig(
        tokenizer=experiment.TOKENIZER,
        cache_dir=None,
        auto_build_caches=False,
        components={name: DatasetComponent(cache_dir=path) for name, path in paths.items()},
        train_weights={name: 0.0 for name in paths},
    )


async def count_population(paths: dict[str, str]) -> dict:
    """Count scored tokens and bytes exactly, independent of evaluation batching."""
    tokenizer = experiment.verified_tokenizer()
    if len(tokenizer) != len(tokenizer.get_vocab()):
        raise ValueError("Tokenizer vocabulary and byte lookup sizes differ")
    byte_lengths = np.array([byte_length_of_token(tokenizer, i) for i in range(len(tokenizer))], dtype=np.int64)
    counts = {}
    for dataset, tags in evaluation_data(paths).tagged_eval_sets(hax.Axis("position", experiment.SEQ_LEN)):
        length = await dataset.async_len()
        tokens = 0
        total_bytes = 0
        for start in range(0, length, 64):
            batch = await dataset.get_batch(list(range(start, min(start + 64, length))))
            for example in batch:
                weights = np.asarray(example.loss_weight.array)
                if not np.all((weights == 0) | (weights == 1)):
                    raise ValueError("Exact TPP10 counts require binary loss masks")
                selected = np.roll(np.asarray(example.tokens.array), -1)[weights.astype(bool)]
                tokens += selected.size
                total_bytes += int(byte_lengths[selected].sum())
        if tokens <= 0 or total_bytes <= 0:
            raise ValueError(f"Empty evaluation population: {tags}")
        for tag in tags:
            counts[tag] = {"examples": length, "tokens": tokens, "bytes": total_bytes}
    return counts


def corrected_metrics(metrics: dict, counts: dict) -> dict:
    """Recover ratio-of-totals BPB from saved token-average losses and exact counts."""
    result = {}
    for tag, count in counts.items():
        key = f"eval/{tag}/loss"
        result[f"eval/{tag}/bpb"] = metrics[key] * count["tokens"] * math.log2(math.e) / count["bytes"]
    component_keys = [f"eval/uncheatable_eval/{name}/bpb" for name in original_eval.COMPONENTS]
    if all(key in result for key in component_keys):
        result[original_eval.METRIC] = math.fsum(result[key] for key in component_keys) / len(component_keys)
    return result


def check_close(observed: float, expected: float, label: str) -> None:
    if not math.isfinite(observed) or not math.isfinite(expected) or abs(observed - expected) > TOLERANCE:
        raise ValueError(f"{label}: observed={observed}, expected={expected}, tolerance={TOLERANCE}")


def audit_checkpoints(plan: dict, counts: dict, requests: list[dict]) -> None:
    """Compare archived and corrected evaluators on the same restored models."""
    experiment.require_central1()
    validate_plan(plan)
    module_spec = importlib.util.spec_from_file_location(
        "tpp10_legacy_eval", original_eval.launcher.REPO / plan["legacy_eval_path"]
    )
    if module_spec is None or module_spec.loader is None:
        raise ValueError("Cannot load the pinned historical evaluator")
    legacy = importlib.util.module_from_spec(module_spec)
    sys.modules[module_spec.name] = legacy
    module_spec.loader.exec_module(legacy)
    trainer = TrainerConfig(
        tracker=NoopConfig(),
        mp=jmp.get_policy("p=f32,c=bfloat16"),
        train_batch_size=requests[0]["request"]["batch_size"],
        per_device_parallelism=-1,
        per_device_eval_parallelism=-1,
        allow_nondivisible_batch_size=True,
        max_eval_batches=None,
        log_jaxprs=False,
        log_xla_hlo=False,
        mesh=MeshConfig(
            axes={"data": -1, "replica": 1, "model": 1},
            compute_mapping={
                "token": (ResourceAxis.REPLICA_DCN, ResourceAxis.REPLICA, ResourceAxis.DATA),
                "token_repeat": (ResourceAxis.REPLICA_DCN, ResourceAxis.REPLICA, ResourceAxis.DATA),
            },
        ),
    )
    trainer.initialize()
    tokenizer = experiment.verified_tokenizer()
    config = experiment.model_config(experiment.Arm(requests[0]["request"]["arm"]))
    paths = plan["evaluation_paths"]
    expanded = {k: v for k, v in paths.items() if k != PALOMA} | {PALOMA: paths[PALOMA]}
    alone = {PALOMA: paths[PALOMA]}
    survey_order = alone | {k: v for k, v in expanded.items() if k != PALOMA}
    with trainer.use_device_mesh():
        evaluators = {}
        for name, cls, population in [
            ("legacy_alone", legacy.TaggedEvaluator, alone),
            ("legacy_expanded", legacy.TaggedEvaluator, expanded),
            ("legacy_survey", legacy.TaggedEvaluator, survey_order),
            ("corrected_alone", TaggedEvaluator, alone),
            ("corrected_expanded", TaggedEvaluator, expanded),
        ]:
            evaluators[name] = cls(
                EvalBatch=trainer.EvalBatch,
                tagged_eval_sets=evaluation_data(population).tagged_eval_sets(config.max_Pos.resize(experiment.SEQ_LEN)),
                loss_fn=original_eval.eval_loss,
                tokenizer=tokenizer,
                device_mesh=trainer.device_mesh,
                axis_mapping=trainer.compute_axis_mapping,
            )
        for item in requests:
            request = item["request"]
            uri = root(plan) + "/checkpoint_audits/" + request["run_name"] + ".json"
            if audit_verified(plan, counts, item):
                continue
            evidence = training_evidence(request, item["training_plan"])
            if checkpoint_snapshot(request) != item["checkpoint"]:
                raise ValueError(f"Pinned checkpoint metadata changed: {request['run_name']}")
            with use_cpu_device():
                model = eqx.filter_eval_shape(config.build, hax.Axis("vocab", len(tokenizer)), key=jax.random.PRNGKey(0))
                model = load_checkpoint(model, evidence["path"], subpath="model")
            model = hax.shard_with_axis_mapping(model, trainer.parameter_axis_mapping)
            names = (
                ["legacy_survey", "corrected_expanded"]
                if "domain" in request
                else ["legacy_alone", "legacy_expanded", "corrected_alone", "corrected_expanded"]
            )
            metrics = {
                name: {k: float(v) for k, v in eval_model(evaluators[name], model, prefix="eval").items()}
                for name in names
            }
            receipt = {
                "repair_sha256": plan["repair_sha256"],
                "request": request,
                "checkpoint": evidence["path"],
                "metadata": evidence["metadata"],
                "metrics": metrics,
                "counts_sha256": canonical_sha256(counts),
                "verified": False,
            }
            if "domain" not in request:
                receipt["legacy_batching_gap"] = (
                    metrics["legacy_expanded"][experiment.PRIMARY_METRIC]
                    - metrics["legacy_alone"][experiment.PRIMARY_METRIC]
                )
            if jax.process_index() == 0:
                write_json(uri, receipt)
            legacy_name = "legacy_survey" if "domain" in request else "legacy_alone"
            for key, value in evidence["metrics"].items():
                if key.endswith(("/bpb", "/loss")) and key in metrics[legacy_name]:
                    check_close(metrics[legacy_name][key], value, f"{request['run_name']} historical {key}")
            if "domain" not in request:
                check_close(
                    metrics["legacy_alone"][f"eval/{PALOMA}/loss"],
                    metrics["legacy_expanded"][f"eval/{PALOMA}/loss"],
                    "Historical token loss batching invariance",
                )
                check_close(
                    metrics["corrected_alone"][experiment.PRIMARY_METRIC],
                    metrics["corrected_expanded"][experiment.PRIMARY_METRIC],
                    "PALOMA batching invariance",
                )
            legacy_counts = counts if "domain" in request else {PALOMA: counts[PALOMA]}
            expected = corrected_metrics(metrics[legacy_name], legacy_counts)
            for key, value in expected.items():
                check_close(metrics["corrected_expanded"][key], value, f"Ratio-of-totals reconstruction {key}")
            receipt["verified"] = True
            if jax.process_index() == 0:
                write_json(uri, receipt)
                logger.info("Verified historical restore and corrected BPB: %s", request["run_name"])
            del model


def collect_corrected(plan: dict, counts: dict, statuses: dict) -> dict:
    for audit in plan["audits"]:
        if not audit_verified(plan, counts, audit):
            raise ValueError(f"Unverified checkpoint audit: {audit['request']['run_name']}")
    rows = []
    for row in plan["artifacts"]["rows"]:
        request = row["request"]
        state = statuses[request["run_name"]]["state"]
        if state not in {"verified", "verified_output"}:
            continue
        metrics = final_record(request)
        rows.append(
            {
                "request": request,
                "metrics": corrected_metrics(metrics, counts),
                "legacy_metrics": metrics,
                "completion_record_verified": state == "verified",
            }
        )
    controls = {}
    for audit in plan["audits"]:
        request = audit["request"]
        if request["percent"] != 0:
            continue
        receipt = read_json(root(plan) + "/checkpoint_audits/" + request["run_name"] + ".json")
        if not receipt["verified"] or receipt["repair_sha256"] != plan["repair_sha256"]:
            raise ValueError("Unverified control audit")
        controls["target" if request["arm"] == "target" else "matched"] = receipt["metrics"]["corrected_expanded"]
    result = {
        "repair_sha256": plan["repair_sha256"],
        "metric": "ratio_of_total_bits_to_total_bytes",
        "counts": counts,
        "rows": rows,
        "controls": controls,
        "complete": all(state["state"] == "verified" for state in statuses.values()),
        "expected_rows": len(plan["artifacts"]["rows"]),
        "omitted": {
            name: state for name, state in statuses.items() if state["state"] not in {"verified", "verified_output"}
        },
        "artifact_recovery_pending": [name for name, state in statuses.items() if state["state"] == "verified_output"],
    }
    snapshot_uri = root(plan) + "/result_snapshots/" + canonical_sha256(result) + ".json"
    fs, _ = fsspec.core.url_to_fs(snapshot_uri)
    if not fs.exists(snapshot_uri):
        write_json(snapshot_uri, result)
    write_json(root(plan) + "/corrected_results.json", result)
    return result


def audit_verified(plan: dict, counts: dict, audit: dict) -> bool:
    uri = root(plan) + "/checkpoint_audits/" + audit["request"]["run_name"] + ".json"
    fs, _ = fsspec.core.url_to_fs(uri)
    if not fs.exists(uri):
        return False
    receipt = read_json(uri)
    if checkpoint_snapshot(audit["request"]) != audit["checkpoint"]:
        raise ValueError(f"Pinned checkpoint changed: {audit['request']['run_name']}")
    return (
        receipt.get("verified") is True
        and receipt["repair_sha256"] == plan["repair_sha256"]
        and receipt["counts_sha256"] == canonical_sha256(counts)
        and receipt["checkpoint"] == audit["checkpoint"]["path"]
        and receipt["metadata"] == audit["checkpoint"]["metadata"]
    )


async def run_audits(plan: dict, counts: dict) -> None:
    calls = []
    for target in (False, True):
        requests = [
            item
            for item in plan["audits"]
            if (item["request"]["arm"] == "target") == target and not audit_verified(plan, counts, item)
        ]
        if not requests:
            continue
        worker = remote(
            audit_checkpoints,
            name=f"bpb-audit-{'target' if target else 'proxy'}",
            resources=original_eval.TPU,
            env_vars={"MARIN_PREFIX": experiment.PREFIX},
        )
        calls.append(asyncio.to_thread(worker, plan, counts, requests))
    outcomes = await asyncio.gather(*calls, return_exceptions=True)
    errors = [outcome for outcome in outcomes if isinstance(outcome, BaseException)]
    if errors:
        raise BaseExceptionGroup("Checkpoint audits failed after all workers settled", errors)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--collect-only", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    plan = json.loads(args.plan.read_text())
    experiment.require_central1()
    validate_plan(plan)
    context = get_iris_ctx()
    if context is None or context.client is None:
        raise ValueError("Record recovery requires the regional Iris coordinator")
    write_json(root(plan) + "/plan.json", plan)
    statuses = recover_records(plan, context.client)
    for audit in plan["audits"]:
        request = audit["request"]
        if "domain" in request and statuses[request["run_name"]]["state"] not in {"verified", "verified_output"}:
            raise ValueError(f"Audit endpoint has not verified: {request['run_name']}")
        training_evidence(request, audit["training_plan"])
        if checkpoint_snapshot(request) != audit["checkpoint"]:
            raise ValueError(f"Audit checkpoint changed: {request['run_name']}")
    if args.collect_only:
        counts = read_json(root(plan) + "/population_counts.json")
    else:
        paths = original_eval.prepare_caches(plan["original_eval_spec"])
        if paths != plan["evaluation_paths"]:
            raise ValueError("Evaluation caches differ from the frozen populations")
        counts_uri = root(plan) + "/population_counts.json"
        fs, _ = fsspec.core.url_to_fs(counts_uri)
        if fs.exists(counts_uri):
            counts = read_json(counts_uri)
        else:
            counts = asyncio.run(count_population(paths))
            write_json(counts_uri, counts)
        asyncio.run(run_audits(plan, counts))
        statuses = recover_records(plan, context.client)
    result = collect_corrected(plan, counts, statuses)
    logger.info("Verified corrected results: %d/28 survey runs and two shared controls", len(result["rows"]))
    while not args.collect_only and not result["complete"]:
        failures = {
            name: state
            for name, state in statuses.items()
            if state["state"] in {"failed", "killed", "worker_failed", "unschedulable", "wandb_mismatch"}
        }
        if failures:
            raise ValueError(f"Source training needs recovery: {failures}")
        retire_finished_coordinator(plan, statuses, context.client)
        time.sleep(570)
        statuses = recover_records(plan, context.client)
        result = collect_corrected(plan, counts, statuses)
        logger.info("Verified corrected results: %d/28 survey runs", len(result["rows"]))


if __name__ == "__main__":
    main()
