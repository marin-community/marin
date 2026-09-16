# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Resume the four approved FineMath targets from their preserved full states."""

import argparse
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from functools import partial
from pathlib import Path
from typing import cast

from google.cloud import storage
from iris.client.client import get_iris_ctx
from iris.resources.state import TERMINAL_JOB_STATES
from levanter.checkpoint import latest_checkpoint_path
from levanter.main.train_lm import TrainLmConfig
from marin.execution.artifact import read_record
from marin.execution.lazy import ArtifactStep, materialized_config, run
from marin.execution.remote import remote
from marin.execution.step_status import STATUS_SUCCESS, StatusFile
from marin.training.training import temporary_checkpoint_base_path

from experiments.domain_phase_mix import launch_tpp10_domain_sweeps as survey
from experiments.domain_phase_mix import repair_tpp10_evaluation as repair
from experiments.domain_phase_mix.starcoder_epoch_matching import canonical_sha256, file_sha256

logger = logging.getLogger(__name__)
DIRECTORY = Path(".agents/projects/starcoder_tpp10/domain_sweeps/target_completion_20260912")
SURVEY = DIRECTORY.parent / "plan.json"
PRESERVED = DIRECTORY.parent / "scope_math_audit_20260912/checkpoint_preservation_receipt.json"
PERCENTS = (30, 50, 70, 100)
OLD_PARENT = "/calvinxu/tpp10-domain-sweeps-cpu4g"


def verify_preserved(entry: dict) -> None:
    """Check object metadata without downloading checkpoint tensors."""
    bucket = storage.Client().bucket("marin-us-central1")
    prefix = entry["checkpoint_uri"].removeprefix("gs://marin-us-central1/") + "/"
    objects = {blob.name.removeprefix(prefix): blob for blob in bucket.list_blobs(prefix=prefix)}
    if set(objects) != {obj["relative_name"] for obj in entry["objects"]}:
        raise ValueError(f"Preserved checkpoint object set changed: {entry['run_name']}")
    for obj in entry["objects"]:
        blob = objects[obj["relative_name"]]
        actual = (int(blob.generation), blob.size, blob.crc32c, blob.md5_hash)
        expected = (obj["destination_generation"], obj["size"], obj["crc32c"], obj["md5_hash"])
        if actual != expected:
            raise ValueError(f"Preserved checkpoint object changed: {blob.name}")
    if repair.read_json(entry["checkpoint_uri"] + "/metadata.json") != entry["metadata"]:
        raise ValueError("Preserved checkpoint metadata changed")


def validate_plan(plan: dict) -> None:
    if plan["recovery_sha256"] != canonical_sha256({k: v for k, v in plan.items() if k != "recovery_sha256"}):
        raise ValueError("Recovery plan changed")
    if plan["wrapper_sha256"] != file_sha256(Path(__file__)):
        raise ValueError("Recovery wrapper changed")
    if tuple(item["request"]["percent"] for item in plan["selected"]) != PERCENTS:
        raise ValueError("Recovery must select exactly four unfinished FineMath targets")
    for item in plan["selected"]:
        request = item["request"]
        if request["run_name"] != f"tpp10_finemath_3plus_target_p{request['percent']:03d}_s20260910":
            raise ValueError("Wrong recovery run identity")
        if request["arm"] != "target" or request["domain"] != "finemath_3plus":
            raise ValueError("Recovery cannot release other domains or proxies")


def frozen_steps(plan: dict) -> tuple[ArtifactStep, ...]:
    current, steps = survey.build_plan()
    if current != plan["training_plan"]:
        raise ValueError("The original training plan no longer reproduces exactly")
    by_name = {row["run_name"]: step for row, step in zip(current["runs"], steps, strict=True)}
    selected = tuple(by_name[item["request"]["run_name"]] for item in plan["selected"])
    for item, step in zip(plan["selected"], selected, strict=True):
        if step.fingerprint() != item["request"]["fingerprint"]:
            raise ValueError("Original training fingerprint changed")
    return selected


def build_plan() -> dict:
    training_plan = repair.read_json(str(SURVEY))
    preserved = repair.read_json(str(PRESERVED))
    entries = {row["run_name"]: row for row in preserved["checkpoints"]}
    selected = [
        {"request": row, "preserved": entries[row["run_name"]]}
        for row in training_plan["runs"]
        if row["domain"] == "finemath_3plus" and row["arm"] == "target" and row["percent"] in PERCENTS
    ]
    plan = {
        "purpose": "User-authorized completion of the four paused FineMath targets; no new proxy or seed",
        "authorization": (
            "actually let's finish training the FineMath sweep then, including the target checkpoints, "
            "and do the math evals on all of them"
        ),
        "training_plan": training_plan,
        "selected": selected,
        "wrapper_sha256": file_sha256(Path(__file__)),
        "source_parent": OLD_PARENT,
        "max_concurrent": 4,
    }
    plan["recovery_sha256"] = canonical_sha256(plan)
    validate_plan(plan)
    frozen_steps(plan)
    return plan


def preflight(plan: dict) -> dict:
    validate_plan(plan)
    steps = frozen_steps(plan)
    proxy_requests = [row for row in plan["training_plan"]["runs"] if row["arm"] == "matched"]
    with ThreadPoolExecutor(max_workers=8) as pool:
        proxy_evidence = list(pool.map(lambda row: repair.training_evidence(row, plan["training_plan"]), proxy_requests))
    if len(proxy_evidence) != 14:
        raise ValueError("Expected all fourteen completed Wikipedia/FineMath proxies")
    results = []
    for item, step in zip(plan["selected"], steps, strict=True):
        request = item["request"]
        verify_preserved(item["preserved"])
        status = StatusFile(request["output_path"], worker_id="finemath-target-completion-check")
        if status.active_lock_holder() is not None:
            raise ValueError(f"Original output has an active lease: {request['run_name']}")
        record = read_record(request["output_path"])
        if record is not None and record.fingerprint != request["fingerprint"]:
            raise ValueError("Original artifact fingerprint changed")
        recipe = materialized_config(step, survey.experiment.PREFIX)
        if recipe.training.pod.train_config.trainer.allow_partial_checkpoint:
            raise ValueError("Full optimizer and trainer-state restoration is required")
        resumed_step = replace(step, run=partial(dispatch_training, plan=plan, item=item))
        if resumed_step.fingerprint() != request["fingerprint"]:
            raise ValueError("Resume dispatch changed the frozen training identity")
        for dependency in step.deps:
            if dependency.override_path is None:
                dependency_status = StatusFile(
                    dependency.path(survey.experiment.PREFIX), worker_id="finemath-dependency-check"
                )
                if dependency_status.status != STATUS_SUCCESS:
                    raise ValueError(f"Recovery requires already-complete data: {dependency.name}")
        results.append(
            {
                "run_name": request["run_name"],
                "fingerprint": step.fingerprint(),
                "resumed_fingerprint": resumed_step.fingerprint(),
                "status": status.status,
                "resume_step": item["preserved"]["step"],
                "allow_partial_checkpoint": False,
                "resources": str(recipe.training.pod.resources),
            }
        )
    return {"recovery_sha256": plan["recovery_sha256"], "proxies_verified": len(proxy_evidence), "selected": results}


def resume_training(recipe: survey.DomainTrainingRecipe, plan: dict, item: dict) -> None:
    """Require full-state restoration, preferring later checkpoints on retries."""
    survey.experiment.require_central1()
    validate_plan(plan)
    request = item["request"]
    if recipe.training.pod.output_path != request["output_path"]:
        raise ValueError("Recovery output differs from the frozen request")
    if recipe.training.pod.train_config.trainer.allow_partial_checkpoint:
        raise ValueError("Partial-state loading is forbidden for continuation")
    verify_preserved(item["preserved"])
    roots = [
        request["output_path"] + "/checkpoints",
        temporary_checkpoint_base_path(request["output_path"]),
        item["preserved"]["checkpoint_uri"],
    ]
    latest = latest_checkpoint_path(*roots)
    metadata = repair.read_json(latest + "/metadata.json")
    if not item["preserved"]["step"] <= metadata["step"] < request["total_steps"]:
        raise ValueError("Recovery checkpoint lies outside the approved continuation")
    train_config = cast(TrainLmConfig, recipe.training.pod.train_config)
    trainer = replace(train_config.trainer, load_checkpoint=True, load_checkpoint_path=roots)
    pod = replace(recipe.training.pod, train_config=replace(train_config, trainer=trainer))
    resumed = replace(recipe, training=replace(recipe.training, pod=pod))
    repair.write_json(
        request["output_path"] + "/target_completion_resume.json",
        {
            "recovery_sha256": plan["recovery_sha256"],
            "training_fingerprint": request["fingerprint"],
            "checkpoint": latest,
            "metadata": metadata,
            "load_checkpoint": True,
            "changes": ["require full-state restore", "include verified preserved checkpoint in search roots"],
        },
    )
    survey.verified_training(resumed)


def dispatch_training(recipe: survey.DomainTrainingRecipe, *, plan: dict, item: dict) -> None:
    remote(
        resume_training,
        name=item["request"]["run_name"],
        resources=recipe.training.pod.resources,
        env_vars={"MARIN_PREFIX": survey.experiment.PREFIX},
    )(recipe, plan, item)


def submit(plan: dict) -> None:
    survey.experiment.require_central1()
    context = get_iris_ctx()
    if context is None or context.client is None:
        raise ValueError("Recovery requires an Iris coordinator")
    old_tree = context.client.list_jobs(prefix=plan["source_parent"])
    if not old_tree or any(job.state not in TERMINAL_JOB_STATES for job in old_tree):
        raise ValueError("Original training tree must be terminal before recovery")
    audit = preflight(plan)
    root = f"{survey.ROOT}/{plan['training_plan']['plan_sha256']}/target_completion/{plan['recovery_sha256']}"
    repair.write_json(root + "/plan.json", plan)
    repair.write_json(root + "/preflight.json", audit)
    steps = tuple(
        replace(step, run=partial(dispatch_training, plan=plan, item=item))
        for item, step in zip(plan["selected"], frozen_steps(plan), strict=True)
    )
    pending = survey.original.pending_training_steps(steps, marin_prefix=survey.experiment.PREFIX)
    if pending:
        run(*pending, max_concurrent=4, force_run_failed=True)
    results = {
        item["request"]["run_name"]: repair.training_evidence(item["request"], plan["training_plan"])
        for item in plan["selected"]
    }
    repair.write_json(root + "/results.json", {"recovery_sha256": plan["recovery_sha256"], "results": results})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=DIRECTORY / "plan.json")
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--build", action="store_true")
    action.add_argument("--submit", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    if args.build:
        plan = build_plan()
        audit = preflight(plan)
        args.plan.parent.mkdir(parents=True, exist_ok=True)
        args.plan.write_text(json.dumps(plan, indent=2) + "\n")
        (args.plan.parent / "preflight.json").write_text(json.dumps(audit, indent=2) + "\n")
        print(json.dumps(audit))
    else:
        submit(repair.read_json(str(args.plan)))


if __name__ == "__main__":
    main()
