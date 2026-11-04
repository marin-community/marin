# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Full-run BFCL admission and an independent run-cluster bootstrap.

Recorded method equality is not worker-policy attestation. This utility starts a
new seeded analysis; historical preregistered continuation streams stay separate.
"""

import copy
import hashlib
import json
import math
import random
import statistics
from dataclasses import dataclass
from typing import Any

from marin.evaluation.records import RunStatus
from marin.inference.config import resolve_tokenizer_revision


@dataclass(frozen=True)
class AdmittedRun:
    logical_run: str
    run_id: str
    method: dict[str, Any]
    rewards: tuple[float | None, ...]
    task_ids: tuple[str, ...]
    errors: dict[str, Any]
    status: str
    target: dict[str, Any]
    n_observed: int
    native_coverage: dict[str, Any]
    coverage_mismatch: bool
    native_error: Any

    @property
    def complete(self) -> bool:
        return bool(self.rewards) and all(value is not None for value in self.rewards) and not self.coverage_mismatch


def recorded_method(record: dict[str, Any]) -> dict[str, Any]:
    """Ignore only target identity/weight references, after resolving tokenizer fallbacks.

    Compare the whole eval definition and eval_runtime, all agent/serve/parser/
    generation settings, and tokenizer revisions. These are recorded declarations,
    not a claim that every worker's effective runtime was independently observed.
    """
    config = copy.deepcopy(record["model"]["config"])
    config["tokenizer_revision"] = resolve_tokenizer_revision(
        config.get("revision"), config.get("tokenizer"), config.get("tokenizer_revision")
    )
    config["tokenizer"] = config.get("tokenizer") or config["location"]
    for name in ("name", "location", "identity", "revision"):
        config.pop(name, None)
    evaluation = record["eval"]
    if evaluation["mechanism"] != "harbor" or "bfcl" not in evaluation["harbor"]["dataset"].lower():
        raise ValueError("Expected a native Harbor BFCL record")
    runtime = record["provenance"]["eval_runtime"]
    if not runtime:
        raise ValueError("Native record lacks evaluator runtime provenance")
    return {
        "eval": copy.deepcopy(evaluation),
        "backend": record["model"]["backend"],
        "model_method": config,
        "eval_runtime": copy.deepcopy(runtime),
        "hardware": copy.deepcopy(record.get("hardware")),
        "serving": copy.deepcopy(record.get("serving")),
    }


def admit_native_run(
    record: dict[str, Any], samples: list[dict[str, Any]], *, logical_run: str, task_ids: tuple[str, ...]
) -> AdmittedRun:
    """Keep unscored/missing grades NULL; retain scored agent-error zeros."""
    status = record["status"]
    if status not in {value.value for value in RunStatus}:
        raise ValueError("Native evaluation has no terminal status")
    if not task_ids or len(set(task_ids)) != len(task_ids):
        raise ValueError("Expected a nonempty fixed task set")
    native_tasks = record["eval"]["tasks"]
    if len(native_tasks) != 1 or native_tasks[0]["benchmark"]["n_benchmark"] != len(task_ids):
        raise ValueError("Frozen task set differs from native benchmark size")
    rewards: dict[str, float | None] = dict.fromkeys(task_ids)
    errors = {}
    seen = set()
    expected_task = native_tasks[0].get("name", record["eval"]["harbor"]["dataset"])
    for sample in samples:
        if sample["task"] != expected_task:
            raise ValueError("Native sample belongs to a different benchmark")
        name = sample["doc_id"]
        if name not in rewards or name in seen:
            raise ValueError(f"Unexpected or duplicate native task: {name}")
        seen.add(name)
        grading = sample.get("grading") or {}
        detail = grading.get("detail") or {}
        if isinstance(detail, str):
            detail = json.loads(detail)
        error = detail.get("error")
        if error is not None:
            errors[name] = error
        if detail.get("scored") is not True:
            continue
        metrics = [value for key, value in sample["metrics"] if key == "reward"]
        if len(metrics) != 1:
            raise ValueError(f"Native task must have exactly one reward: {name}")
        reward = metrics[0]
        if (
            type(reward) not in (int, float)
            or not math.isfinite(reward)
            or reward not in (0, 1)
            or grading.get("method") != "harbor:verifier"
            or grading.get("filter") not in (None, "")
            or sample.get("filter") not in (None, "")
            or grading.get("passed") is not bool(reward)
            or grading.get("metric") != "reward"
            or grading.get("score") != reward
            or detail.get("rewards", {}).get("reward") != reward
        ):
            raise ValueError(f"Scored native grading/reward mismatch: {name}")
        rewards[name] = float(reward)
    coverage = copy.deepcopy(record.get("coverage", {}))
    native_task = native_tasks[0].get("name", record["eval"]["harbor"]["dataset"])
    declared_scored = coverage.get(native_task, {}).get("n_scored")
    observed_scored = sum(value is not None for value in rewards.values())
    return AdmittedRun(
        logical_run=logical_run,
        run_id=record["run_id"],
        method=recorded_method(record),
        rewards=tuple(rewards[name] for name in task_ids),
        task_ids=task_ids,
        errors=errors,
        status=status,
        target={key: record["model"]["config"].get(key) for key in ("location", "revision", "identity")},
        n_observed=len(seen),
        native_error=copy.deepcopy(record.get("error")),
        native_coverage=coverage,
        coverage_mismatch=declared_scored is not None and declared_scored != observed_scored,
    )


def compare_run_clusters(
    arms: dict[str, list[AdmittedRun]], *, reference: str, seed: int = 297, iterations: int = 10000
) -> dict[str, Any]:
    """Bootstrap whole independent run means, never tasks or matched run numbers.

    All complete attempts are retained. Two complete attempts for one logical run
    are rejected rather than selecting a preferred score. At least five complete
    logical runs per arm are required for a primary mean, SD or interval.
    """
    if reference not in arms or iterations < 100:
        raise ValueError("Reference arm and at least 100 bootstrap iterations are required")
    all_runs = [run for runs in arms.values() for run in runs]
    if len({run.run_id for run in all_runs}) != len(all_runs):
        raise ValueError("Native run IDs cannot be reused across logical runs or arms")
    if all_runs:
        expected = all_runs[0].method
        tasks = all_runs[0].task_ids
        for run in all_runs:
            if run.method != expected or run.task_ids != tasks:
                raise ValueError("Recorded BFCL method/task-set mismatch")
    else:
        expected = None
    result: dict[str, Any] = {
        "reference": reference,
        "seed": seed,
        "iterations": iterations,
        "method": expected,
        "method_sha256": (
            None
            if expected is None
            else hashlib.sha256(json.dumps(expected, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        ),
        "sampling_unit": "independent complete logical run",
        "minimum_complete_runs": 5,
        "arms": {},
        "contrasts": {},
    }
    means = {}
    for arm, runs in sorted(arms.items()):
        if runs and any(run.target != runs[0].target for run in runs):
            raise ValueError(f"Different target checkpoints within arm {arm}")
        logical = {}
        for run in sorted(runs, key=lambda r: r.run_id):
            slot = logical.setdefault(run.logical_run, {"attempts": [], "complete": []})
            slot["attempts"].append(
                {
                    "run_id": run.run_id,
                    "native_status": run.status,
                    "rewards": run.rewards,
                    "errors": run.errors,
                    "complete": run.complete,
                    "target": run.target,
                    "n_observed": run.n_observed,
                    "n_scored": sum(value is not None for value in run.rewards),
                    "native_coverage": run.native_coverage,
                    "coverage_mismatch": run.coverage_mismatch,
                    "native_error": run.native_error,
                }
            )
            if run.complete:
                slot["complete"].append(statistics.fmean(value for value in run.rewards if value is not None))
        if any(len(slot["complete"]) > 1 for slot in logical.values()):
            raise ValueError(f"Multiple complete attempts for a logical run in {arm}; no score selection")
        values = [slot["complete"][0] for slot in logical.values() if slot["complete"]]
        if len(values) >= 5:
            means[arm] = values
        result["arms"][arm] = {
            "n_complete": len(values),
            "n_logical": len(logical),
            "n_physical": len(runs),
            "logical_runs": logical,
            "mean": statistics.fmean(values) if len(values) >= 5 else None,
            "sample_sd": statistics.stdev(values) if len(values) >= 5 else None,
            "ci95": None,
            "primary_available": len(values) >= 5,
        }
    rng = random.Random(seed)
    draws = {arm: [] for arm in means}
    for _ in range(iterations):
        for arm, values in sorted(means.items()):
            draws[arm].append(statistics.fmean(rng.choice(values) for _ in values))

    def interval(values):
        ordered = sorted(values)
        return [ordered[int(0.025 * iterations)], ordered[int(0.975 * iterations)]]

    for arm, values in draws.items():
        result["arms"][arm]["ci95"] = interval(values)
    for arm in sorted(arms):
        if arm == reference:
            continue
        available = arm in means and reference in means
        result["contrasts"][arm] = {
            "reference": reference,
            "delta": result["arms"][arm]["mean"] - result["arms"][reference]["mean"] if available else None,
            "ci95": interval([a - b for a, b in zip(draws[arm], draws[reference], strict=True)]) if available else None,
        }
    return result
