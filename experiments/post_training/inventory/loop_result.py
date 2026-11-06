# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Persist bounded terminal evidence independently of dashboard publication."""

import hashlib
import json
import logging
import math
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import yaml
from iris.cluster.types import JobName
from marin.rl.skyrl import TRACKING_FILENAME, SkyRLRun, SkyRLRunConfig, run_skyrl
from rigging.filesystem.storage_path import StoragePath
from rigging.provenance import Provenance

from experiments.post_training import snowball_rl_presets as presets

FIXED_PANEL_DECLARED_SHA256 = "33af492d1d1cbe0602effd0b0f5154c87d6e733b8d9e8b48178be7030f3e0426"
MAX_EVIDENCE_BYTES = 2 * 1024 * 1024
RESULT_FILENAME = "loop-result.json"
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SnowballAttempt:
    config_fingerprint: str
    layout: str


@dataclass(frozen=True)
class LoopRunConfig:
    skyrl: SkyRLRunConfig
    label: str
    expected_steps: int
    snowball: SnowballAttempt | None = None


def _bounded_text(uri: StoragePath, limit: int = MAX_EVIDENCE_BYTES) -> str | None:
    try:
        with uri.open("rb") as source:
            content = source.read(limit + 1)
    except FileNotFoundError:
        return None
    if len(content) > limit:
        raise ValueError("Loop evidence exceeds its bounded read budget")
    return content.decode("utf-8")


def _json_observation(uri: StoragePath, issues: list[str]) -> dict[str, Any] | None:
    try:
        text = _bounded_text(uri)
        if text is None:
            return None
        result = json.loads(text)
        if not isinstance(result, dict):
            raise ValueError("Loop observation must be an object")
        return result
    except (OSError, ValueError) as error:
        issues.append(f"Could not validate evidence object {uri}: {type(error).__name__}.")
        return None


def _evaluation_result(metrics: dict[str, Any] | None, issues: list[str]) -> dict[str, Any] | None:
    if metrics is None:
        return None
    if not isinstance(metrics, dict):
        issues.append("Evaluation metrics are not an object.")
        return None
    attempted = metrics.get("eval/all/num_attempted")
    scored = metrics.get("eval/all/num_scored")
    reward = metrics.get("eval/all/avg_verifier_score")
    if not all(type(value) in (int, float) and math.isfinite(value) for value in (attempted, scored)):
        issues.append("Evaluation coverage is missing or nonfinite.")
        return None
    if attempted < 0 or scored < 0 or scored > attempted or int(attempted) != attempted or int(scored) != scored:
        issues.append("Evaluation coverage counts are invalid.")
        return None
    if type(reward) not in (int, float) or not math.isfinite(reward):
        issues.append(f"Evaluation reward unavailable; attempted={attempted}, scored={scored}. No zero was substituted.")
        return None
    return {"reward": reward, "attempted": int(attempted), "scored": int(scored)}


def _write_result(
    config: LoopRunConfig,
    started: datetime,
    *,
    status: str,
    optimizer_steps: int | None,
    values: dict[str, Any],
    destination: StoragePath | None = None,
) -> dict[str, Any]:
    skyrl = config.skyrl
    now = datetime.now(UTC)
    provenance = Provenance.capture()
    config_hash = hashlib.sha256(skyrl.launch_config_yaml.encode()).hexdigest()
    source_revision = f"launcher base={provenance.base_commit or 'unknown'} tree={provenance.tree_hash or 'unknown'}"
    source_revision += f"; launch SHA256={config_hash}"
    result = {
        "run_id": skyrl.run_id,
        "attempt_id": skyrl.attempt_id,
        "label": config.label,
        "run_date": started.isoformat().replace("+00:00", "Z"),
        "date_basis": "launch_started",
        "started_at": started.isoformat().replace("+00:00", "Z"),
        "completed_at": now.isoformat().replace("+00:00", "Z"),
        "elapsed_seconds": (now - started).total_seconds(),
        "timing_scope": "loop_launcher_to_terminal",
        "status": status,
        "optimizer_steps": optimizer_steps,
        "source_revision": source_revision,
        **values,
    }
    destination = destination or StoragePath(skyrl.output.export_root) / RESULT_FILENAME
    destination.parent.mkdirs()
    destination.write_text(json.dumps(result, allow_nan=False, indent=2))
    return result


def _persist_result(config: LoopRunConfig, started: datetime, native_succeeded: bool) -> dict[str, Any]:
    skyrl = config.skyrl
    launch = yaml.safe_load(skyrl.launch_config_yaml)
    recipe = launch["skyrl"]
    expected_count = recipe["data"]["expected_eval_rows"] * recipe["generator"].get("eval_n_samples_per_prompt", 1)
    export = StoragePath(skyrl.output.export_root)
    issues = []
    observations = _json_observation(export / "loop-evidence.json", issues)
    if observations is not None:
        try:
            observation_start = datetime.fromisoformat(observations["started_at"])
            if (
                observations.get("schema_version") != 1
                or observations.get("expected_steps") != config.expected_steps
                or observation_start.utcoffset() is None
                or observation_start < started
                or observation_start > datetime.now(UTC)
                or not isinstance(observations.get("steps"), list)
                or not isinstance(observations.get("evaluations"), dict)
            ):
                raise ValueError("Observation does not match this launch interval")
        except (KeyError, TypeError, ValueError):
            issues.append("Loop observations are malformed, stale, or do not match the expected training count.")
            observations = None
    steps = [] if observations is None else observations["steps"]
    valid_steps = []
    for item in steps:
        if not isinstance(item, dict) or type(item.get("step")) is not int or not isinstance(item.get("metrics"), dict):
            issues.append("Malformed optimizer-window observation.")
            continue
        metrics = item["metrics"]
        values = [metrics.get(key) for key in ("policy/policy_loss", "policy/raw_grad_norm")]
        if (
            metrics.get("policy/policy_update_steps") == 1
            and metrics.get("policy/skipped_steps") == 0
            and all(type(value) in (int, float) and math.isfinite(value) for value in values)
        ):
            valid_steps.append(item["step"])
    steps_complete = len(steps) == config.expected_steps and valid_steps == list(range(1, config.expected_steps + 1))
    if not steps_complete:
        issues.append("Retained observations do not establish every expected finite applied optimizer update.")
    checkpoint = None
    try:
        marker = _bounded_text(StoragePath(skyrl.output.checkpoint_root) / "latest_ckpt_global_step.txt", 1024)
        checkpoint = None if marker is None else int(marker.strip())
    except (OSError, ValueError) as error:
        issues.append(f"Checkpoint marker could not be validated: {type(error).__name__}.")
    if checkpoint != config.expected_steps:
        issues.append("Final checkpoint marker does not match the expected update count.")
    evaluations = {} if observations is None else observations.get("evaluations", {})
    phases = []
    for step in (0, config.expected_steps):
        metrics = evaluations.get(str(step))
        if metrics is None and observations is not None:
            metrics = _json_observation(
                export / "dumped_evals" / f"global_step_{step}_evals" / "aggregated_results.jsonl", issues
            )
        phases.append(_evaluation_result(metrics, issues))
    baseline, final = phases
    if baseline is None and final is not None:
        issues.append(f"Final observation without a valid baseline: {json.dumps(final, allow_nan=False)}.")
        final = None
    full_coverage = all(
        phase is not None and phase["attempted"] == phase["scored"] == expected_count for phase in (baseline, final)
    )
    if not full_coverage:
        issues.append("Baseline and final scoring must both cover the complete declared evaluation cohort.")
    if not native_succeeded:
        issues.append("Native training launcher failed; available observations do not override that failure.")
    inputs = launch["inputs"]
    status = (
        "succeeded"
        if native_succeeded and steps_complete and checkpoint == config.expected_steps and full_coverage and not issues
        else "failed"
    )
    return _write_result(
        config,
        started,
        status=status,
        optimizer_steps=len(valid_steps),
        values={
            "data_identity": "; ".join(item["identity"] for item in inputs["train_data"]),
            "initializer_identity": skyrl.model.identity,
            "evaluation": {
                "name": "snowball-fixed-development",
                "version": "; ".join(item["identity"] for item in inputs["validation_data"]),
                "task_count": expected_count,
                "task_set_sha256": None,
                "protocol_sha256": None,
            },
            "baseline": baseline,
            "final": final,
            "artifact_uri": skyrl.output.terminal_manifest_uri,
            "gpu_hours": None,
            "wandb_url": None if observations is None else observations.get("wandb_url"),
            "evidence": [
                *issues,
                "Launcher configuration and input identities are declarations, not full loaded-runtime attestations; "
                "runtime task-set and protocol identity remain unknown/unranked.",
                f"Declared reference fixed100 task-content digest: {FIXED_PANEL_DECLARED_SHA256}; "
                "this record does not attest actual runtime membership.",
                "Elapsed time includes the blocking launcher; allocated GPU-hours "
                "and cancelled generation work are unknown.",
                "Optimizer evidence uses callback metrics, native terminal success and checkpoint marker; "
                "no independent per-token accounting is claimed.",
            ],
        },
    )


def _persist_snowball_result(
    config: LoopRunConfig,
    started: datetime,
    native: SkyRLRun | None,
    issues: list[str],
) -> dict[str, Any]:
    skyrl = config.skyrl
    metadata = config.snowball
    assert metadata is not None
    export = StoragePath(skyrl.output.export_root)
    tracking = _json_observation(export / TRACKING_FILENAME, issues)
    wandb_url = (
        tracking.get("wandb_url") if tracking is not None and tracking.get("attempt_id") == skyrl.attempt_id else None
    )
    if wandb_url is None:
        issues.append("W&B tracking is not attributable to this native attempt.")
    attempt = _json_observation(StoragePath(skyrl.output.attempts_root) / f"{skyrl.attempt_id}.json", issues)
    job = native.iris_job_id if native is not None else None
    if attempt is not None:
        recorded = attempt.get("config", {}).get("run", {})
        if (recorded.get("id"), recorded.get("attempt_id")) == (skyrl.run_id, skyrl.attempt_id):
            job = attempt.get("result", {}).get("iris_job_id") or job
        else:
            issues.append("Native attempt receipt identity does not match this launch.")
    applied = []
    for path in sorted((export / "metrics/step-*.json").glob(), key=str):
        row = _json_observation(path, issues)
        if row is None or row.get("attempt_id") != skyrl.attempt_id:
            continue
        step = row.get("step", 0)
        if (
            not 0 < step <= config.expected_steps
            or row.get("trainer/global_step") != step
            or "timing/step_wall/unaccounted" not in row
        ):
            continue
        count = row.get("policy/policy_update_steps")
        values = [row.get(key) for key in ("policy/policy_loss", "policy/raw_grad_norm")]
        if (
            type(count) in (int, float)
            and math.isfinite(count)
            and count >= 0
            and int(count) == count
            and all(type(value) in (int, float) and math.isfinite(value) for value in values)
        ):
            applied.append(int(count))
    if len(applied) != config.expected_steps:
        issues.append("Applied optimizer-window evidence is incomplete for this attempt.")
    interval = {"elapsed_seconds": 0.0, "timing_scope": "native_job_start_to_finish"}
    try:
        with presets.iris_client() as client:
            status = client.job_status(JobName.from_wire(job))
        if status.started_at is None or status.finished_at is None:
            raise ValueError(f"Native timing is unavailable for {job}")
        start, finish = status.started_at.epoch_seconds(), status.finished_at.epoch_seconds()
        if finish < start:
            raise ValueError(f"Native completion precedes its start: {job}")
        interval.update(
            run_date=datetime.fromtimestamp(start, UTC).isoformat(),
            started_at=datetime.fromtimestamp(start, UTC).isoformat(),
            completed_at=datetime.fromtimestamp(finish, UTC).isoformat(),
            elapsed_seconds=finish - start,
        )
    except Exception as error:
        issues.append(f"Native job timing: {error}")
    launch = yaml.safe_load(skyrl.launch_config_yaml)
    return _write_result(
        config,
        started,
        status="succeeded" if native is not None else "failed",
        optimizer_steps=sum(applied),
        destination=export / "attempts" / skyrl.attempt_id / RESULT_FILENAME,
        values={
            "config_fingerprint": metadata.config_fingerprint,
            "layout": metadata.layout,
            "iris_job_id": job,
            "wandb_url": wandb_url,
            "data_identity": "; ".join(item["identity"] for item in launch["inputs"]["train_data"]),
            "initializer_identity": skyrl.model.identity,
            **interval,
            "evidence": [*issues, "Panel phases and quality verdict are evaluated independently of native execution."],
        },
    )


def run_loop(config: LoopRunConfig) -> SkyRLRun:
    """Run native training and persist its attempt record."""
    started = datetime.now(UTC)
    issues = []

    def persist(native: SkyRLRun | None) -> dict[str, Any]:
        if config.snowball is not None:
            return _persist_snowball_result(config, started, native, issues)
        return _persist_result(config, started, native_succeeded=native is not None)

    try:
        run = run_skyrl(config.skyrl)
    except Exception as training_error:
        try:
            persist(None)
        except Exception as recording_error:
            raise ExceptionGroup(
                "Training and terminal-summary persistence failed", [training_error, recording_error]
            ) from None
        raise
    try:
        summary = persist(run)
    except Exception:
        if config.snowball is None:
            raise
        logger.warning("Native run completed; attempt record write failed", exc_info=True)
        return run
    if config.snowball is None and summary["status"] != "succeeded":
        raise RuntimeError("Training completed but loop evidence is incomplete; see exports/loop-result.json")
    return run
