# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Launch a midtraining job: build a Fray request, submit, write artifacts.

Operators call these as library functions from a Python launcher script.
There is no CLI — the launcher script is the discoverable surface
(matches the ``downstream_scaling`` pattern).
"""

import datetime as _dt
import json
import logging
import os
from dataclasses import dataclass, field

from marin.midtraining.identity import RunIdentity, attempt_group_manifest_uri, expected_run_env
from marin.midtraining.levanter_config import render_train_lm_yaml
from marin.midtraining.preflight import CooldownStageRecord, PreflightReport
from marin.midtraining.schema import (
    SCHEMA_VERSION,
    RunManifestRow,
    TokenizerRecord,
    write_run_manifest,
)
from marin.midtraining.schema import (
    CooldownStageRecord as CooldownStageRecordRow,
)
from marin.midtraining.spec import ComputeProfile, ResolvedMidtrainSpec

logger = logging.getLogger(__name__)

LEVANTER_TRAIN_LM_MODULE = "levanter.main.train_lm"


# ---------------------------------------------------------------------------
# Manifest construction
# ---------------------------------------------------------------------------


def build_manifest_row(
    resolved: ResolvedMidtrainSpec,
    preflight: PreflightReport,
    *,
    stage_record: CooldownStageRecord | None = None,
    status: str = "planned",
) -> RunManifestRow:
    """Materialize a :class:`RunManifestRow` from the resolved spec + preflight."""
    spec = resolved.spec
    run = spec.run
    tokenizer: TokenizerRecord = {
        "key": spec.tokenizer.key,
        "hf_repo": spec.tokenizer.hf_repo,
        "revision": spec.tokenizer.revision,
        "bos_token_id": spec.tokenizer.bos_token_id,
        "eos_token_id": spec.tokenizer.eos_token_id,
        "vocab_size": spec.tokenizer.vocab_size,
        "fingerprint": spec.tokenizer.fingerprint,
    }
    cooldown_row: CooldownStageRecordRow | None = _stage_record_row(stage_record) if stage_record is not None else None
    data_manifest_uri = spec.data_manifest_uri or f"legacy:{spec.data_section_provenance or 'unknown'}"
    data_manifest_fingerprint = (
        resolved.data_manifest.fingerprint()
        if resolved.data_manifest is not None
        else f"legacy:{spec.data_section_provenance or 'unknown'}"
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "written_at": _dt.datetime.now(tz=_dt.UTC).isoformat(),
        "logical_cell_id": run.logical_cell_id,
        "attempt": run.attempt,
        "run_id": run.run_id,
        "mode": spec.mode.kind,
        "output_path": run.output_path,
        "wandb_project": run.wandb_project,
        "wandb_entity": run.wandb_entity,
        "base_flops_key": spec.base.flops_key,
        "tpu_type": spec.compute.tpu_type,
        "train_batch_size": spec.compute.batch_size,
        "per_device_parallelism": spec.compute.per_device_parallelism,
        "max_retries_failure": spec.compute.max_retries_failure,
        "max_task_failures": spec.compute.max_task_failures,
        "data_manifest_uri": data_manifest_uri,
        "data_manifest_fingerprint": data_manifest_fingerprint,
        "tokenizer": tokenizer,
        "seq_len": spec.seq_len,
        "num_train_steps": resolved.num_train_steps,
        "actual_tokens": resolved.actual_tokens,
        "train_config_uri": run.train_config_uri,
        "permanent_checkpoints_uri": preflight.permanent_checkpoints_uri,
        "temp_checkpoints_uri": preflight.temp_checkpoints_uri,
        "init_checkpoint_uri": preflight.init_checkpoint_uri,
        "staged_checkpoint_uri": preflight.staged_checkpoint_uri,
        "cooldown_stage_record": cooldown_row,
        "preflight_failures": list(preflight.failures),
        "preflight_warnings": list(preflight.warnings),
        "preflight_notes": list(preflight.notes),
        "extra_tags": list(spec.extra_tags),
        "status": status,
    }


def _stage_record_row(record: CooldownStageRecord) -> CooldownStageRecordRow:
    return {
        "source": record.source,
        "destination": record.destination,
        "cross_region_copy": record.cross_region_copy,
        "bytes_copied": record.bytes_copied,
        "budget_gb": record.budget_gb,
        "reason": record.reason,
    }


def write_manifest(row: RunManifestRow, *, output_path: str) -> None:
    """Write a manifest to ``<output_path>/midtrain_manifest.json``."""
    write_run_manifest(row, f"{output_path.rstrip('/')}/midtrain_manifest.json")


def write_train_config(resolved: ResolvedMidtrainSpec) -> None:
    """Write the rendered Levanter YAML next to the run manifest."""
    import fsspec

    yaml_text = render_train_lm_yaml(resolved)
    with fsspec.open(resolved.spec.run.train_config_uri, "w", encoding="utf-8") as f:
        f.write(yaml_text)


def append_to_attempt_group(row: RunManifestRow, *, region: str) -> None:
    """Append/replace this attempt's record in the attempt-group manifest."""
    import fsspec

    group_uri = attempt_group_manifest_uri(logical_cell_id=row["logical_cell_id"], region=region)
    record = {
        "attempt": row["attempt"],
        "run_id": row["run_id"],
        "output_path": row["output_path"],
        "manifest_uri": f"{row['output_path'].rstrip('/')}/midtrain_manifest.json",
        "status": row["status"],
        "written_at": row["written_at"],
        "mode": row["mode"],
    }
    existing: dict = {"logical_cell_id": row["logical_cell_id"], "attempts": []}
    try:
        with fsspec.open(group_uri, "r", encoding="utf-8") as f:
            existing = json.loads(f.read())
    except FileNotFoundError:
        pass

    attempts = [a for a in existing.get("attempts", []) if a.get("run_id") != row["run_id"]]
    attempts.append(record)
    attempts.sort(key=lambda a: a.get("attempt", 0))
    existing["attempts"] = attempts
    existing["logical_cell_id"] = row["logical_cell_id"]

    with fsspec.open(group_uri, "w", encoding="utf-8") as f:
        f.write(json.dumps(existing, indent=2, sort_keys=True))


# ---------------------------------------------------------------------------
# Fray submission
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LaunchRequest:
    """Materialized request that one ``submit_launch`` call submits."""

    job_name: str
    train_config_uri: str
    resources_kwargs: dict[str, object]
    env: dict[str, str]
    max_retries_failure: int
    max_task_failures: int
    extras: tuple[str, ...] = ("tpu",)

    def command_args(self) -> tuple[str, ...]:
        return ("-m", LEVANTER_TRAIN_LM_MODULE, "--config", self.train_config_uri)


@dataclass(frozen=True)
class LaunchResult:
    request: LaunchRequest
    job: object = field(repr=False)

    def wait(self, *, raise_on_failure: bool = True) -> object:
        return self.job.wait(raise_on_failure=raise_on_failure)


def build_launch_request(
    resolved: ResolvedMidtrainSpec,
    *,
    train_config_uri: str | None = None,
    job_name: str | None = None,
    extra_env: dict[str, str] | None = None,
) -> LaunchRequest:
    spec = resolved.spec
    run = spec.run
    train_config_uri = train_config_uri or run.train_config_uri
    job_name = job_name or _default_job_name(run)
    env = _launch_env(run, extra_env)
    resources_kwargs = _resources_kwargs(spec.compute, run)
    return LaunchRequest(
        job_name=job_name,
        train_config_uri=train_config_uri,
        resources_kwargs=resources_kwargs,
        env=env,
        max_retries_failure=spec.compute.max_retries_failure,
        max_task_failures=spec.compute.max_task_failures,
    )


def submit_launch(request: LaunchRequest, *, client: object | None = None) -> LaunchResult:
    """Submit the launch request to Iris/Fray and return a handle."""
    from fray import Entrypoint, JobRequest, ResourceConfig, create_environment, current_client

    resources = ResourceConfig.with_tpu(
        request.resources_kwargs["tpu_type"],
        regions=request.resources_kwargs.get("regions") or None,
        ram=request.resources_kwargs["ram"],
    )
    environment = create_environment(env_vars=request.env, extras=list(request.extras))
    job_request = JobRequest(
        name=request.job_name,
        entrypoint=Entrypoint.from_binary("python", list(request.command_args())),
        resources=resources,
        environment=environment,
        max_retries_failure=request.max_retries_failure,
        max_task_failures=request.max_task_failures,
    )
    client = client or current_client()
    job = client.submit(job_request)
    logger.info("Submitted midtraining job %s", request.job_name)
    return LaunchResult(request=request, job=job)


def _default_job_name(run: RunIdentity) -> str:
    return f"midtrain-{run.run_id}"[:96]


def _launch_env(run: RunIdentity, extra_env: dict[str, str] | None) -> dict[str, str]:
    env: dict[str, str] = {}
    api_key = os.environ.get("WANDB_API_KEY")
    if not api_key:
        raise ValueError("WANDB_API_KEY is not set; export it before launching or set WANDB_MODE=disabled explicitly.")
    env["WANDB_API_KEY"] = api_key
    env.update(expected_run_env(run))
    if extra_env:
        for key, value in extra_env.items():
            if key in env and env[key] != value:
                raise ValueError(f"extra_env[{key!r}] conflicts with launcher-derived value")
            env[key] = value
    return env


def _resources_kwargs(compute: ComputeProfile, run: RunIdentity) -> dict[str, object]:
    regions = compute.regions or (run.output_region,)
    if regions != (run.output_region,):
        raise ValueError(f"ComputeProfile.regions={regions!r} must equal ({run.output_region!r},) for a real launch.")
    return {"tpu_type": compute.tpu_type, "regions": list(regions), "ram": compute.ram}
