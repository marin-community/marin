# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Operating layer for Atlas Difficulty measurement rounds.

One entry point replaces the per-experiment launch, poll and export scripts:

    uv run --no-sync python -m experiments.post_training.task_curation.difficulty_rounds launch     [...]
    uv run --no-sync python -m experiments.post_training.task_curation.difficulty_rounds poll       [...]
    uv run --no-sync python -m experiments.post_training.task_curation.difficulty_rounds seal       [...]
    uv run --no-sync python -m experiments.post_training.task_curation.difficulty_rounds coverage   [...]
    uv run --no-sync python -m experiments.post_training.task_curation.difficulty_rounds export-snapshot [...]
    uv run --no-sync python -m experiments.post_training.task_curation.difficulty_rounds next-round [...]
    uv run --no-sync python -m experiments.post_training.task_curation.difficulty_rounds preflight  [...]
    uv run --no-sync python -m experiments.post_training.task_curation.difficulty_rounds report     [...]

`launch` submits one round through Iris with every launch parameter explicit: the root
federates to the serving peer (`--target-cluster`), runs privileged with the cluster
egress policy, forwards the serving secrets, and maps the Daytona credentials onto the
names Harbor reads (`DAYTONA_API_KEY` plus `DAYTONA_API_URL`/`DAYTONA_TARGET`). Keys
come from the documented secret sources — the secrets bundle via `require_secret`, and
the Daytona eval key from `DAYTONA_EVAL_API_KEY` in the environment the caller exported
(`gcloud secrets versions access latest --project=hai-gcp-models --secret=DAYTONA_EVAL_API_KEY`).
Nothing is printed or committed. A continuation declares `--previous-campaign` with
`--previous-serving-sha256` — the digest of the previous run's own sealed `serving.json`,
never an inherited pointer (docs/references/rl-data-atlas.md).

`seal` reports whether a round sealed (`completion.json` with `worker_exit: 0`, or the
root job SUCCEEDED) and prints the sealed `serving.json` SHA-256 that the next
continuation binds to. `next-round` chains the steps: seal check, file-mode coverage
recompute, queue-digest idempotency against a ledger, then the continuation launch;
`--dry-run` prints the plan without submitting.

Only `launch` and `next-round` submit; both refuse when the output root exists or an
Iris job with the same name is already live. A real continuation launch first runs
the local `preflight` (frozen receipt comparison + continuation bindings) and
refuses to submit on any mismatch. Everything else is read-only.

`preflight` and `report` never submit: the preflight reproduces the campaign's
receipt-stage work (panel verification, contract-vintage resolution, frozen
digest comparison) plus the continuation bindings; the report answers per-solver
and per-run questions from the durable artifacts and Iris job state only.
"""

import argparse
import dataclasses
import datetime
import hashlib
import json
import logging
import os
import shlex
import subprocess
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any
from uuid import NAMESPACE_URL, uuid5

from finestore.reader import ReadView
from finestore.rl.difficulty import (
    MEASUREMENTS,
    DifficultyMeasurement,
    DifficultyObservation,
    Solver,
    register_difficulty,
)
from finestore.store import DataStore
from pydantic import ValidationError
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath
from rigging.secrets import require_secret
from taskcompendium.evaluation.corpus import file_sha256

from experiments.post_training.task_curation import difficulty_report
from experiments.post_training.task_curation.difficulty_contracts import ContractVintageError
from experiments.post_training.task_curation.difficulty_preflight import preflight_continuation
from infra.marina.apps.rl_data_catalog.server.difficulty_coverage import (
    DifficultyCoverage,
    campaign_queue,
    difficulty_coverage,
)
from infra.marina.apps.rl_data_catalog.server.source_identity import canonical_source_id

LOGGER = logging.getLogger(__name__)

# The campaign root must federate to the serving peer and run privileged with the
# cluster egress policy; secrets are forwarded explicitly (this Mac has no
# /run/rigging/secrets bundle). Values from the v17 round launches and the
# v94 launch runbook.
ROOT_CLUSTER = "marin"
TARGET_CLUSTER = "cw-rno2a"
ROOT_RESOURCES = ("--cpu", "32", "--memory", "128GB", "--disk", "1TB", "--enable-extra-resources")
CONTAINER_PROFILE = "container_profile_privileged"
EGRESS_POLICY = "egress_policy_cluster"
FORWARDED_SECRETS = ("GLM_BULK_TOKEN", "TOGETHER_API_KEY", "CW_KEY_ID", "CW_KEY_SECRET", "HF_TOKEN")
DAYTONA_KEY_ENV = "DAYTONA_EVAL_API_KEY"
DAYTONA_API_URL = "https://app.daytona.io/api"
DAYTONA_TARGET = "us"
CAMPAIGN_QUEUE_SOURCES = "s3://marin-rno2a/marin/rl-data/atlas/difficulty/2026.11.13.8/samples/sources.json"
CAMPAIGN_QUEUE_SOURCES_SHA256 = "9d318f52e75dd0e76acaf450f3b824e236e9666b12458bd8dc517335fbfc5695"
DEFAULT_WORKERS = 8
DEFAULT_ATTEMPTS_PER_TASK = 3
DEFAULT_DECLARATIONS_APPLET = "c9742a15-fb46-4d56-bc88-aa916351ebd3"
COVERAGE_PUBLISH_ROOT = "s3://marin-rno2a/marin/rl-data/atlas/difficulty/coverage"

SEAL_COMPLETION = "completion.json"
SEAL_FINAL_TRAFFIC = "trials/model-requests-final.json"


class RoundsError(RuntimeError):
    """The rounds operator refused an action or hit an invalid input."""


# ---------------------------------------------------------------------------
# launch


@dataclasses.dataclass(frozen=True)
class LaunchPlan:
    """Everything needed to submit one round; composed and validated before Iris runs."""

    job_name: str
    solver: Solver
    priority: str
    sources: str
    sources_sha256: str
    output_root: str
    local_root: Path
    workers: int
    attempts_per_task: int
    daytona_provisioning: str
    target_cluster: str
    source_ids: tuple[str, ...]
    previous_campaign: str | None = None
    previous_serving_sha256: str | None = None
    task_limit: int | None = None
    daytona_api_url: str = DAYTONA_API_URL
    daytona_target: str = DAYTONA_TARGET


def build_launch_command(plan: LaunchPlan, daytona_key: str | None = None) -> list[str]:
    """The iris CLI argv that submits one difficulty round.

    With `daytona_key` the argv carries the resolved Daytona value inline exactly like
    the runbook's shell form; without it a `env:` placeholder stands in so plans and
    dry runs never print a secret.
    """
    if bool(plan.previous_campaign) != bool(plan.previous_serving_sha256):
        raise RoundsError("Continuation requires both --previous-campaign and --previous-serving-sha256")
    daytona_value = daytona_key if daytona_key is not None else f"env:{DAYTONA_KEY_ENV}"
    command = [
        "uv",
        "run",
        "--no-sync",
        "iris",
        f"--cluster={ROOT_CLUSTER}",
        "job",
        "run",
        "--job-name",
        plan.job_name,
        *ROOT_RESOURCES,
        f"--target-cluster={plan.target_cluster}",
        f"--container-profile={CONTAINER_PROFILE}",
        f"--egress-policy={EGRESS_POLICY}",
        f"--priority={plan.priority}",
        "--max-retries",
        "0",
        "--no-wait",
    ]
    for name in FORWARDED_SECRETS:
        command += ["--secret", name, f"env:{name}"]
    command += ["--secret", "DAYTONA_API_KEY", f"env:{DAYTONA_KEY_ENV}"]
    command += [
        "-e",
        "DAYTONA_API_KEY",
        daytona_value,
        "-e",
        "DAYTONA_API_URL",
        plan.daytona_api_url,
        "-e",
        "DAYTONA_TARGET",
        plan.daytona_target,
        "--",
        "python",
        "-m",
        "experiments.post_training.task_curation.difficulty_campaign",
        "--solver",
        plan.solver.value,
        "--sources",
        plan.sources,
        "--sources-sha256",
        plan.sources_sha256,
        "--local-root",
        str(plan.local_root),
        "--workers",
        str(plan.workers),
        "--attempts-per-task",
        str(plan.attempts_per_task),
        "--daytona-provisioning",
        plan.daytona_provisioning,
        "--output-root",
        plan.output_root,
        "--priority",
        plan.priority,
    ]
    if plan.previous_campaign is not None:
        command += [
            "--previous-campaign",
            plan.previous_campaign,
            "--previous-serving-sha256",
            plan.previous_serving_sha256 or "",
        ]
    if plan.task_limit is not None:
        command += ["--task-limit", str(plan.task_limit)]
    for source_id in plan.source_ids:
        command += ["--source-id", source_id]
    return command


def launch_environment() -> dict[str, str]:
    """The submission environment: forwarded secrets resolved from the documented
    sources, plus the Daytona eval key. Values are never logged."""
    environment = {**os.environ}
    for name in FORWARDED_SECRETS:
        environment[name] = require_secret(name)
    if not environment.get(DAYTONA_KEY_ENV):
        raise RoundsError(
            f"{DAYTONA_KEY_ENV} is not set; export it first (gcloud secrets versions access latest "
            f"--project=hai-gcp-models --secret={DAYTONA_KEY_ENV})"
        )
    return environment


CREDENTIAL_ENV_DECLARATIONS = ("DAYTONA_API_KEY", "DAYTONA_API_URL", "DAYTONA_TARGET")
"""The env names the campaign's Daytona connection resolves from the job environment."""


def declared_environment_names(command: list[str]) -> set[str]:
    """The env and secret names one iris submit argv declares."""
    names: set[str] = set()
    for index, token in enumerate(command):
        if token in ("-e", "--secret") and index + 1 < len(command):
            names.add(command[index + 1])
    return names


def require_credential_declarations(command: list[str]) -> None:
    """Refuse a submit whose argv does not declare the Daytona credential env.

    The trials worker resolves the Daytona connection from the job's own
    environment, so a spec that drops one declaration surfaces only at the
    first Daytona trial (large-v22-relay spent 17 hours of serving before its
    env gap raised). Never print the command: the resolved argv carries the
    key inline.
    """
    missing = [name for name in CREDENTIAL_ENV_DECLARATIONS if name not in declared_environment_names(command)]
    if missing:
        raise RoundsError(
            "The submit command does not declare the Daytona credential environment "
            f"({', '.join(missing)}); refusing to submit"
        )


TERMINAL_IRIS_STATES = ("succeeded", "failed", "cancelled", "killed")


def iris_job_exists(cluster: str, job_name: str) -> bool:
    """Whether an Iris job with this name is already live (double-submit guard).

    Terminal prior attempts (failed, cancelled) do not block a relaunch; the
    row's state column decides, so a failed first attempt of a round can be
    resubmitted under a fresh output root.
    """
    job_id = f"/benfeuer/{job_name}"
    try:
        listing = subprocess.run(
            ["uv", "run", "--no-sync", "iris", f"--cluster={cluster}", "job", "list", "--prefix", job_id],
            capture_output=True,
            text=True,
            timeout=240,
            check=True,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        raise RoundsError(f"could not verify whether {job_id} is live; refusing to submit: {e}") from e
    for line in listing.stdout.splitlines():
        stripped = line.strip()
        if not stripped.startswith(job_id) or stripped.startswith(f"{job_id}/"):
            continue
        fields = stripped.split()
        state = fields[1].lower() if len(fields) > 1 else ""
        if state and state not in TERMINAL_IRIS_STATES:
            return True
    return False


def preflight_launch(plan: LaunchPlan) -> dict[str, Any]:
    """Fail a submission locally before Iris runs it.

    The real failure classes, each caught here now: hosted-v13 died on a
    frozen-receipt vintage mismatch, hosted-v13b on a runtime/task validation
    conflict, hosted-v13c on a serving pod deleted behind a running relay
    record, and small-v22 on a truncated serving digest. The gate reproduces
    every one on the submitting machine: hosted endpoint liveness, the
    continuation bindings, the frozen-receipt comparison per selected source,
    and full local task staging.
    """
    try:
        with tempfile.TemporaryDirectory(prefix=f"{plan.job_name}-preflight-") as scratch:
            receipt = preflight_continuation(
                plan.sources,
                plan.sources_sha256,
                plan.source_ids or None,
                plan.previous_campaign,
                plan.previous_serving_sha256,
                stage_root=Path(scratch) / "tasks",
                probe_hosted=plan.solver == Solver.HOSTED,
            )
    except (ContractVintageError, ValueError) as error:
        raise RoundsError(f"launch preflight failed; refusing to submit: {error}") from error
    LOGGER.info(
        "launch preflight passed: %d sources, %d tasks, contract vintages %s (staged locally)",
        receipt["sources"],
        receipt["tasks"],
        receipt["vintages"],
    )
    return receipt


def launch_round(plan: LaunchPlan, *, dry_run: bool = False) -> dict[str, Any]:
    """Submit one round; refuses to double-submit on output root or live job.

    The double-submit probes (output root, live job) run only for real submissions;
    a dry-run plan reports without provider access. A real continuation runs the
    local preflight first and refuses to submit on a frozen-receipt mismatch.
    """
    if not dry_run:
        configure_coreweave_s3()
        if StoragePath(plan.output_root).exists():
            raise RoundsError(f"output root already exists (preserve prior campaigns): {plan.output_root}")
        if iris_job_exists(ROOT_CLUSTER, plan.job_name):
            raise RoundsError(f"an Iris job named {plan.job_name} is already live; refusing to double-submit")
        if plan.previous_campaign is not None or plan.solver == Solver.HOSTED:
            plan_state = preflight_launch(plan)
            LOGGER.info("preflight receipt: %s", json.dumps(plan_state, indent=2))
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
    if dirty and not dry_run:
        raise RoundsError("working tree is dirty; commit the round's code before submitting")
    # A launch ships this working tree; warn when it is behind the rounds landing
    # branch so a stale worktree cannot ship an old bundle unnoticed.
    lineage = subprocess.run(
        ["git", "fetch", "--quiet", "origin", "atlas-env-audit"],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    if lineage.returncode == 0:
        behind = subprocess.run(
            ["git", "merge-base", "--is-ancestor", "HEAD", "origin/atlas-env-audit"],
            capture_output=True,
            text=True,
            check=False,
        )
        if behind.returncode != 0:
            LOGGER.warning(
                "HEAD %s is not an ancestor of origin/atlas-env-audit; the submitted bundle "
                "will not contain that branch's later fixes",
                revision[:12],
            )
    dry_command = build_launch_command(plan)
    require_credential_declarations(dry_command)
    state = {
        "job_id": f"/benfeuer/{plan.job_name}",
        "root_cluster": ROOT_CLUSTER,
        "target_cluster": plan.target_cluster,
        "code_revision": revision,
        "output_root": plan.output_root,
        "solver": plan.solver.value,
        "priority": plan.priority,
        "previous_campaign": plan.previous_campaign,
        "previous_serving_sha256": plan.previous_serving_sha256,
        "submitted_at_utc": datetime.datetime.now(datetime.UTC).isoformat(),
        # The recorded command uses the env: placeholder; the executed argv carries the
        # resolved Daytona key inline and is never logged.
        "resubmit_command": shlex.join(dry_command),
        "source_ids": list(plan.source_ids),
    }
    if dry_run:
        state["dry_run"] = True
        LOGGER.info("dry run; command:\n%s", state["resubmit_command"])
        return state
    log_path = Path(f"{plan.job_name}-submit.log")
    if log_path.exists():
        raise RoundsError(f"submit log already exists: {log_path}")
    environment = launch_environment()
    command = build_launch_command(plan, daytona_key=environment[DAYTONA_KEY_ENV])
    require_credential_declarations(command)
    with log_path.open("x") as stream:
        subprocess.run(command, env=environment, stdout=stream, stderr=subprocess.STDOUT, check=True)
    LOGGER.info("submitted %s -> %s (log: %s)", state["job_id"], plan.output_root, log_path)
    return state


# ---------------------------------------------------------------------------
# poll and seal


def _iris_root_job_state(job_id: str) -> str:
    """The live Iris state of one root job from the submitting cluster's list."""
    try:
        listing = subprocess.run(
            ["uv", "run", "--no-sync", "iris", f"--cluster={ROOT_CLUSTER}", "job", "list", "--prefix", job_id],
            capture_output=True,
            text=True,
            timeout=240,
            check=True,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        return f"iris_list_failed: {e}"
    for line in listing.stdout.splitlines():
        stripped = line.strip()
        if stripped.startswith(job_id) and not stripped.startswith(f"{job_id}/"):
            parts = stripped.split()
            return parts[1] if len(parts) > 1 else "unknown"
    return "not_found"


def read_campaign_progress(job_id: str, output_root: str) -> dict[str, Any]:
    """Primary-evidence progress for one campaign: Iris state plus catalog counts."""
    configure_coreweave_s3()
    catalog = StoragePath(output_root) / "trials/catalog"
    rows = []
    if catalog.exists():
        rows = list(ReadView(str(catalog)).iter_rows(MEASUREMENTS, columns=["status", "gpu_hours"]))
    return {
        "job_id": job_id,
        "job_state": _iris_root_job_state(job_id),
        "output_root": output_root,
        "aggregated": len(rows),
        "measured": sum(row["status"] == "measured" for row in rows),
        "pending": sum(row["status"] == "pending" for row in rows),
        "gpu_hours": round(sum(float(row["gpu_hours"] or 0) for row in rows), 1),
        "polled_at_utc": datetime.datetime.now(datetime.UTC).isoformat(),
    }


def seal_status(output_root: str, job_id: str | None = None) -> dict[str, Any]:
    """Whether one round sealed, with the serving.json digest a continuation binds to."""
    configure_coreweave_s3()
    root = StoragePath(output_root)
    completion_ok = False
    completion = root / SEAL_COMPLETION
    if completion.exists():
        completion_ok = json.loads(completion.read_text()).get("worker_exit") == 0
    final_traffic = (root / SEAL_FINAL_TRAFFIC).exists()
    job_state = _iris_root_job_state(job_id) if job_id is not None else None
    serving_path = root / "serving.json"
    serving_sha = file_sha256(serving_path) if serving_path.exists() else None
    return {
        "output_root": output_root,
        "job_id": job_id,
        "job_state": job_state,
        "completion_exit_zero": completion_ok,
        "final_traffic_sealed": final_traffic,
        "sealed": completion_ok or job_state == "succeeded",
        "serving_json_sha256": serving_sha,
        "checked_at_utc": datetime.datetime.now(datetime.UTC).isoformat(),
    }


# ---------------------------------------------------------------------------
# coverage (file mode)


def fetch_declarations(applet_id: str) -> dict[str, dict[str, Any]]:
    """The served catalog's declaration map from the Atlas app's current load."""
    applet_table = f"applet_{applet_id.replace('-', '')}"
    # The full declarations exceed the Marina SQL response limit; the coverage
    # classifier reads these five fields, so fetch exactly them.
    sql = (
        f"SELECT id, jsonb_build_object("
        f"'id', declaration->'id', 'status', declaration->'status', "
        f"'current_release', declaration->'current_release', "
        f"'pipeline', declaration->'pipeline', "
        f"'actor_census_input', declaration->'actor_census_input') AS declaration "
        f"FROM {applet_table}.atlas_sources "
        f"WHERE load_id=(SELECT load_id FROM {applet_table}.atlas_current)"
    )
    raw = subprocess.run(
        ["uv", "run", "--no-sync", "marina", "apps", "sql", applet_id, sql],
        capture_output=True,
        text=True,
        timeout=300,
        check=True,
    ).stdout
    payload = json.loads(raw[raw.index("{") :])
    declarations: dict[str, dict[str, Any]] = {}
    for row in payload["rows"]:
        declaration = row["declaration"]
        declarations[declaration.get("id") or row["id"]] = declaration
    return declarations


def collect_observations(measurement_locations: Sequence[str]) -> dict[str, list[DifficultyObservation]]:
    """Difficulty observations from qualified snapshot exports and campaign trial dirs.

    Qualified snapshots expose a `measurements.json`; campaign roots expose
    `trials/<dir>/measurement.json`. Rows carry a sample.tasks payload the observation
    model forbids, so it is stripped exactly like the app's loader does.
    """
    configure_coreweave_s3()
    observations: dict[str, list[DifficultyObservation]] = {}
    kept = dropped = 0
    for location in measurement_locations:
        root = StoragePath(location)
        rows: list[dict[str, Any]] = []
        if (root / "measurements.json").exists():
            rows = json.loads((root / "measurements.json").read_text())
        else:
            trials = root / "trials"
            if not trials.exists():
                raise RoundsError(f"no measurements.json and no trials/ under {location}")
            for trial_dir in trials.ls():
                if trial_dir.name in ("catalog", "judge"):
                    continue
                measurement = trial_dir / "measurement.json"
                if measurement.exists():
                    rows.append(json.loads(measurement.read_text()))
        for row in rows:
            sample = row.get("sample")
            if isinstance(sample, dict) and "tasks" in sample:
                row = {**row, "sample": {key: value for key, value in sample.items() if key != "tasks"}}
            try:
                observation = DifficultyObservation.model_validate(row)
            except ValidationError as e:
                # A stored row that fails the observation contract is recorded as
                # dropped with its reason; a programming error propagates loudly.
                dropped += 1
                if dropped == 1:
                    LOGGER.warning("first dropped observation %s: %s", row.get("measurement_id"), str(e)[:200])
                continue
            kept += 1
            observations.setdefault(canonical_source_id(observation.source_id), []).append(observation)
    LOGGER.info("observations kept %d dropped %d from %d location(s)", kept, dropped, len(measurement_locations))
    return observations


def compute_coverage(
    declarations: dict[str, dict[str, Any]], measurement_locations: Sequence[str]
) -> DifficultyCoverage:
    """The coverage document the Atlas app's own classifier computes, file mode."""
    return difficulty_coverage(declarations, collect_observations(measurement_locations))


def write_coverage_artifacts(
    document: DifficultyCoverage,
    declarations: dict[str, dict[str, Any]],
    output_dir: Path | None,
) -> tuple[list[dict[str, Any]], str]:
    """Emit the coverage document, declarations and campaign queue; returns (queue, digest)."""
    queue = campaign_queue(document, declarations)
    digest = queue_digest(queue)
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "coverage-document.json").write_text(json.dumps(document.model_dump(mode="json"), indent=2) + "\n")
        (output_dir / "declarations.json").write_text(json.dumps(declarations, indent=2) + "\n")
        (output_dir / "campaign-queue.json").write_text(json.dumps(queue, indent=2) + "\n")
    return queue, digest


# ---------------------------------------------------------------------------
# snapshot export


def export_snapshot(
    queue_path: Path,
    input_catalogs: Sequence[tuple[str, tuple[str, ...] | None]],
    destination: str,
    qualification_holds_path: Path | None,
) -> dict[str, Any]:
    """Build a qualified difficulty snapshot: latest measurement per (source, solver),
    not_measured backfill, FineStore catalog, and an import receipt.

    Legacy bespoke inputs (grade/traffic audits, legacy hosted holds) stay outside the
    repo; pass their prepared `qualification-holds` document when a snapshot needs one.
    """
    configure_coreweave_s3()
    queue = json.loads(queue_path.read_text())
    as_of = datetime.datetime.now(datetime.UTC).isoformat()
    holds: dict[str, Any] = {"holds": []}
    if qualification_holds_path is not None:
        holds = json.loads(qualification_holds_path.read_text())

    selected: dict[tuple[str, str], dict[str, Any]] = {}
    for catalog, source_ids in input_catalogs:
        for row in ReadView(catalog).iter_rows(MEASUREMENTS, columns=list(DifficultyMeasurement.model_fields)):
            if source_ids is not None and row["source_id"] not in source_ids:
                continue
            key = (row["source_id"], row["solver"])
            prior = selected.get(key)
            if prior is None or str(prior["created_at"]) < str(row["created_at"]):
                selected[key] = row

    destination_path = StoragePath(destination)
    records: list[dict[str, Any]] = []
    for row in queue:
        sample = row["sample"]
        source_id = sample["source_id"]
        for solver in Solver:
            record = selected.get((source_id, solver.value))
            matching_holds = [
                hold
                for hold in holds["holds"]
                if hold["source_id"] == source_id and hold.get("solver", solver.value) == solver.value
            ]
            reason = "No qualified Harbor measurement has been recorded for this source/model in this snapshot."
            if matching_holds:
                record = None
                reason = "; ".join(hold["reason"] for hold in matching_holds)
            if record is None:
                record = DifficultyMeasurement(
                    measurement_id=uuid5(
                        NAMESPACE_URL, str(destination_path) + "/" + source_id + "/" + solver.value
                    ).hex,
                    protocol_id="atlas-difficulty-v3",
                    created_at=as_of,
                    source_id=source_id,
                    release=sample["release"],
                    converter_version=sample["converter_version"],
                    solver=solver,
                    status="not_measured",
                    reason=reason,
                    sample=sample,
                    sample_receipt=row["sample_receipt"],
                    model=None,
                    settings=None,
                    n=0,
                    solved=0,
                    wilson_95=None,
                    infrastructure_attempts=0,
                    pending_tasks=len(sample["tasks"]),
                    attempts_receipt=None,
                    gpu_hours=0,
                    input_tokens=0,
                    output_tokens=0,
                ).model_dump(mode="json")
            records.append(record)

    rows_data = (json.dumps(records, indent=2) + "\n").encode()
    (destination_path / "measurements.json").write_bytes(rows_data)
    with DataStore.open(str(destination_path / "catalog")) as store:
        register_difficulty(store)
        with store.transaction() as transaction:
            for record in records:
                transaction.table(MEASUREMENTS).add(record)
    receipt = {
        "as_of": as_of,
        "protocol": "atlas-difficulty-v3",
        "catalog_root": str(destination_path / "catalog"),
        "records": len(records),
        "sources": len(queue),
        "models": [solver.value for solver in Solver],
        "measurements": {
            "uri": str(destination_path / "measurements.json"),
            "sha256": hashlib.sha256(rows_data).hexdigest(),
        },
        "input_catalogs": [catalog for catalog, _ in input_catalogs],
        "qualification_holds": str(qualification_holds_path) if qualification_holds_path else None,
    }
    receipt_data = (json.dumps(receipt, indent=2) + "\n").encode()
    (destination_path / "import-receipt.json").write_bytes(receipt_data)
    LOGGER.info("exported %d records for %d sources to %s", len(records), len(queue), destination)
    return receipt


# ---------------------------------------------------------------------------
# next-round chaining


def queue_digest(queue_rows: Sequence[dict[str, Any]]) -> str:
    """Canonical digest of the emitted campaign queue, binding idempotency."""
    payload = json.dumps(list(queue_rows), sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(payload).hexdigest()


def read_ledger(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def append_ledger(path: Path, entry: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as stream:
        stream.write(json.dumps(entry, separators=(",", ":"), default=str) + "\n")


def queue_source_ids(queue: Sequence[dict[str, Any]]) -> tuple[str, ...]:
    """The --source-id selection for a round launched from a coverage queue."""
    ids = []
    for entry in queue:
        declaration = entry["declaration"]
        source_id = declaration.get("id")
        if source_id is None:
            raise RoundsError(f"queue entry declaration carries no id: {sorted(declaration)[:8]}")
        ids.append(source_id)
    return tuple(ids)


def gather_source_ids(source_id: Sequence[str], source_ids_file: Path | None) -> list[str]:
    """The launch/preflight source selection: repeated flags plus a one-per-line file."""
    ids = list(source_id)
    if source_ids_file:
        ids += [
            line.strip()
            for line in source_ids_file.read_text().splitlines()
            if line.strip() and not line.startswith("#")
        ]
    return ids


def frozen_source_ids() -> set[str]:
    """Source ids in the frozen sample selection the campaigns measure."""
    configure_coreweave_s3()
    rows = json.loads(StoragePath(CAMPAIGN_QUEUE_SOURCES).read_text())
    return {row["declaration"]["id"] for row in rows}


def plan_next_round(
    sealed: dict[str, Any],
    document: DifficultyCoverage,
    declarations: dict[str, dict[str, Any]],
    ledger: list[dict[str, Any]],
) -> dict[str, Any]:
    """The continuation plan for a sealed round: queue, digest, idempotency verdict."""
    queue = campaign_queue(document, declarations)
    # The campaign measures the frozen sample selection; queued sources absent from
    # it (new ingests, re-pinned sources) go back through sampling and are reported
    # rather than submitted.
    frozen_ids = frozen_source_ids()
    measurable = [entry for entry in queue if entry["declaration"]["id"] in frozen_ids]
    absent = sorted(entry["declaration"]["id"] for entry in queue if entry["declaration"]["id"] not in frozen_ids)
    digest = queue_digest(measurable)
    plan: dict[str, Any] = {
        "queue_entries": len(measurable),
        "queue_digest": digest,
        "source_ids": list(queue_source_ids(measurable)),
        "absent_from_frozen_selection": absent,
        "already_submitted": any(row.get("queue_digest") == digest for row in ledger),
        "previous_campaign": sealed["output_root"],
        "previous_serving_sha256": sealed["serving_json_sha256"],
    }
    if sealed["serving_json_sha256"] is None:
        plan["refusal"] = "the sealed round has no serving.json; cannot chain a continuation"
    if plan["already_submitted"]:
        plan["refusal"] = f"queue digest {digest[:12]} already in the ledger; refusing to double-submit"
    return plan


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)

    launch = commands.add_parser("launch", help="Submit one round (fresh or continuation).")
    launch.add_argument("--solver", required=True, choices=[solver.value for solver in Solver])
    launch.add_argument("--job-name", required=True)
    launch.add_argument("--output-root", required=True)
    launch.add_argument("--local-root", required=True, type=Path)
    launch.add_argument("--priority", required=True, choices=("production", "interactive", "batch", "system"))
    launch.add_argument("--sources", default=CAMPAIGN_QUEUE_SOURCES)
    launch.add_argument("--sources-sha256", default=CAMPAIGN_QUEUE_SOURCES_SHA256)
    launch.add_argument("--source-id", action="append", default=[])
    launch.add_argument("--source-ids-file", type=Path, help="One source id per line; # comments allowed.")
    launch.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    launch.add_argument("--attempts-per-task", type=int, default=DEFAULT_ATTEMPTS_PER_TASK)
    launch.add_argument("--task-limit", type=int)
    launch.add_argument("--daytona-provisioning", default="declarative", choices=("snapshot", "declarative"))
    launch.add_argument("--target-cluster", default=TARGET_CLUSTER)
    launch.add_argument("--previous-campaign")
    launch.add_argument("--previous-serving-sha256")
    launch.add_argument("--daytona-api-url", default=DAYTONA_API_URL)
    launch.add_argument("--daytona-target", default=DAYTONA_TARGET)
    launch.add_argument("--dry-run", action="store_true")

    poll = commands.add_parser("poll", help="Primary-evidence progress for named campaigns.")
    poll.add_argument("--campaign", action="append", required=True, metavar="NAME=JOB_ID,OUTPUT_ROOT")
    poll.add_argument("--json", type=Path)

    seal = commands.add_parser("seal", help="Report whether a round sealed and its serving.json digest.")
    seal.add_argument("--output-root", required=True)
    seal.add_argument("--job-id")

    coverage = commands.add_parser("coverage", help="Recompute the coverage document and queue (file mode).")
    coverage.add_argument("--declarations-applet", default=DEFAULT_DECLARATIONS_APPLET)
    coverage.add_argument("--declarations-file", type=Path, help="Skip the app query and read declarations from a file.")
    coverage.add_argument(
        "--measurements",
        action="append",
        required=True,
        help="Qualified measurements.json or campaign root; repeatable.",
    )
    coverage.add_argument(
        "--output-dir", type=Path, help="Write coverage-document/declarations/campaign-queue artifacts here."
    )

    exporter = commands.add_parser("export-snapshot", help="Export a qualified difficulty snapshot.")
    exporter.add_argument("--queue", type=Path, required=True)
    exporter.add_argument("--input-catalog", action="append", required=True, metavar="CATALOG[,ID1;ID2]")
    exporter.add_argument("--destination", required=True)
    exporter.add_argument("--qualification-holds", type=Path)

    chain = commands.add_parser("next-round", help="Seal -> coverage -> queue -> continuation launch, idempotent.")
    chain.add_argument("--watch-job-id", required=True)
    chain.add_argument("--watch-output-root", required=True)
    chain.add_argument("--job-name", required=True, help="The next round's Iris job name.")
    chain.add_argument("--output-root", required=True, help="The next round's output root (must not exist).")
    chain.add_argument("--declarations-applet", default=DEFAULT_DECLARATIONS_APPLET)
    chain.add_argument("--measurements", action="append", required=True)
    chain.add_argument("--coverage-output-dir", type=Path)
    chain.add_argument("--ledger", type=Path, required=True)
    chain.add_argument("--solver", default="small", choices=[solver.value for solver in Solver])
    chain.add_argument("--priority", default="interactive", choices=("production", "interactive", "batch", "system"))
    chain.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    chain.add_argument("--attempts-per-task", type=int, default=DEFAULT_ATTEMPTS_PER_TASK)
    chain.add_argument("--daytona-provisioning", default="declarative", choices=("snapshot", "declarative"))
    chain.add_argument(
        "--operator-stopped",
        action="store_true",
        help=(
            "Treat an operator-stopped watch round (killed/failed root with final traffic "
            "sealed) as sealed for the carry-over; recorded in the ledger."
        ),
    )
    chain.add_argument("--dry-run", action="store_true")

    preflight = commands.add_parser(
        "preflight", help="Verify a continuation's frozen receipts and bindings locally before submission."
    )
    preflight.add_argument("--sources", default=CAMPAIGN_QUEUE_SOURCES)
    preflight.add_argument("--sources-sha256", default=CAMPAIGN_QUEUE_SOURCES_SHA256)
    preflight.add_argument("--source-id", action="append", default=[])
    preflight.add_argument("--source-ids-file", type=Path, help="One source id per line; # comments allowed.")
    preflight.add_argument("--previous-campaign")
    preflight.add_argument("--previous-serving-sha256")
    preflight.add_argument(
        "--solver",
        choices=[solver.value for solver in Solver],
        help="With hosted, also probes the hosted endpoint's liveness.",
    )
    preflight.add_argument(
        "--receipt-stage-only",
        action="store_true",
        help="Skip local task staging; verify receipts and bindings only.",
    )
    preflight.add_argument("--json", type=Path)

    report = commands.add_parser("report", help="Report campaign state from the durable artifacts (never STATUS.md).")
    report.add_argument("--base", default=difficulty_report.DIFFICULTY_ROOT)
    report.add_argument("--solver", action="append", choices=[solver.value for solver in Solver])
    report.add_argument("--source", action="append", help="Show one source across solvers; repeatable.")
    report.add_argument("--prepared", action="store_true", help="Read prepared.json for exact per-run selections.")
    report.add_argument("--no-iris", action="store_true", help="Skip the Iris job listing.")
    report.add_argument("--queue", action="append", metavar="SHA256=PATH", help="Queue path override; repeatable.")
    report.add_argument("--json", type=Path)

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

    if args.command == "launch":
        source_ids = gather_source_ids(args.source_id, args.source_ids_file)
        plan = LaunchPlan(
            job_name=args.job_name,
            solver=Solver(args.solver),
            priority=args.priority,
            sources=args.sources,
            sources_sha256=args.sources_sha256,
            output_root=args.output_root,
            local_root=args.local_root,
            workers=args.workers,
            attempts_per_task=args.attempts_per_task,
            daytona_provisioning=args.daytona_provisioning,
            target_cluster=args.target_cluster,
            source_ids=tuple(source_ids),
            previous_campaign=args.previous_campaign,
            previous_serving_sha256=args.previous_serving_sha256,
            task_limit=args.task_limit,
            daytona_api_url=args.daytona_api_url,
            daytona_target=args.daytona_target,
        )
        print(json.dumps(launch_round(plan, dry_run=args.dry_run), indent=2, default=str))
        return

    if args.command == "poll":
        snapshot: dict[str, Any] = {"campaigns": {}}
        for item in args.campaign:
            name, _, rest = item.partition("=")
            job_id, _, output_root = rest.partition(",")
            snapshot["campaigns"][name] = read_campaign_progress(job_id, output_root)
        if args.json:
            args.json.write_text(json.dumps(snapshot, indent=2) + "\n")
        print(json.dumps(snapshot, indent=2))
        return

    if args.command == "seal":
        status = seal_status(args.output_root, args.job_id)
        print(json.dumps(status, indent=2))
        raise SystemExit(0 if status["sealed"] else 1)

    if args.command == "coverage":
        configure_coreweave_s3()
        declarations = fetch_declarations(args.declarations_applet)
        if args.declarations_file:
            declarations = json.loads(args.declarations_file.read_text())
        document = compute_coverage(declarations, args.measurements)
        queue, digest = write_coverage_artifacts(document, declarations, args.output_dir)
        print(json.dumps({"counts": document.counts, "queue_entries": len(queue), "queue_digest": digest}, indent=2))
        return

    if args.command == "export-snapshot":
        inputs = []
        for item in args.input_catalog:
            catalog, _, ids = item.partition(",")
            inputs.append((catalog, tuple(i for i in ids.split(";") if i) if ids else None))
        receipt = export_snapshot(args.queue, inputs, args.destination, args.qualification_holds)
        print(json.dumps(receipt, indent=2, default=str))
        return

    if args.command == "next-round":
        configure_coreweave_s3()
        sealed = seal_status(args.watch_output_root, args.watch_job_id)
        if not sealed["sealed"] and args.operator_stopped:
            # An operator stop (rule 729 cancellation) leaves per-source evidence and
            # the final traffic seal durable; the carry-over continues from them.
            if sealed["final_traffic_sealed"] and sealed["job_state"] in ("killed", "failed"):
                sealed["sealed"] = True
                sealed["operator_stopped"] = True
        if not sealed["sealed"]:
            print(json.dumps({"sealed": False, "seal": sealed}, indent=2))
            return
        declarations = fetch_declarations(args.declarations_applet)
        measurements = [*args.measurements, args.watch_output_root]
        document = compute_coverage(declarations, measurements)
        queue, digest = write_coverage_artifacts(document, declarations, args.coverage_output_dir)
        plan = plan_next_round(sealed, document, declarations, read_ledger(args.ledger))
        plan["job_name"] = args.job_name
        plan["output_root"] = args.output_root
        plan["solver"] = args.solver
        plan["priority"] = args.priority
        if "refusal" in plan:
            plan["action"] = "refused"
            print(json.dumps({"sealed": True, "plan": plan}, indent=2))
            return
        launch_plan = LaunchPlan(
            job_name=args.job_name,
            solver=Solver(args.solver),
            priority=args.priority,
            sources=CAMPAIGN_QUEUE_SOURCES,
            sources_sha256=CAMPAIGN_QUEUE_SOURCES_SHA256,
            output_root=args.output_root,
            local_root=Path("/tmp") / args.job_name,
            workers=args.workers,
            attempts_per_task=args.attempts_per_task,
            daytona_provisioning=args.daytona_provisioning,
            target_cluster=TARGET_CLUSTER,
            source_ids=tuple(plan["source_ids"]),
            previous_campaign=plan["previous_campaign"],
            previous_serving_sha256=plan["previous_serving_sha256"],
        )
        state = launch_round(launch_plan, dry_run=args.dry_run)
        if not args.dry_run:
            append_ledger(
                args.ledger,
                {
                    "submitted_at_utc": state["submitted_at_utc"],
                    "queue_digest": plan["queue_digest"],
                    "queue_entries": plan["queue_entries"],
                    "previous_campaign": plan["previous_campaign"],
                    "previous_serving_sha256": plan["previous_serving_sha256"],
                    "job_id": state["job_id"],
                    "output_root": state["output_root"],
                    **({"operator_stopped": True} if sealed.get("operator_stopped") else {}),
                },
            )
        print(json.dumps({"sealed": True, "plan": plan, "launch": state}, indent=2, default=str))
        return

    if args.command == "preflight":
        source_ids = gather_source_ids(args.source_id, args.source_ids_file)
        configure_coreweave_s3()
        with tempfile.TemporaryDirectory(prefix="difficulty-preflight-") as scratch:
            receipt = preflight_continuation(
                args.sources,
                args.sources_sha256,
                tuple(source_ids) or None,
                args.previous_campaign,
                args.previous_serving_sha256,
                None if args.receipt_stage_only else Path(scratch) / "tasks",
                probe_hosted=args.solver == Solver.HOSTED,
            )
        print(json.dumps(receipt, indent=2))
        if args.json:
            args.json.write_text(json.dumps(receipt, indent=2) + "\n")
        return

    if args.command == "report":
        queue_overrides = {}
        for item in args.queue or []:
            sha, _, path = item.partition("=")
            queue_overrides[sha] = path
        document = difficulty_report.build_report(
            base=args.base,
            solvers=tuple(args.solver or ()),
            include_prepared=args.prepared,
            with_iris=not args.no_iris,
            queue_overrides=queue_overrides,
            source_ids=tuple(args.source or ()),
        )
        print(difficulty_report.render_text(document))
        if args.json:
            args.json.write_text(json.dumps(document, indent=2, default=str) + "\n")
        return

    raise RoundsError(f"unknown command {args.command}")


if __name__ == "__main__":
    main()
