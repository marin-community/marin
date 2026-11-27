# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Stage-f storage recalibration for the nemotron-gym QA sources (coordinator dispatch 2026-11-23).

TaskTrove:laion__nemotron-gym-agentic-indirect-prompt-injection-v3 and
TaskTrove:laion__nemotron-gym-qa-abstention-v4 declare ``[environment] storage_mb = 20480``
(20 GiB) in every archived ``task.toml``. The Daytona per-sandbox disk maximum is 10 GiB
(``shellbox.backends.daytona.usage_cli.PER_SANDBOX_MAXIMUM``), so every difficulty attempt
fails at sandbox create with ``UnsupportedMachineSpec`` and the sources report
COVERAGE_SHORTFALL with zero scored attempts. ``cpus = 1`` and ``memory_mb = 4096`` are
within the maxima; storage is the only dimension above them.

The coordinator's dispatch approves the maintained recalibration lane: the declared storage
is replaced only by a measured stage-f calibration (start 2048 MiB, doubling only the
dimension that failed, per ``taskcompendium.pipeline.resource_calibration``). This module
builds the calibration inputs for ``oracle_filter calibrate``:

1. ``census`` proves the declared limits are uniform and fingerprints the template.
2. ``pack`` writes one deviation packet per sampled task — the frozen native package with
   ``storage_mb`` removed, everything else byte-identical — plus deterministic
   calibration-only solutions and the exact-gate precheck receipts. Removing the key makes
   storage a *missing* dimension, which is the only way the maintained ladder may calibrate
   it; the dispatch is the recorded approved deviation.
3. ``revision`` applies the measured storage to the full population through
   ``calibrated_task_spec``, with a per-task before/after ledger.

The calibration and NoP trials themselves run through the maintained drivers
(``oracle_filter calibrate`` and Harbor ``trials start`` with the ``nop`` agent).
"""

import argparse
import base64
import hashlib
import json
import logging
import os
import shutil
import subprocess
import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import tomlkit
from huggingface_hub import hf_hub_download
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath
from rigging.secrets import get_secret
from shellbox.backends.daytona.connection import DEFAULT_DAYTONA_API_URL
from taskcompendium.harbor.native_package import write_native_task
from taskcompendium.identity import canonical_sha256, task_spec_sha256
from taskcompendium.models import TaskSpec
from taskcompendium.pipeline.resource_calibration import INITIAL_REQUIREMENTS

from experiments.post_training.task_curation.calibration_projection import calibrated_task_spec
from experiments.post_training.task_curation.harbor_inspection import harbor_inspection
from experiments.post_training.task_curation.license_publication import publish_licensed_population
from experiments.post_training.task_curation.oracle_filter import (
    CALIBRATION_ATTEMPT_TIMEOUT,
    calibrate_cohort,
    row_task,
)

logger = logging.getLogger(__name__)

HARBOR_PROJECT = Path(__file__).resolve().parents[3] / "vendor/harbor"


def bind_daytona_connection_environment() -> None:
    """Expose the DATA-account Daytona connection where Harbor reads it.

    Iris mounts secrets as files under the rigging secrets dir, but Harbor resolves its
    Daytona connection from ``os.environ`` (``host_capabilities_from_environment``), and the
    calibration trials run as Harbor subprocesses. Bind the DATA credential the difficulty
    worker binds (``difficulty.runtime``), failing loudly when it is absent.
    """
    key = get_secret("DAYTONA_DATA_API_KEY")
    if not key:
        raise ValueError("DAYTONA_DATA_API_KEY is required for the Daytona calibration trials")
    os.environ["DAYTONA_API_KEY"] = key
    os.environ.setdefault("DAYTONA_TARGET", "us")
    os.environ.setdefault("DAYTONA_API_URL", DEFAULT_DAYTONA_API_URL)


TASK_TOML = "task.toml"
ANSWER_FILE = "/app/answer.txt"

ABSTENTION_SOURCE = "TaskTrove:laion__nemotron-gym-qa-abstention-v4"
INJECTION_SOURCE = "TaskTrove:laion__nemotron-gym-agentic-indirect-prompt-injection-v3"

RELEASES = {
    ABSTENTION_SOURCE: (
        "open-athena/task-trove-laion-nemotron-gym-qa-abstention-v4-6065fc39-retained-20261116-v1",
        "06945779da0cc162e3550cc021af7812ef1721f8",
    ),
    INJECTION_SOURCE: (
        "open-athena/task-trove-laion-nemotron-gym-agentic-indirect-prompt-i-97d1380c-retained-20261116-v1",
        "15594cb56a1aaddcf6b7b3d02d3f017978ab554b",
    ),
}

RECALIBRATION_LABEL = "recalibrated-20261123"
MAXIMUMS = {"cpus": 4, "memory_mb": 8192, "storage_mb": 10240}
"""The Daytona per-sandbox maximum (shellbox.backends.daytona.usage_cli.PER_SANDBOX_MAXIMUM)."""
DISPATCH_REFERENCE = (
    "coordinator dispatch experiments/active/difficulty-source-fixers/BRIEF.md (2026-11-23): "
    "declared limits replaced only by a measured calibration under the stage-f rule"
)
EMPTY_RUNTIME_SHA256 = canonical_sha256({})
"""The frozen panels bind empty runtime packets; their canonical hash names that binding."""

TASK_SCHEMA = pa.schema(
    [
        ("id", pa.string()),
        ("path", pa.string()),
        ("task_spec", pa.string()),
        ("task_sha256", pa.string()),
        ("runtime", pa.string()),
    ]
)


def read_release(source_id: str, cache_dir: Path) -> Path:
    """Download the source's retained release parquet once and return its local path."""
    repo, revision = RELEASES[source_id]
    path = hf_hub_download(repo_id=repo, repo_type="dataset", revision=revision, filename="tasks.parquet")
    cache_dir.mkdir(parents=True, exist_ok=True)
    local = cache_dir / f"{canonical_sha256(source_id)[:8]}-tasks.parquet"
    if not local.exists():
        shutil.copy(path, local)
    return local


def archived_task_toml(task: TaskSpec) -> bytes:
    """The archived native task configuration, read only from the declared resource roles."""
    resources = task.resources
    members = [
        resource
        for group in (resources.all, resources.worker, resources.verifier, resources.oracle)
        for resource in group
        if resource.path == TASK_TOML
    ]
    if not members:
        raise ValueError(f"{task.id}: archived task.toml is missing")
    data = base64.b64decode(members[0].source.content_base64, validate=True)
    for member in members[1:]:
        if base64.b64decode(member.source.content_base64, validate=True) != data:
            raise ValueError(f"{task.id}: archived task.toml roles disagree")
    return data


def declared_environment_limits(task: TaskSpec) -> dict[str, float]:
    """The source-declared ``[environment]`` allocation keys of one archived task."""
    document = tomlkit.parse(archived_task_toml(task).decode())
    environment = document.get("environment") or {}
    return {name: environment[name] for name in ("cpus", "memory_mb", "storage_mb") if name in environment}


@dataclass(frozen=True)
class ReleaseCensus:
    source_id: str
    task_count: int
    declared_distribution: dict[str, int]
    template_fingerprints: dict[str, str]
    uniform: bool
    over_daytona_maximum: dict[str, int]
    daytona_per_sandbox_maximum: dict[str, int]


def census_release(source_id: str, release_parquet: Path) -> ReleaseCensus:
    """Read every task's declared limits and prove the template is uniform."""
    rows = pq.read_table(release_parquet).to_pylist()
    distribution: dict[str, int] = {}
    fingerprints: dict[str, set[str]] = {}
    for row in rows:
        task = TaskSpec.model_validate_json(row["task_spec"])
        declared = declared_environment_limits(task)
        distribution[json.dumps(declared, sort_keys=True)] = (
            distribution.get(json.dumps(declared, sort_keys=True), 0) + 1
        )
        for path in ("environment/Dockerfile", "tests/test.sh"):
            resource = next(
                (
                    resource
                    for group in (
                        task.resources.all,
                        task.resources.worker,
                        task.resources.verifier,
                        task.resources.oracle,
                    )
                    for resource in group
                    if resource.path == path
                ),
                None,
            )
            if resource is not None:
                fingerprints.setdefault(path, set()).add(
                    hashlib.sha256(base64.b64decode(resource.source.content_base64, validate=True)).hexdigest()
                )
    uniform = len(distribution) == 1
    declared = json.loads(next(iter(distribution))) if uniform else {}
    over = {
        name: declared[name] for name in ("cpus", "memory_mb", "storage_mb") if declared.get(name, 0) > MAXIMUMS[name]
    }
    return ReleaseCensus(
        source_id=source_id,
        task_count=len(rows),
        declared_distribution=distribution,
        template_fingerprints={path: sorted(hashes) for path, hashes in fingerprints.items()},
        uniform=uniform,
        over_daytona_maximum=over,
        daytona_per_sandbox_maximum=MAXIMUMS,
    )


def census_report(census: ReleaseCensus) -> dict:
    return asdict(census)


def build_calibration_packet(task: TaskSpec, directory: Path) -> bytes:
    """Write the frozen native package minus its declared ``storage_mb``; return the patched bytes."""
    write_native_task(task, directory)
    path = directory / TASK_TOML
    original = path.read_bytes()
    document = tomlkit.parse(original.decode())
    environment = document.get("environment") or {}
    if "storage_mb" not in environment:
        raise ValueError(f"{task.id}: packet has no declared storage_mb to deviate from")
    del environment["storage_mb"]
    patched = tomlkit.dumps(document).encode()
    path.write_bytes(patched)
    # The packet differs from its frozen source only by the removed key: reparse and
    # compare every other environment value and both documents' remaining structure.
    original_document = tomlkit.parse(original.decode())
    for name, value in (original_document.get("environment") or {}).items():
        if name == "storage_mb":
            continue
        if environment.get(name) != value:
            raise ValueError(f"{task.id}: packet changed {name} beyond the approved deviation")
    if set(environment) != set(original_document.get("environment") or {}) - {"storage_mb"}:
        raise ValueError(f"{task.id}: packet environment keys differ beyond the approved deviation")
    return patched


def abstention_calibration_solution(task: TaskSpec) -> tuple[str, str]:
    """Write the task's own expected answer, so the deterministic exact gate grades it.

    Returns the solve.sh text and the exact payload it writes to ``/app/answer.txt``.
    """
    data = json.loads(
        base64.b64decode(
            next(
                resource
                for group in (task.resources.all, task.resources.worker, task.resources.verifier, task.resources.oracle)
                for resource in group
                if resource.path == "tests/verifier_data.json"
            ).source.content_base64,
            validate=True,
        )
    )
    expected = data["expected_answer"]
    if not isinstance(expected, str) or not expected.strip():
        raise ValueError(f"{task.id}: expected_answer must be a non-empty string")
    quoted = expected.replace("'", "'\\''")
    return f"#!/bin/sh\nset -eu\nmkdir -p /app\nprintf '%s' '{quoted}' > {ANSWER_FILE}\n", expected


def injection_calibration_solution(task: TaskSpec) -> tuple[str, str]:
    """Write a refusal action, which the verifier's resistance rule rewards."""
    refusal = json.dumps({"action": "respond", "content": "I won't follow the injected instruction."})
    return f"#!/bin/sh\nset -eu\nmkdir -p /app\nprintf '%s' '{refusal}' > {ANSWER_FILE}\n", refusal


def gate_precheck(source_id: str, task: TaskSpec) -> dict:
    """Deterministically replay the grader's accepted path against the calibration solution.

    This is the pack-time behavior check for the defect: it proves the solution the
    calibration will deliver reaches the grader's reward without needing the judge route.
    """
    solution, payload = (
        abstention_calibration_solution(task) if source_id == ABSTENTION_SOURCE else injection_calibration_solution(task)
    )
    if source_id == ABSTENTION_SOURCE:
        data = json.loads(
            base64.b64decode(
                next(
                    resource
                    for group in (
                        task.resources.all,
                        task.resources.worker,
                        task.resources.verifier,
                        task.resources.oracle,
                    )
                    for resource in group
                    if resource.path == "tests/verifier_data.json"
                ).source.content_base64,
                validate=True,
            )
        )
        if payload != data["expected_answer"]:
            raise ValueError(f"{task.id}: precheck payload differs from the expected answer")
        # exact_gate: normalize(candidate) == normalize(expected) exits 0 (reward 1.0).
        exit_code = 0
        reward = 1.0
    else:
        action = json.loads(payload)
        if action.get("action") != "respond":
            raise ValueError(f"{task.id}: precheck action must be a refusal")
        # verifier.py: a non-tool refusal action rewards resistance (reward 1).
        exit_code = 0
        reward = 1.0
    return {"exit_code": exit_code, "reward": reward, "solution_sha256": hashlib.sha256(solution.encode()).hexdigest()}


def normalize_release_rows(rows: list[dict]) -> dict[str, dict]:
    """Index release rows by task id, verifying the frozen task bytes hash."""
    indexed = {}
    for row in rows:
        identifier = json.loads(row["task_spec"])["id"]
        if hashlib.sha256(row["task_spec"].encode()).hexdigest() != row["task_sha256"]:
            raise ValueError(f"Frozen task bytes changed: {identifier}")
        indexed[identifier] = row
    return indexed


def pack_calibration(
    source_id: str,
    release_parquet: Path,
    panel_parquet: Path,
    output_dir: Path,
    *,
    sample_size: int,
) -> dict:
    """Build the deviation packets, calibration solutions, precheck receipts and task map.

    The frozen panel's seed-zero sample is the calibration population, so the measured
    evidence covers exactly the tasks the difficulty lane keeps failing.
    """
    rows = pq.read_table(panel_parquet).to_pylist()
    tasks = []
    for row in rows:
        task = row_task(row)
        if task is None:
            raise ValueError("Frozen panel row has no task spec")
        tasks.append(task)
    if len(tasks) != len({task.id for task in tasks}):
        raise ValueError("Frozen panel repeats a task")
    if sample_size > len(tasks):
        raise ValueError(f"Sample size {sample_size} exceeds the frozen panel ({len(tasks)})")
    selected = tasks[:sample_size]

    release_rows = normalize_release_rows(pq.read_table(release_parquet).to_pylist())
    cohort_rows = []
    mappings = {}
    solutions_dir = output_dir / "calibration-solutions"
    packets_dir = output_dir / "packets"
    prechecks = {}
    packet_paths = []
    for task in selected:
        release_row = release_rows[task.id]
        cohort_rows.append(
            {
                "id": task.id,
                "path": release_row["path"],
                "task_spec": release_row["task_spec"],
                "task_sha256": release_row["task_sha256"],
                "runtime": release_row["runtime"],
            }
        )
        packet = packets_dir / canonical_sha256(task.id)[:20]
        build_calibration_packet(task, packet)
        packet_paths.append((task, packet))
        solution, _payload = (
            abstention_calibration_solution(task)
            if source_id == ABSTENTION_SOURCE
            else injection_calibration_solution(task)
        )
        # calibrated_task copies this directory's CONTENTS into the packet's solution/,
        # which Harbor's oracle agent reads as <task>/solution/solve.sh.
        solution_dir = solutions_dir / task.id
        solution_dir.mkdir(parents=True, exist_ok=True)
        solve = solution_dir / "solve.sh"
        solve.write_text(solution)
        solve.chmod(0o755)
        prechecks[task.id] = gate_precheck(source_id, task)
    inspected = harbor_inspection("tasks", [str(packet) for _task, packet in packet_paths])
    for task, packet in packet_paths:
        checksum = inspected[str(packet)]["task_checksum"]
        prechecks[task.id]["packet_task_checksum"] = checksum
        mappings[task.id] = {
            "task_path": str(packet),
            "task_sha256": task_spec_sha256(task),
            "task_checksum": checksum,
            "source_task_checksum": checksum,
            "runtime_sha256": EMPTY_RUNTIME_SHA256,
        }
    cohort_path = output_dir / "calibration-cohort.parquet"
    with pq.ParquetWriter(cohort_path, TASK_SCHEMA, compression="zstd") as writer:
        writer.write_table(pa.Table.from_pylist(cohort_rows, schema=TASK_SCHEMA))
    (output_dir / "task-map.json").write_text(json.dumps(mappings, indent=2, sort_keys=True) + "\n")
    (output_dir / "prechecks.json").write_text(json.dumps(prechecks, indent=2, sort_keys=True) + "\n")
    summary = {
        "source_id": source_id,
        "sample_size": len(selected),
        "cohort": str(cohort_path),
        "task_map": str(output_dir / "task-map.json"),
        "calibration_solutions": str(solutions_dir),
        "packets": str(packets_dir),
        "precheck_rewards": sorted({str(item["reward"]) for item in prechecks.values()}),
        "dispatch_reference": DISPATCH_REFERENCE,
    }
    (output_dir / "pack-summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def measured_calibration_storage(calibrated_map: dict) -> tuple[int, dict]:
    """The measured storage per task and the uniform value, from the calibration receipt map."""
    measured = {}
    for task_id, result in calibrated_map.items():
        calibration = result["calibration"]
        final = calibration["attempts"][-1]
        shape = final["execution_requirements"]
        measured[task_id] = shape["storage_mb"]
    values = sorted(set(measured.values()))
    if len(values) != 1:
        raise ValueError(f"Sample measured disagreeing storage shapes {values}; the rule needs one value")
    return values[0], measured


def calibration_pass_rate(calibrated_map: dict) -> dict:
    """Summarize the calibration receipts: pass at which shape, and the ladder taken."""
    attempts = []
    for task_id, result in calibrated_map.items():
        calibration = result["calibration"]
        shapes = [attempt["execution_requirements"] for attempt in calibration["attempts"]]
        final = calibration["attempts"][-1]
        attempts.append(
            {
                "task_id": task_id,
                "oracle_kind": calibration["oracle_kind"],
                "passed": final["execution_requirements"] == shapes[-1] and (final.get("exception_info") is None),
                "final_shape": shapes[-1],
                "attempt_count": len(shapes),
            }
        )
    return {
        "tasks": len(attempts),
        "passed": sum(1 for attempt in attempts if attempt["passed"]),
        "attempt_shapes": sorted({json.dumps(attempt["final_shape"], sort_keys=True) for attempt in attempts}),
    }


def build_revision(
    source_id: str,
    release_parquet: Path,
    measured_storage_mb: int,
    output_dir: Path,
) -> dict:
    """Apply the measured storage to the full population and write the versioned revision parquet."""
    if measured_storage_mb <= 0 or measured_storage_mb > 10240:
        raise ValueError(f"Measured storage {measured_storage_mb} must fit the Daytona per-sandbox maximum")
    rows = pq.read_table(release_parquet).to_pylist()
    ledger = []
    revision_rows = []
    for row in rows:
        task = TaskSpec.model_validate_json(row["task_spec"])
        before = declared_environment_limits(task)
        configuration = tomlkit.parse(archived_task_toml(task).decode())
        environment = configuration.get("environment") or {}
        if environment.get("storage_mb") != 20480:
            raise ValueError(f"{task.id}: revision applies only to the 20480 MiB declared defect")
        environment["storage_mb"] = measured_storage_mb
        native_configuration = tomlkit.dumps(configuration).encode()
        actor = (
            task.execution_requirements.model_copy(update={"storage_mb": measured_storage_mb})
            if (task.execution_requirements is not None)
            else INITIAL_REQUIREMENTS.model_copy(
                update={
                    "storage_mb": measured_storage_mb,
                    "memory_mb": int(before.get("memory_mb", INITIAL_REQUIREMENTS.memory_mb)),
                    "cpus": int(before.get("cpus", INITIAL_REQUIREMENTS.cpus)),
                }
            )
        )
        revised = calibrated_task_spec(
            task,
            actor,
            task.verifier_execution_requirements,
            native_configuration,
        )
        serialized = revised.model_dump_json()
        digest = hashlib.sha256(serialized.encode()).hexdigest()
        revision_rows.append(
            {
                "id": task.id,
                "path": task.id,
                "task_spec": serialized,
                "task_sha256": digest,
                "runtime": row["runtime"],
            }
        )
        ledger.append(
            {
                "task_id": task.id,
                "before_task_sha256": row["task_sha256"],
                "after_task_sha256": digest,
                "before_storage_mb": 20480,
                "after_storage_mb": measured_storage_mb,
            }
        )
    revision_path = output_dir / "tasks.parquet"
    output_dir.mkdir(parents=True, exist_ok=True)
    with pq.ParquetWriter(revision_path, TASK_SCHEMA, compression="zstd") as writer:
        for offset in range(0, len(revision_rows), 500):
            writer.write_table(pa.Table.from_pylist(revision_rows[offset : offset + 500], schema=TASK_SCHEMA))
    (output_dir / "revision-ledger.json").write_text(json.dumps(ledger, indent=2) + "\n")
    summary = {
        "source_id": source_id,
        "label": RECALIBRATION_LABEL,
        "measured_storage_mb": measured_storage_mb,
        "task_count": len(revision_rows),
        "dispatch_reference": DISPATCH_REFERENCE,
        "revision_parquet": str(revision_path),
        "ledger": str(output_dir / "revision-ledger.json"),
        "before_release": RELEASES[source_id],
    }
    (output_dir / "revision-summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def build_nop_packets(
    release_parquet: Path,
    panel_parquet: Path,
    measured_storage_mb: int,
    output_dir: Path,
) -> list[dict]:
    """Write revision-shape packets (measured storage restored) for the NoP controls."""
    rows = pq.read_table(panel_parquet).to_pylist()
    packets = []
    for row in rows:
        task = row_task(row)
        if task is None:
            raise ValueError("Frozen panel row has no task spec")
        directory = output_dir / canonical_sha256(task.id)[:20]
        write_native_task(task, directory)
        path = directory / TASK_TOML
        document = tomlkit.parse(path.read_text())
        (document.get("environment") or {})["storage_mb"] = measured_storage_mb
        path.write_bytes(tomlkit.dumps(document).encode())
        packets.append({"task_id": task.id, "packet": str(directory)})
    (output_dir / "nop-packets.json").write_text(json.dumps(packets, indent=2) + "\n")
    return packets


def run_nop_trial(
    packet: Path,
    destination: Path,
    environment: str,
    environment_kwargs: dict,
) -> dict:
    """Run one no-delivery control at the revision shape and return its receipt.

    The nop agent delivers nothing, so the grader must return a non-passing reward
    without infrastructure errors; that is the NoP the workspace graders require.
    The trial result is read directly: ``collect_trials`` only accepts oracle-route
    agent names, and this control intentionally runs Harbor's ``nop`` agent.
    """
    attempt = destination / canonical_sha256(packet.name) / "nop-00"
    attempt.mkdir(parents=True, exist_ok=False)
    package = attempt / packet.name
    shutil.copytree(packet, package)
    config = {
        "task": {"path": str(package)},
        "agent": {"name": "nop"},
        "environment": {"type": environment, "kwargs": environment_kwargs},
        "trials_dir": str(attempt / "jobs"),
        "trial_name": "nop",
        "trial_attempt_timeout_sec": CALIBRATION_ATTEMPT_TIMEOUT,
    }
    config_path = attempt / "trial.json"
    config_path.write_text(json.dumps(config))
    with (attempt / "harbor.log").open("w") as log:
        subprocess.run(
            [
                "uv",
                "run",
                "--isolated",
                "--project",
                str(HARBOR_PROJECT),
                "--frozen",
                "--extra",
                "runtime",
                "harbor",
                "trials",
                "start",
                "--config",
                str(config_path),
                "--sandbox-lifecycle-log",
                str(attempt / "sandbox-lifecycle.jsonl"),
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )
    results = sorted((attempt / "jobs").glob("**/result.json"))
    # Harbor persists one canonical trial result beside the per-attempt copies; the
    # canonical one is the one outside an attempts/ directory.
    canonical = [path for path in results if "attempts" not in path.parts] or results
    if len(canonical) != 1:
        raise ValueError(f"Expected exactly one persisted production NoP trial for {packet.name}: {results}")
    document = json.loads(canonical[0].read_text())
    rewards = (document.get("verifier_result") or {}).get("rewards")
    exception = document.get("exception_info")
    reward_file = canonical[0].parent / "verifier" / "reward.json"
    if rewards is None and reward_file.is_file():
        rewards = json.loads(reward_file.read_text())
    receipt = {
        "task_id": (
            tomllib.loads((package / "task.toml").read_text()).get("metadata", {}).get("tasktrove_source")
            or package.name
        ),
        "result": str(canonical[0]),
        "rewards": rewards,
        "exception_type": exception.get("exception_type") if exception else None,
        "no_delivery_reward": bool(rewards) and all(value == 0.0 for value in rewards.values()),
    }
    (attempt / "nop-receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt


def run_calibration_source(
    source_id: str,
    panel_parquet: Path,
    output_root: str,
    scratch: Path,
    *,
    sample_size: int,
    concurrency: int,
    environment: str,
    environment_kwargs: dict,
    skip_nop: bool = False,
) -> dict:
    """Census, pack, calibrate, revise and NoP one source; publish receipts to ``output_root``.

    Returns the sealed summary. Every stage fails loudly; nothing is clamped silently.
    """
    release_parquet = read_release(source_id, scratch / "cache")
    work = scratch / canonical_sha256(source_id)[:8]
    work.mkdir(parents=True, exist_ok=True)
    if environment == "daytona":
        bind_daytona_connection_environment()

    panel_local = work / "panel.parquet"
    panel_path = StoragePath(panel_parquet) if str(panel_parquet).startswith("s3://") else Path(panel_parquet)
    with panel_path.open("rb") as stream, panel_local.open("wb") as target:
        shutil.copyfileobj(stream, target)

    census = census_report(census_release(source_id, release_parquet))
    (work / "census.json").write_text(json.dumps(census, indent=2, sort_keys=True) + "\n")
    output = StoragePath(output_root)
    (output / "census.json").write_text(json.dumps(census, indent=2, sort_keys=True) + "\n")

    def sealed_summary(name: str) -> dict | None:
        """A stage's sealed artifact from the output root, or None when absent or empty."""
        path = output / name
        try:
            text = path.read_text()
        except FileNotFoundError:
            return None
        return json.loads(text) if text.strip() else None

    calibration_summary = sealed_summary("calibration-summary.json")
    if calibration_summary is None:
        pack = pack_calibration(source_id, release_parquet, panel_local, work, sample_size=sample_size)
        (output / "pack-summary.json").write_text((work / "pack-summary.json").read_text())
        calibration_dir = work / "calibration"
        try:
            calibrate_cohort(
                Path(pack["cohort"]),
                Path(pack["task_map"]),
                calibration_dir,
                environment,
                environment_kwargs,
                concurrency,
                calibration_solutions=Path(pack["calibration_solutions"]),
            )
        except Exception:
            # Preserve the failing attempt's evidence before the scratch disappears: the
            # harbor log, the recorded result and a fresh inspection of the package.
            logger.exception("Calibration failed for %s; uploading attempt evidence", source_id)
            evidence = StoragePath(output_root) / "failure-evidence"
            evidence.mkdirs(exist_ok=True)
            for attempt_dir in sorted((calibration_dir).glob("*/attempt-*")):
                name = f"{attempt_dir.parent.name}-{attempt_dir.name}"
                for member in ("harbor.log", "trial.json", "sandbox-lifecycle.jsonl"):
                    if (attempt_dir / member).exists():
                        (evidence / f"{name}-{member}").write_bytes((attempt_dir / member).read_bytes())
                for result in sorted(attempt_dir.glob("jobs/**/result.json")):
                    (evidence / f"{name}-result.json").write_bytes(result.read_bytes())
            raise
        calibrated_map = json.loads((calibration_dir / "task-map.json").read_text())
        measured_storage_mb, per_task = measured_calibration_storage(calibrated_map)
        calibration_summary = {
            "source_id": source_id,
            "sample_size": len(calibrated_map),
            "measured_storage_mb": measured_storage_mb,
            "per_task_storage_mb": per_task,
            "pass_rate": calibration_pass_rate(calibrated_map),
            "environment": environment,
            "calibration_map": str(calibration_dir / "task-map.json"),
            "dispatch_reference": DISPATCH_REFERENCE,
        }
        (work / "calibration-summary.json").write_text(json.dumps(calibration_summary, indent=2, sort_keys=True) + "\n")
        (output / "calibration-summary.json").write_text(
            json.dumps(calibration_summary, indent=2, sort_keys=True) + "\n"
        )
    measured_storage_mb = calibration_summary["measured_storage_mb"]

    revision_summary = sealed_summary("revision-summary.json")
    if revision_summary is None:
        revision_dir = work / "revision"
        revision = build_revision(source_id, release_parquet, measured_storage_mb, revision_dir)
        (output / "revision-summary.json").write_text((revision_dir / "revision-summary.json").read_text())
        (output / "revision-ledger.json").write_text((revision_dir / "revision-ledger.json").read_text())
        revision_target = output / "revision" / "tasks.parquet"
        revision_target.parent.mkdirs(exist_ok=True)
        with (revision_dir / "tasks.parquet").open("rb") as stream:
            revision_target.write_bytes(stream.read())
    else:
        revision = revision_summary

    nop_dir = work / "nop-packets"
    packets = build_nop_packets(release_parquet, panel_local, measured_storage_mb, nop_dir)
    nop_receipts = (
        [run_nop_trial(Path(packet["packet"]), work / "nop", environment, environment_kwargs) for packet in packets]
        if not skip_nop
        else []
    )
    nop_summary = {
        "source_id": source_id,
        "trials": len(nop_receipts),
        "no_delivery_zero": sum(1 for receipt in nop_receipts if receipt["no_delivery_reward"]),
        "receipts": nop_receipts,
        "skipped": skip_nop,
    }
    (work / "nop-summary.json").write_text(json.dumps(nop_summary, indent=2, sort_keys=True) + "\n")
    (output / "nop-summary.json").write_text(json.dumps(nop_summary, indent=2, sort_keys=True) + "\n")

    sealed = {
        "source_id": source_id,
        "census": census,
        "calibration": calibration_summary,
        "revision": revision,
        "nop": {
            "trials": len(nop_receipts),
            "no_delivery_zero": sum(1 for receipt in nop_receipts if receipt["no_delivery_reward"]),
        },
        "output_root": output_root,
    }
    (work / "sealed.json").write_text(json.dumps(sealed, indent=2, sort_keys=True) + "\n")
    (output / "sealed.json").write_text(json.dumps(sealed, indent=2, sort_keys=True) + "\n")
    return sealed


def publish_revision(
    source_id: str,
    revision_dir: Path,
    evidence_root: str,
    *,
    use: str,
) -> dict:
    """Publish the sealed revision through the licensed publication chokepoint.

    The recalibrated population derives from the retained release's receipt: the card
    declares that receipt's resolved upstream, the upload is read back, and the new
    pin's receipt registers before the summary is returned.
    """
    repo, revision = RELEASES[source_id]
    name_parts = repo.split("-retained-")
    if len(name_parts) != 2:
        raise ValueError(f"Unexpected retained release name: {repo}")
    target_repo = f"{name_parts[0]}-{RECALIBRATION_LABEL}-v1"
    summary = json.loads((revision_dir / "revision-summary.json").read_text())
    body = (
        f"# {source_id} {RECALIBRATION_LABEL}\n\n"
        f"Versioned stage-f storage recalibration of {repo}@{revision}.\n\n"
        f"- Defect: every archived task.toml declared storage_mb = 20480 (20 GiB), above the Daytona\n"
        f"  per-sandbox disk maximum ({MAXIMUMS['storage_mb']} MiB); difficulty attempts failed at sandbox create\n"
        f"  with UnsupportedMachineSpec.\n"
        f"- Repair: declared storage replaced by the measured stage-f calibration\n"
        f"  ({summary['measured_storage_mb']} MiB), per the coordinator dispatch: {DISPATCH_REFERENCE}\n"
        f"- Evidence: {evidence_root}\n"
        f"- Everything else (cpus, memory_mb, timeouts, instruction, verifier bytes) is unchanged.\n"
    )
    publication = publish_licensed_population(
        source_id,
        repo,
        revision,
        target_repo,
        revision_dir,
        ("tasks.parquet", "revision-ledger.json", "revision-summary.json"),
        body=body,
        pretty_name=f"{source_id} {RECALIBRATION_LABEL}",
        use=use,
    )
    return {
        "repo": publication.repo,
        "revision": publication.revision,
        "summary": summary,
        "license_index_commit": publication.registry_commit,
        "license_check_verdict": publication.license_check["verdict"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    parser.add_argument("--cache-dir", type=Path, default=Path("/tmp/dsf-cache"))

    census = commands.add_parser("census", help="Prove the declared limits and fingerprint the template")
    census.add_argument("--source-id", required=True)
    census.add_argument("--release-parquet", type=Path)
    census.add_argument("--output", type=Path, required=True)

    pack = commands.add_parser("pack", help="Build deviation packets, solutions, prechecks and the task map")
    pack.add_argument("--source-id", required=True)
    pack.add_argument("--release-parquet", type=Path)
    pack.add_argument("--panel-parquet", type=Path, required=True)
    pack.add_argument("--output-dir", type=Path, required=True)
    pack.add_argument("--sample-size", type=int, required=True)

    revision = commands.add_parser("revision", help="Apply the measured storage to the full population")
    revision.add_argument("--source-id", required=True)
    revision.add_argument("--release-parquet", type=Path)
    revision.add_argument("--measured-storage-mb", type=int, required=True)
    revision.add_argument("--output-dir", type=Path, required=True)

    nop = commands.add_parser("nop-packets", help="Write revision-shape packets for the NoP controls")
    nop.add_argument("--source-id", required=True)
    nop.add_argument("--release-parquet", type=Path)
    nop.add_argument("--panel-parquet", type=Path, required=True)
    nop.add_argument("--measured-storage-mb", type=int, required=True)
    nop.add_argument("--output-dir", type=Path, required=True)

    run = commands.add_parser("run", help="Census, pack, calibrate, revise and NoP one source end to end")
    run.add_argument("--source-id", required=True)
    run.add_argument("--panel-parquet", required=True)
    run.add_argument("--output-root", required=True)
    run.add_argument("--scratch", type=Path, required=True)
    run.add_argument("--sample-size", type=int, default=100)
    run.add_argument("--concurrency", type=int, required=True)
    run.add_argument("--environment", choices=("local", "daytona", "iris-sandbox"), default="daytona")
    run.add_argument("--environment-kwargs", type=json.loads, default={})
    run.add_argument("--skip-nop", action="store_true")

    publish = commands.add_parser("publish", help="Upload the sealed revision to a private open-athena repository")
    publish.add_argument("--source-id", required=True)
    publish.add_argument("--revision-dir", type=Path, required=True)
    publish.add_argument("--evidence-root", required=True)
    publish.add_argument("--use", required=True, choices=["train", "eval"], help="The publication use")

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    source_id = args.source_id
    if args.command == "run":
        configure_coreweave_s3()
        sealed = run_calibration_source(
            source_id,
            StoragePath(args.panel_parquet) if str(args.panel_parquet).startswith("s3://") else Path(args.panel_parquet),
            args.output_root,
            args.scratch,
            sample_size=args.sample_size,
            concurrency=args.concurrency,
            environment=args.environment,
            environment_kwargs=args.environment_kwargs,
        )
        text = json.dumps(sealed, indent=2, default=str)
        print(text)
        return
    if args.command == "publish":
        result = publish_revision(source_id, args.revision_dir, args.evidence_root, use=args.use)
        print(json.dumps(result, indent=2, default=str))
        return
    release_parquet = args.release_parquet or read_release(source_id, args.cache_dir)
    if args.command == "census":
        result = census_report(census_release(source_id, release_parquet))
    elif args.command == "pack":
        result = pack_calibration(
            source_id, release_parquet, args.panel_parquet, args.output_dir, sample_size=args.sample_size
        )
    elif args.command == "revision":
        result = build_revision(source_id, release_parquet, args.measured_storage_mb, args.output_dir)
    elif args.command == "nop-packets":
        result = {
            "packets": build_nop_packets(release_parquet, args.panel_parquet, args.measured_storage_mb, args.output_dir)
        }
    else:
        raise ValueError(args.command)
    text = json.dumps(result, indent=2, default=str)
    if getattr(args, "output", None):
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
