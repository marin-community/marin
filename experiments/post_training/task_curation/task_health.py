# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Record an RL run's agent attempts and publish versioned task skip lists from the records.

``extract`` writes one row per attempt in the run's task health receipts. ``skip`` lists a
selection's training rows whose tasks repeatedly fail declared setup or verification without ever
grading, and every row of an environment where enough tasks do. A skip file is a selection recipe's
``exclusions`` input in ``task_digest`` format.
"""

import argparse
import hashlib
import io
import json
import math
import os
import re
import tomllib
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from types import MappingProxyType

import pyarrow as pa
import pyarrow.parquet as pq
from finestore.rl.run_records import LaunchState
from finestore.rl.task_health import ATTEMPT_RECEIPTS_DIRECTORY, AttemptReceipt, Phase
from marin.execution.artifact import validate_path_segment
from marin.execution.lazy import OUT, ArtifactStep, apply
from marin.rl.skyrl import read_skyrl_attempt
from marin.rl.skyrl_records import read_skyrl_launch_config
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath
from rigging.provenance import Provenance
from taskcompendium.grading_result import GradingFailure, Outcome
from taskcompendium.identity import canonical_sha256
from taskcompendium.models import TaskSpec
from taskcompendium.runtime.environment import environment_identity
from taskcompendium.runtime.resources import resource_bytes

from experiments.post_training.rl_compose import SELECTION_ROOT, artifact_prefix
from experiments.post_training.task_curation import CANONICAL_ARTIFACT_ROOT, publish_once
from experiments.post_training.task_curation.selection import (
    SELECTION_PREFIX,
    TASK_SPEC_COLUMN,
    TRAIN_FILENAME,
    with_agentic_tag,
)

MIN_FAILED_ATTEMPTS = 2
MIN_FAILED_TASKS_PER_ENVIRONMENT = 2

TASK_HEALTH_ROOT = str(StoragePath(CANONICAL_ARTIFACT_ROOT) / "rl-data/task-health")
ATTEMPTS_FILENAME = "attempts.parquet"
SKIP_FILENAME = "skip.parquet"
MANIFEST_FILENAME = "manifest.json"
# The installed-agent runtime grades this reward key unless the launch names another.
DEFAULT_REWARD_KEY = "reward"
ERROR_CHARS = 1024
TAIL_BYTES = 4096
READ_THREADS = 32
NO_TRIAL_RESULT = "attempt has no trial result.json"


VERIFIER_TASK_FAILURES = frozenset(
    {"RewardFileNotFoundError", "RewardFileEmptyError", "VerifierOutputParseError", "VerifierRuntimeError"}
)
NATIVE_VERIFIER_TASK_FAILURES = frozenset(
    {
        GradingFailure.MISSING_REWARD,
        GradingFailure.EMPTY_REWARD,
        GradingFailure.INVALID_REWARD,
        GradingFailure.GRADER_EXIT,
    }
)
HARBOR_SETUP_COMMAND_FAILURE = re.compile(r"^Declared task setup command \d+ exited -?[1-9]\d*\b")
# Only ungraded attempts carry the output tails and the attempt's address.
ATTEMPT_COLUMNS = (
    "run selection attempt_id task_id task_digest outcome grading_failure exception_type error phase "
    "docker_image setup_sha256 setup_completed setup_exit_code setup_stdout setup_stderr "
    "verifier_stdout verifier_stderr address"
).split()
ATTEMPT_COLUMN_TYPES = MappingProxyType({"setup_completed": pa.bool_(), "setup_exit_code": pa.int64()})
ATTEMPT_SCHEMA = pa.schema([(name, ATTEMPT_COLUMN_TYPES.get(name, pa.string())) for name in ATTEMPT_COLUMNS])
SKIP_COLUMNS = (TASK_SPEC_COLUMN, "task_id", "unit", "attempts", "failed_attempts", "error", "example")
SKIP_SCHEMA = pa.schema([(name, pa.int64() if name.endswith("attempts") else pa.string()) for name in SKIP_COLUMNS])


def _text(path: StoragePath) -> str | None:
    try:
        return path.read_text()
    except FileNotFoundError:
        return None


def _json(path: StoragePath) -> dict | None:
    text = _text(path)
    return json.loads(text) if text is not None else None


def _tail(path: StoragePath) -> bytes:
    try:
        with path.open("rb") as stream:
            length = stream.seek(0, io.SEEK_END)
            stream.seek(max(0, length - TAIL_BYTES))
            return stream.read(TAIL_BYTES)
    except FileNotFoundError:
        return b""


def _docker_image(task_toml: str) -> str | None:
    return tomllib.loads(task_toml).get("environment", {}).get("docker_image")


def _setup_sha256(command_hashes: list[str]) -> str | None:
    """The SHA-256 of each declared setup command, space-separated, as the setup receipt lists them."""
    return " ".join(command_hashes) or None


def _environment(task: TaskSpec) -> tuple[str | None, str | None]:
    """A training row's image and setup, encoded as an attempt row records them."""
    image, setup = environment_identity(task.environment_requirements)
    if image is not None:
        return image, setup
    configs = [resource for resource in task.resources.verifier if resource.path == "task.toml"]
    image = _docker_image(resource_bytes(configs[0]).decode()) if configs else None
    return image, setup


def _tasks(data: bytes) -> dict[str, tuple[TaskSpec, str]]:
    """A task file's rows keyed by the digest a selection's task manifest records."""
    tasks = {}
    for raw in pq.read_table(io.BytesIO(data), columns=[TASK_SPEC_COLUMN])[TASK_SPEC_COLUMN].to_pylist():
        task = TaskSpec.model_validate_json(raw)
        tasks[canonical_sha256(task.model_dump(mode="json"))] = task, raw
    return tasks


def _grade(result: dict, reward_key: str) -> tuple[Outcome, GradingFailure | None]:
    """The rule of SkyRL's ``trial_grade``, which Marin's environment cannot import, on a stored result."""
    rewards = (result.get("verifier_result") or {}).get("rewards") or {}
    if reward_key not in rewards:
        return Outcome.INFRA_ERROR, GradingFailure.MISSING_REWARD
    score = float(rewards[reward_key])
    if not math.isfinite(score) or not 0 <= score <= 1:
        return Outcome.INFRA_ERROR, GradingFailure.INVALID_REWARD
    return Outcome.GRADED, None


def _phase(result: dict | None, receipt: dict | None) -> Phase:
    """Harbor prepares the environment, runs declared setup once it is ready, then the agent and verifier."""
    timings = result or {}
    starts = (
        (Phase.VERIFIER, timings.get("verifier")),
        (Phase.AGENT, timings.get("agent_setup")),
        (Phase.TASK_SETUP, receipt),
        (Phase.ENVIRONMENT, timings.get("environment_setup")),
    )
    return next((phase for phase, record in starts if record and record.get("started_at")), Phase.NONE)


def _attempt_row(attempt: StoragePath, reward_key: str) -> dict | None:
    """Read one attempt in the layout SkyRL's installed-agent runtime and Harbor's trial paths write."""
    identity = _json(attempt / "identity.json")
    if identity is None:
        return None
    trial, setup = attempt / "trials" / identity["attempt_id"], attempt / "engine" / "task-setup"
    result, receipt = _json(trial / "result.json"), _json(setup / "receipt.json")
    task_toml = _text(attempt / "task" / "task.toml")
    exception = (result or {}).get("exception_info") or {}
    outcome, failure = _grade(result, reward_key) if result is not None else (None, None)
    error = exception.get("exception_message") if result is not None else NO_TRIAL_RESULT
    row = {
        "attempt_id": identity["attempt_id"],
        "task_id": identity["task_id"],
        "outcome": outcome,
        "grading_failure": failure,
        "exception_type": exception.get("exception_type"),
        "error": error[:ERROR_CHARS] if error else None,
        "phase": _phase(result, receipt),
        "docker_image": _docker_image(task_toml) if task_toml is not None else None,
        "setup_sha256": _setup_sha256(receipt["command_sha256"]) if receipt else None,
        "setup_completed": receipt["completed"] if receipt else None,
    }
    if outcome == Outcome.GRADED:
        return row
    commands = len(receipt["command_sha256"]) if receipt else 0
    tails = {
        f"setup_{kind}": b"".join(_tail(setup / f"{index}.{kind}") for index in range(commands))
        for kind in ("stdout", "stderr")
    }
    for kind in ("stdout", "stderr"):
        tails[f"verifier_{kind}"] = _tail(trial / "verifier" / f"test-{kind}.txt")
    return {
        **row,
        **{key: data[-TAIL_BYTES:].decode("utf-8", errors="ignore") for key, data in tails.items() if data},
        "address": str(attempt),
    }


def _parquet(table: pa.Table) -> bytes:
    buffer = io.BytesIO()
    pq.write_table(table, buffer)
    return buffer.getvalue()


def extract_attempts(run_output: StoragePath, run: str, output: StoragePath) -> StoragePath:
    """Publish one row per stored agent attempt, graded or not."""
    return _extract_attempts(read_skyrl_launch_config(str(run_output)), run, output)


def _native_attempt_rows(export_path: str) -> list[dict]:
    root = StoragePath(export_path) / ATTEMPT_RECEIPTS_DIRECTORY
    if not root.exists():
        return []
    rows: dict[str, tuple[AttemptReceipt, str | None]] = {}

    def read_batch(path: StoragePath) -> list[AttemptReceipt]:
        return [AttemptReceipt(**json.loads(line)) for line in path.read_text().splitlines()]

    paths = sorted((path for path in root.ls() if path.name.endswith(".jsonl")), key=str)
    with ThreadPoolExecutor(READ_THREADS) as pool:
        for path, batch in zip(paths, pool.map(read_batch, paths), strict=True):
            for receipt in batch:
                address = str(path) if receipt.outcome != Outcome.GRADED else None
                prior, _ = rows.setdefault(receipt.attempt_id, (receipt, address))
                if prior != receipt:
                    raise ValueError(f"Conflicting records for attempt {receipt.attempt_id}")
    return [{**asdict(receipt), "address": address} for receipt, address in rows.values()]


def _historical_attempt_rows(agent: dict, source: dict) -> list[dict]:
    tasks = _tasks((StoragePath(source["uri"]) / source["relative_path"]).read_bytes())
    digests = {task.id: digest for digest, (task, _) in tasks.items()}
    reward_key = agent.get("trial_reward_key", DEFAULT_REWARD_KEY)
    attempts = sorted(StoragePath(agent["output_path"]).ls(), key=str)
    with ThreadPoolExecutor(READ_THREADS) as pool:
        return [
            {**row, "task_digest": digests.get(row["task_id"])}
            for row in pool.map(lambda attempt: _attempt_row(attempt, reward_key), attempts)
            if row is not None
        ]


def _extract_attempts(config: dict, run: str, output: StoragePath) -> StoragePath:
    skyrl = config["skyrl"]
    (source,) = config["inputs"]["train_data"]
    selection = source["identity"].rpartition(":")[0].removeprefix(f"{SELECTION_PREFIX}/")
    if agent := skyrl.get("terminal_bench_config", {}).get("installed_agent"):
        rows = _historical_attempt_rows(agent, source)
    else:
        rows = _native_attempt_rows(skyrl["trainer"]["export_path"])
    table = pa.Table.from_pylist(
        [{**row, "run": run, "selection": selection} for row in rows],
        schema=ATTEMPT_SCHEMA,
    )
    publish_once(output, {ATTEMPTS_FILENAME: _parquet(table)})
    return output / ATTEMPTS_FILENAME


def publish_skip_list(
    selection: StoragePath,
    attempts: list[StoragePath],
    output: StoragePath,
    code_commit: str,
    release: StoragePath | None = None,
) -> dict:
    """Publish the selection's training rows whose tasks or environments fail setup or verification."""
    training = selection / TRAIN_FILENAME
    data = {str(path): path.read_bytes() for path in (training, *attempts, *([release] if release else []))}
    tasks = _tasks(data[str(training)])
    # An attempt of a task outside its run's training selection has no digest.
    unique = {}
    for path in dict.fromkeys(attempts):
        for row in pq.read_table(io.BytesIO(data[str(path)])).to_pylist():
            if row["task_digest"] is None:
                continue
            key = (row["run"], row["attempt_id"])
            row.setdefault("setup_exit_code", None)
            if unique.setdefault(key, row) != row:
                raise ValueError(f"Conflicting records for attempt {key}")
    rows = unique.values()
    graded = {row["task_digest"] for row in rows if row["outcome"] == Outcome.GRADED}
    attempted = Counter(row["task_digest"] for row in rows)
    failures: defaultdict[str, list[dict]] = defaultdict(list)
    for row in rows:
        if row["outcome"] != Outcome.INFRA_ERROR or row["task_digest"] in graded:
            continue
        setup_exit_code = row["setup_exit_code"]
        failed_setup = (
            row["phase"] == Phase.TASK_SETUP
            and row["setup_completed"] is False
            and (
                0 < setup_exit_code < 128
                if setup_exit_code is not None
                else HARBOR_SETUP_COMMAND_FAILURE.match(row["error"] or "") is not None
            )
        )
        failed_verifier = (
            row["phase"] == Phase.VERIFIER
            and (
                row["exception_type"] in VERIFIER_TASK_FAILURES
                or row["grading_failure"] in NATIVE_VERIFIER_TASK_FAILURES
            )
            and (row["verifier_stdout"] or row["verifier_stderr"])
        )
        if failed_setup or failed_verifier:
            failures[row["task_digest"]].append(row)
    broken = {digest: failed[0] for digest, failed in failures.items() if len(failed) >= MIN_FAILED_ATTEMPTS}
    distinct_tasks = {(row["docker_image"], row["setup_sha256"], row["task_id"]) for row in broken.values()}
    environments = Counter((image, setup) for image, setup, _ in distinct_tasks)
    broken_environments = {
        environment
        for environment, count in environments.items()
        if environment[0] is not None and count >= MIN_FAILED_TASKS_PER_ENVIRONMENT
    }
    released = {task.id: task for task, _ in _tasks(data[str(release)]).values()} if release else {}
    listed = []
    for digest, (task, raw) in tasks.items():
        unit = "task" if digest in broken else "environment" if _environment(task) in broken_environments else None
        if unit is None:
            continue
        replacement = released.get(task.id)
        if replacement is not None:
            if "agentic" in task.tags:
                replacement = with_agentic_tag(replacement)
            if canonical_sha256(replacement.model_dump(mode="json")) != digest:
                continue
        failed = failures.get(digest, [])
        lines = Counter(row["error"].partition("\n")[0] for row in failed if row["error"])
        listed.append(
            {
                TASK_SPEC_COLUMN: raw,
                "task_id": task.id,
                "unit": unit,
                "attempts": attempted[digest],
                "failed_attempts": len(failed),
                "error": lines.most_common(1)[0][0] if lines else None,
                "example": failed[0]["address"] if failed else None,
            }
        )
    if not listed:
        raise ValueError("No tasks meet the skip criteria")
    manifest = {
        "inputs": {uri: hashlib.sha256(contents).hexdigest() for uri, contents in data.items()},
        "min_failed_attempts": MIN_FAILED_ATTEMPTS,
        "min_failed_tasks_per_environment": MIN_FAILED_TASKS_PER_ENVIRONMENT,
        "code_commit": code_commit,
        "rows": len(listed),
    }
    publish_once(
        output,
        {
            SKIP_FILENAME: _parquet(pa.Table.from_pylist(listed, schema=SKIP_SCHEMA)),
            MANIFEST_FILENAME: (json.dumps(manifest, indent=2) + "\n").encode(),
        },
    )
    return manifest


def _name_version(value: str) -> tuple[str, str]:
    name, _, version = value.rpartition("@")
    validate_path_segment("name", name)
    validate_path_segment("version", version)
    return name, version


def _extract_stopped_run(run_output: str, run: str) -> StoragePath:
    """Snapshot attempt receipts after the run's current launch stops."""
    attempt = read_skyrl_attempt(run_output)
    if attempt is None or attempt.native.result.state not in {LaunchState.SUCCEEDED, LaunchState.FAILED}:
        raise ValueError(f"Task health extraction requires a stopped run: {run}")
    name, version = _name_version(run)
    location = canonical_sha256({"run_output": run_output, "attempt": attempt.attempt_id})[:16]
    output = StoragePath(TASK_HEALTH_ROOT) / "attempts" / name / version / location
    return _extract_attempts(attempt.native.config, run, output)


def _record_attempts(run_output: str, run: str, output: str) -> None:
    snapshot = _extract_stopped_run(run_output, run)
    publish_once(StoragePath(output), {"attempts.json": json.dumps({"attempts_uri": str(snapshot)}).encode()})


def task_health_step(run: ArtifactStep) -> ArtifactStep:
    """Collect one completed run's agent attempts."""
    return apply(
        f"rl-data/task-health/attempts/by-run/{run.name}",
        _record_attempts,
        version=run.version,
        run_output=run,
        run=f"{run.name}@{run.version}",
        output=OUT,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    extract = commands.add_parser("extract", help="record every stored agent attempt of a run")
    extract.add_argument("--cluster", required=True, help="Iris cluster whose storage prefix holds the run")
    extract.add_argument("--run", required=True, help="run artifact NAME@VERSION")
    skip = commands.add_parser("skip", help="publish a versioned skip list for a task selection")
    skip.add_argument("--selection", required=True, help="task selection NAME@VERSION")
    skip.add_argument("--attempts", nargs="+", required=True, help="attempts.parquet files from completed run snapshots")
    skip.add_argument("--version", required=True, help="skip list version, YYYY.MM.DD.N")
    skip.add_argument("--release", help="fixed task file; a listed task whose digest differs there leaves the list")
    args = parser.parse_args()
    configure_coreweave_s3()
    root = StoragePath(TASK_HEALTH_ROOT)
    if args.command == "extract":
        name, version = _name_version(args.run)
        prefix = os.environ.get("MARIN_PREFIX") or artifact_prefix(args.cluster)
        run_output = StoragePath(prefix) / name / version
        print(_extract_stopped_run(str(run_output), args.run))
        return
    provenance = Provenance.capture()
    if provenance.dirty or not provenance.base_commit:
        raise SystemExit("Publish from a clean commit so the manifest identifies the code")
    name, version = _name_version(args.selection)
    manifest = publish_skip_list(
        StoragePath(SELECTION_ROOT) / name / version,
        [StoragePath(address) for address in args.attempts],
        root / "skip" / name / args.version,
        provenance.base_commit,
        StoragePath(args.release) if args.release else None,
    )
    print(json.dumps(manifest))


if __name__ == "__main__":
    main()
