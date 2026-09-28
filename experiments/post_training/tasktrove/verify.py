# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Throw away converted tasks that fail a grading contract check.

Checks run in cost order:

- the spec parses and every referenced file exists;
- the Dockerfile installs the verifier and contains no old grader;
- the agent cannot see an expected value or solution;
- the mode-specific task shape is valid;
- an empty output scores zero;
- the expected output scores one; and
- a safe perturbation scores zero.

Output probes run the verifier in-process. Each rejection names its failed check in the ledger.
"""

import hashlib
import json
import tempfile
from collections.abc import Iterator
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from rigging.filesystem.storage_path import StoragePath
from tasktrove_verify.grade import Status, grade, local_output_path, negative_candidate, positive_candidate
from tasktrove_verify.modes.extract import collapse_whitespace
from tasktrove_verify.modes.ifeval import CONSTRAINTS
from tasktrove_verify.spec import (
    RUBRIC_CHECKLIST,
    RUBRIC_REFERENCE,
    RUBRICS,
    ExactSpec,
    GotestSpec,
    IfevalSpec,
    JsonSchemaSpec,
    JudgeSpec,
    JunitSpec,
    MathSpec,
    NumericSpec,
    PytestSpec,
    ReasoningGymSpec,
    ScriptSpec,
    Spec,
    StdioSpec,
    parse_spec,
)
from zephyr.context import ZephyrContext
from zephyr.dataset import Dataset

from experiments.post_training.tasktrove.convert import CONVERTED_GLOB, CONVERTED_SCHEMA
from experiments.post_training.tasktrove.converters.converted_task import ConvertStatus
from experiments.post_training.tasktrove.dataset import APPROX_SHARD_BYTES, WORKER_RESOURCES, WORKING_SHARDS
from experiments.post_training.tasktrove.task_format import (
    INSTALL_MARKER,
    OLD_GRADER_LINE,
    VERIFIER_TOML,
    VERIFY_TEST_SH,
)
from experiments.post_training.tasktrove.taskbinary import (
    DOCKERFILE,
    INSTRUCTION,
    SOLUTION_DIR,
    TASK_TOML,
    TEST_SH,
    TaskFiles,
    read_task_binary,
)

FILTERED_GLOB = "graded/*.parquet"
VERIFIED_STATUS = "verified:"
"""Status prefix of a row the grader checks rejected; the check name follows."""
REQUIRED_FILES = (INSTRUCTION, TASK_TOML, DOCKERFILE, TEST_SH, VERIFIER_TOML)
_MIN_LEAK_CHARS = 12
"""Expected values shorter than this are not checked against the instruction: a single letter or
small number appears in almost any prompt."""
_CAP_SEED = b"tasktrove-clean-cap-v1"


class Check(StrEnum):
    SPEC = "spec"
    DOCKERFILE = "dockerfile"
    GOLD_LEAK = "gold_leak"
    SHAPE = "shape"
    EMPTY = "empty"
    EXPECTED = "expected"
    PERTURBED = "perturbed"


class DedupStatus:
    DUPLICATE = "duplicate"
    CAPPED = "capped"


@dataclass(frozen=True)
class Rejection:
    check: Check
    detail: str


def cap_rank(path: str) -> str:
    return hashlib.sha256(_CAP_SEED + path.encode()).hexdigest()


def dedup_key(row: dict) -> tuple[str, str, str]:
    if row["status"] == ConvertStatus.CONVERTED:
        return ("instruction", row["source"], row["instruction_key"])
    return ("path", row["source"], row["path"])


def dropped(row: dict, status: str, detail: str) -> dict:
    return {**row, "status": status, "error": detail, "task_binary": None, "solution_binary": None}


def keep_first(_key: tuple, rows: Iterator[dict]) -> Iterator[dict]:
    first = next(rows)
    yield first
    for row in rows:
        yield dropped(row, DedupStatus.DUPLICATE, f"same instruction as {first['path']}")


def cap_source(rows: Iterator[dict], max_tasks: int) -> Iterator[dict]:
    kept = 0
    for row in rows:
        if row["status"] != ConvertStatus.CONVERTED:
            yield row
            continue
        kept += 1
        yield row if kept <= max_tasks else dropped(row, DedupStatus.CAPPED, f"beyond the {max_tasks}-task cap")


def _spec_paths(spec: Spec) -> list[str]:
    """Files under tests/ the spec refers to and that must ship in the binary."""
    if isinstance(spec, JsonSchemaSpec):
        return [spec.schema]
    if isinstance(spec, ReasoningGymSpec):
        return [spec.entry]
    if isinstance(spec, ScriptSpec):
        return [spec.path]
    if isinstance(spec, StdioSpec):
        return [spec.special_judge] if spec.special_judge else []
    if isinstance(spec, JudgeSpec):
        return [spec.context] if spec.context else []
    return list(spec.restore if isinstance(spec, PytestSpec | JunitSpec | GotestSpec) else ())


def check_spec(task: TaskFiles) -> tuple[Spec | None, Rejection | None]:
    missing = [p for p in REQUIRED_FILES if p not in task.files]
    if missing:
        return None, Rejection(Check.SPEC, f"missing {missing}")
    if task.text(TEST_SH) != VERIFY_TEST_SH:
        return None, Rejection(Check.SPEC, "test.sh is not the verify shim")
    try:
        spec = parse_spec(task.text(VERIFIER_TOML))
    except (ValueError, KeyError) as error:
        return None, Rejection(Check.SPEC, f"verifier.toml: {error}")
    absent = [p for p in _spec_paths(spec) if f"tests/{p}" not in task.files and not task.under(f"tests/{p}/")]
    if absent:
        return None, Rejection(Check.SPEC, f"spec references missing tests/ files {absent}")
    return spec, None


def check_dockerfile(task: TaskFiles) -> Rejection | None:
    text = task.text(DOCKERFILE)
    if INSTALL_MARKER not in text:
        return Rejection(Check.DOCKERFILE, "tool install block missing")
    for line in text.splitlines():
        lowered = line.lower()
        if OLD_GRADER_LINE.search(line):
            return Rejection(Check.DOCKERFILE, f"old grader dependency: {line.strip()[:120]}")
        if lowered.startswith("copy ") and " tests/" in f" {line}":
            source = line.split()[1]
            if source not in task.files and not task.under(source.rstrip("/") + "/"):
                return Rejection(Check.DOCKERFILE, f"COPY of a file not in the task: {source}")
    return None


def _expected_strings(spec: Spec) -> list[str]:
    if isinstance(spec, MathSpec | NumericSpec):
        return [str(spec.expected)]
    if isinstance(spec, ExactSpec):
        return list(spec.expected)
    if isinstance(spec, JudgeSpec):
        return list(spec.references)
    return []


def check_gold_leak(task: TaskFiles, spec: Spec) -> Rejection | None:
    if task.has_solution:
        return Rejection(Check.GOLD_LEAK, f"{SOLUTION_DIR} shipped inside the task binary")
    instruction = collapse_whitespace(task.text(INSTRUCTION)).lower()
    for value in _expected_strings(spec):
        needle = collapse_whitespace(value).lower()
        if len(needle) >= _MIN_LEAK_CHARS and needle in instruction:
            return Rejection(Check.GOLD_LEAK, f"expected value appears in instruction: {needle[:60]!r}")
    return None


def check_shape(task: TaskFiles, spec: Spec) -> Rejection | None:
    if isinstance(spec, JsonSchemaSpec):
        try:
            json.loads(task.text(f"tests/{spec.schema}"))
        except json.JSONDecodeError as error:
            return Rejection(Check.SHAPE, f"schema is not JSON: {error}")
    if isinstance(spec, IfevalSpec | JudgeSpec):
        unknown = sorted({c.name for c in spec.constraints} - set(CONSTRAINTS))
        if unknown:
            return Rejection(Check.SHAPE, f"unknown ifeval constraints {unknown}")
    if isinstance(spec, IfevalSpec) and not spec.constraints:
        return Rejection(Check.SHAPE, "ifeval task has no constraints")
    if isinstance(spec, JudgeSpec):
        if spec.rubric not in RUBRICS:
            return Rejection(Check.SHAPE, f"unknown judge rubric {spec.rubric!r}")
        if spec.rubric == RUBRIC_CHECKLIST and not any(c.strip() for c in spec.criteria):
            return Rejection(Check.SHAPE, "checklist judge has no criteria")
        if spec.rubric == RUBRIC_REFERENCE and not any(r.strip() for r in spec.references):
            return Rejection(Check.SHAPE, "reference judge has no references")
    if isinstance(spec, StdioSpec):
        inputs = [p for p in task.under(f"tests/{spec.cases}/") if p.rsplit("/", 1)[-1].startswith("input_")]
        if len(inputs) < spec.min_cases:
            return Rejection(Check.SHAPE, f"{len(inputs)} stdio cases, min_cases is {spec.min_cases}")
    return None


def _materialize(task: TaskFiles, root: Path) -> Path:
    TaskFiles(task.under("tests/")).write_to(root)
    return root / "tests"


def check_grading(task: TaskFiles, spec: Spec) -> Rejection | None:
    """Run the real grader on an empty output, the expected value, and a perturbation."""
    positive = positive_candidate(spec)
    if positive is None:
        return None
    negative = negative_candidate(spec)
    with tempfile.TemporaryDirectory(prefix="tasktrove-verify-") as tmp:
        root = Path(tmp)
        tests_dir = _materialize(task, root)
        workspace = root / "app"
        workspace.mkdir()
        answer = local_output_path(spec.output, workspace)
        answer.parent.mkdir(parents=True, exist_ok=True)

        reward = grade(spec, tests_dir, workspace)
        if reward.status != Status.SCORED or reward.reward != 0.0:
            return Rejection(Check.EMPTY, f"empty output scored {reward.reward} {reward.status}: {reward.detail}")
        answer.write_text(positive)
        reward = grade(spec, tests_dir, workspace)
        if reward.status != Status.SCORED or reward.reward != 1.0:
            return Rejection(Check.EXPECTED, f"expected value scored {reward.reward} {reward.status}: {reward.detail}")
        if negative is not None:
            answer.write_text(negative)
            reward = grade(spec, tests_dir, workspace)
            if reward.status != Status.SCORED or reward.reward != 0.0:
                return Rejection(Check.PERTURBED, f"perturbed value scored {reward.reward}: {reward.detail}")
    return None


def verify_task(blob: bytes) -> Rejection | None:
    task = read_task_binary(blob)
    spec, rejection = check_spec(task)
    if rejection is not None or spec is None:
        return rejection
    return check_dockerfile(task) or check_gold_leak(task, spec) or check_shape(task, spec) or check_grading(task, spec)


def verify_rows(rows: Iterator[dict]) -> Iterator[dict]:
    """Every row, with converted rows that fail a check re-stamped ``verified:<check>`` and their binaries dropped."""
    for row in rows:
        if row["status"] != ConvertStatus.CONVERTED:
            yield row
            continue
        rejection = verify_task(row["task_binary"])
        yield row if rejection is None else dropped(row, VERIFIED_STATUS + rejection.check.value, rejection.detail)


def filter_tasks(converted_path: str, output_path: str, max_tasks_per_source: int | None) -> None:
    """Deduplicate converted rows and remove tasks that fail a verifier check."""
    files = Dataset.from_files(str(StoragePath(converted_path) / CONVERTED_GLOB))
    ds = files.load_parquet(approx_shard_bytes=APPROX_SHARD_BYTES)
    ds = ds.group_by(
        key=dedup_key,
        reducer=keep_first,
        sort_by=lambda row: row["path"],
        num_output_shards=WORKING_SHARDS,
    )
    if max_tasks_per_source is not None:
        ds = ds.group_by(
            key=lambda row: row["source"],
            reducer=lambda _source, rows: cap_source(rows, max_tasks_per_source),
            sort_by=lambda row: cap_rank(row["path"]),
            num_output_shards=WORKING_SHARDS,
        )
    ds = ds.map_shard(lambda rows, _: verify_rows(rows))
    ds = ds.write_parquet(str(StoragePath(output_path) / "graded/part-{shard:05d}.parquet"), schema=CONVERTED_SCHEMA)
    ZephyrContext(name="tasktrove-filter", resources=WORKER_RESOURCES).execute(ds)
