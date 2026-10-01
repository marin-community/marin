# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Generate execution-verified Python tasks for the code curriculum with GLM.

Two task kinds cover the failure modes seen on the matched code benchmarks:

- ``implement`` (HumanEval+/MBPP+): GLM writes a function specification, a reference solution, and
  assert-based tests. A task is kept only when the reference passes every test in a subprocess and
  a stub that returns ``None`` fails at least one test, so the tests constrain behavior.
- ``trace`` (CRUXEval): GLM writes a short pure function and a call. The reference answer is the
  call's value obtained by running the code twice in a subprocess; GLM's claimed value is kept only
  for audit. Tasks whose value is not a Python literal or differs between runs are rejected.

The problems Parquet matches ``generation.PROBLEMS_FILENAME``, so ``self_distill`` can read it. Its
``answer`` is the value's ``repr`` for trace tasks and the JSON-encoded tests for implement tasks.
"""

from __future__ import annotations

import ast
import json
import logging
import resource
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import pyarrow as pa
from marin.execution.artifact import Artifact
from marin.execution.lazy import ArtifactStep, StepContext
from marin.experiment.namespacing import user_owned_name
from marin.inference.openai_batch import CHAT_COMPLETIONS_ENDPOINT
from marin.inference.structured_output import StructuredTool
from pydantic import Field, ValidationError
from rigging.filesystem.storage_path import StoragePath

from experiments.post_training.baby_rsi.generation import (
    MANIFEST_FILENAME,
    PROBLEMS_FILENAME,
    RAW_RESPONSES_FILENAME,
    _failure_reason,
    _glm_client,
    _responses_by_id,
    _run_batch,
    capability_packet,
    problem_targets,
    write_table,
)
from experiments.post_training.glm import DEFAULT_GLM_RELAY_JOB, GLM_MODEL
from experiments.post_training.task_curriculum.catalog_artifact import TASK_CURRICULUM, TaskCurriculumCatalogArtifact
from experiments.post_training.task_curriculum.models import StrictModel

logger = logging.getLogger(__name__)

IMPLEMENTATION_CAPABILITY = "d02.swe.implementation"
DYNAMIC_SEMANTICS_CAPABILITY = "d02.pl.dynamic_semantics"

EXECUTION_TIMEOUT = 5.0
MEMORY_LIMIT_BYTES = 1 << 30
OUTPUT_LIMIT_BYTES = 1 << 20
TRACE_MIN_LINES = 5
TRACE_MAX_LINES = 15
# Printed before the harness's JSON result so output written by the executed code cannot be mistaken for it.
RESULT_SENTINEL = "__CURRICULUM_RESULT__"

# Probe failures on HumanEval+/MBPP+ are dominated by string manipulation, then number theory and list processing.
TOPICS = (
    "string manipulation: slicing, splitting, case, character classes, and building strings",
    "number theory: divisibility, primes, digits, gcd and lcm, and modular arithmetic",
    "list and dict processing: filtering, grouping, counting, sorting with keys, and windows",
    "parsing structured text: tokens, delimiters, brackets, simple formats, and validation",
)


class TaskKind(StrEnum):
    IMPLEMENT = "implement"
    """Implement a specified function; graded by hidden tests."""
    TRACE = "trace"
    """Predict the value of a call to a short function; graded by literal equality."""


class ExecutionStatus(StrEnum):
    OK = "ok"
    ERROR = "error"
    TIMEOUT = "timeout"


@dataclass(frozen=True)
class ExecutionResult:
    status: ExecutionStatus
    stdout: str


class ImplementTask(StrictModel):
    function_name: str = Field(min_length=1)
    specification: str = Field(min_length=1)
    solution: str = Field(min_length=1)
    tests: list[str] = Field(min_length=6, max_length=10)


class TraceTask(StrictModel):
    code: str = Field(min_length=1)
    call: str = Field(min_length=1)
    output: str = Field(min_length=1)


IMPLEMENT_TOOL = StructuredTool(
    name="submit_task",
    description="Submit one Python function specification, its reference solution, and assert-based tests.",
    output_type=ImplementTask,
)

TRACE_TOOL = StructuredTool(
    name="submit_task",
    description="Submit one short pure Python function, a call to it, and the call's value as a Python literal.",
    output_type=TraceTask,
)

CODE_TASK_SCHEMA = pa.schema(
    [
        pa.field("request_id", pa.string(), nullable=False),
        pa.field("capability_id", pa.string(), nullable=False),
        pa.field("kind", pa.string(), nullable=False),
        pa.field("facet_id", pa.string(), nullable=False),
        pa.field("topic", pa.string(), nullable=False),
        pa.field("problem", pa.string()),
        pa.field("answer", pa.string()),
        pa.field("reference_code", pa.string()),
        pa.field("claimed_output", pa.string()),
        pa.field("accepted", pa.bool_(), nullable=False),
        pa.field("rejection_reason", pa.string()),
    ]
)

TESTS_HARNESS = f"""
import io, json, sys
payload = json.loads(sys.stdin.read())
sys.stdout = io.StringIO()
namespace = {{"__name__": "__candidate__"}}
exec(payload["code"], namespace)
passed = []
for test in payload["tests"]:
    try:
        exec(test, namespace)
        passed.append(True)
    except BaseException:
        passed.append(False)
sys.__stdout__.write("{RESULT_SENTINEL}" + json.dumps(passed))
"""

TRACE_HARNESS = f"""
import ast, io, json, sys
payload = json.loads(sys.stdin.read())
sys.stdout = io.StringIO()
namespace = {{"__name__": "__candidate__"}}
exec(payload["code"], namespace)
value = eval(payload["call"], namespace)
literal = repr(value)
try:
    round_trips = ast.literal_eval(literal) == value
except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
    round_trips = False
sys.__stdout__.write("{RESULT_SENTINEL}" + json.dumps({{"literal": literal, "round_trips": round_trips}}))
"""


@dataclass(frozen=True)
class CodeTaskAssignment:
    facet_id: str
    facet_description: str
    topic: str


@dataclass(frozen=True)
class GenerateCodeTasksConfig:
    catalog_path: str
    output_path: str
    capability_id: str
    kind: TaskKind
    requested: int
    seed: int
    max_completion_tokens: int
    relay_job: str


def _limit_resources(timeout: float) -> None:
    resource.setrlimit(resource.RLIMIT_CPU, (int(timeout) + 1, int(timeout) + 1))
    resource.setrlimit(resource.RLIMIT_FSIZE, (OUTPUT_LIMIT_BYTES, OUTPUT_LIMIT_BYTES))
    # macOS does not enforce address-space limits.
    if sys.platform == "linux":
        resource.setrlimit(resource.RLIMIT_AS, (MEMORY_LIMIT_BYTES, MEMORY_LIMIT_BYTES))


def execute_python(source: str, stdin: str, timeout: float) -> ExecutionResult:
    """Run ``source`` in an isolated interpreter in an empty temporary directory.

    The child runs with ``python -I`` and an empty environment, and CPU time, file size, and (on
    Linux) address space are capped. This guards against accidents in generated code, not against
    adversarial code: the child can still reach the network and the filesystem.
    """
    with tempfile.TemporaryDirectory() as workdir:
        try:
            completed = subprocess.run(
                [sys.executable, "-I", "-c", source],
                input=stdin,
                capture_output=True,
                text=True,
                cwd=workdir,
                env={},
                timeout=timeout,
                preexec_fn=lambda: _limit_resources(timeout),
            )
        except subprocess.TimeoutExpired:
            return ExecutionResult(ExecutionStatus.TIMEOUT, "")
    status = ExecutionStatus.OK if completed.returncode == 0 else ExecutionStatus.ERROR
    return ExecutionResult(status, completed.stdout)


def _harness_result(harness: str, payload: dict[str, Any], timeout: float) -> Any:
    """The harness's JSON result, or None when the code crashed, timed out, or printed no result."""
    result = execute_python(harness, json.dumps(payload), timeout)
    if result.status is not ExecutionStatus.OK or RESULT_SENTINEL not in result.stdout:
        return None
    return json.loads(result.stdout.rpartition(RESULT_SENTINEL)[2])


def run_tests(solution_code: str, tests: list[str], timeout: float) -> list[bool]:
    """Whether each assert-based test passes against ``solution_code``; all fail if the code crashes or times out."""
    passed = _harness_result(TESTS_HARNESS, {"code": solution_code, "tests": tests}, timeout)
    return passed if passed is not None else [False] * len(tests)


def trace_value(code: str, call: str, timeout: float) -> tuple[str | None, str | None]:
    """Execute ``call`` after ``code`` twice and return ``(repr of the value, rejection reason)``."""
    runs = [_harness_result(TRACE_HARNESS, {"code": code, "call": call}, timeout) for _ in range(2)]
    if any(run is None for run in runs):
        return None, "trace_failed"
    if not all(run["round_trips"] for run in runs):
        return None, "non_literal_output"
    # Each run gets a fresh hash seed, so a value that depends on set or string-hash iteration order differs.
    if not literals_equal(runs[0]["literal"], runs[1]["literal"]):
        return None, "nondeterministic"
    return runs[0]["literal"], None


def _same_value(expected: Any, actual: Any) -> bool:
    # Tracing distinguishes True from 1 and 1.0 from 1, which == does not.
    if type(expected) is not type(actual):
        return False
    if isinstance(expected, list | tuple):
        return len(expected) == len(actual) and all(map(_same_value, expected, actual))
    if isinstance(expected, dict):
        return expected.keys() == actual.keys() and all(_same_value(expected[key], actual[key]) for key in expected)
    return expected == actual


def literals_equal(reference: str, candidate: str) -> bool:
    """Whether two Python literals denote the same value of the same types, ignoring formatting."""
    try:
        actual = ast.literal_eval(candidate.strip())
    except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
        return False
    return _same_value(ast.literal_eval(reference), actual)


def code_task_assignment(packet: dict[str, Any], index: int) -> CodeTaskAssignment:
    """Cycle requests through every target, then every topic."""
    targets = problem_targets(packet)
    target = targets[index % len(targets)]
    return CodeTaskAssignment(
        facet_id=target["id"],
        facet_description=target["description"],
        topic=TOPICS[(index // len(targets)) % len(TOPICS)],
    )


def _capability_json(packet: dict[str, Any]) -> str:
    capability = {key: packet[key] for key in ("subject", "capability_id", "name", "outcome", "includes", "excludes")}
    return json.dumps(capability, ensure_ascii=False, sort_keys=True)


def implement_prompt(packet: dict[str, Any], assignment: CodeTaskAssignment) -> str:
    return (
        "Write one original, self-contained Python function-implementation task at the level of HumanEval "
        "and MBPP, with a reference solution and tests.\n"
        "- `specification`: the function signature with type hints and a docstring that states the full "
        "behavior, including edge cases and one or two examples. No function body.\n"
        "- `solution`: a complete, correct implementation of that function using only the standard library.\n"
        "- `tests`: 6 to 10 independent one-line `assert` statements that call the function. Cover typical "
        "inputs and edge cases such as empty input, single elements, ties, zero, and negative numbers. "
        "Each test must pass for any correct implementation and fail for a plausible wrong one.\n"
        "- Do not copy published benchmark problems; no I/O, randomness, time, or network access.\n"
        f"Capability: {_capability_json(packet)}\n"
        f"Focus: {assignment.facet_description}\n"
        f"Topic: {assignment.topic}"
    )


def trace_prompt(packet: dict[str, Any], assignment: CodeTaskAssignment) -> str:
    return (
        "Write one original program-tracing task at the level of CRUXEval.\n"
        f"- `code`: one pure, deterministic Python function of {TRACE_MIN_LINES} to {TRACE_MAX_LINES} lines "
        "using only built-ins, with no imports, I/O, or randomness. Use loops, branches, slicing, string or "
        "container methods, or arithmetic whose result takes careful step-by-step tracing.\n"
        "- `call`: one call to the function whose arguments are Python literals.\n"
        "- `output`: the exact value of the call as a Python literal, as `repr` prints it.\n"
        f"Capability: {_capability_json(packet)}\n"
        f"Focus: {assignment.facet_description}\n"
        f"Topic: {assignment.topic}"
    )


def implement_problem(specification: str) -> str:
    return (
        "Implement the following Python function so that it satisfies the docstring.\n\n"
        f"```python\n{specification.strip()}\n```\n\n"
        "Give the complete function in one ```python code block."
    )


def trace_problem(code: str, call: str) -> str:
    return (
        "Consider the following Python code.\n\n"
        f"```python\n{code.strip()}\n```\n\n"
        f"What is the value of `{call.strip()}`? Give the exact value as a Python literal, as `repr` would "
        "print it, in \\boxed{}."
    )


def code_task_request(config: GenerateCodeTasksConfig, packet: dict[str, Any], index: int) -> dict[str, Any]:
    assignment = code_task_assignment(packet, index)
    if config.kind is TaskKind.IMPLEMENT:
        prompt, tool = implement_prompt(packet, assignment), IMPLEMENT_TOOL
    else:
        prompt, tool = trace_prompt(packet, assignment), TRACE_TOOL
    body = {
        "model": GLM_MODEL,
        "messages": [
            {"role": "system", "content": "Write one original Python task and check it carefully before submitting."},
            {"role": "user", "content": prompt},
        ],
        "chat_template_kwargs": {"reasoning_effort": "high"},
        "temperature": 1.0,
        "seed": config.seed + index,
        "max_tokens": config.max_completion_tokens,
    }
    body.update(tool.request_fields())
    return {"custom_id": f"task-{index:05d}", "method": "POST", "url": CHAT_COMPLETIONS_ENDPOINT, "body": body}


def _stub(function_name: str) -> str:
    return f"def {function_name}(*args, **kwargs):\n    return None\n"


def verify_implement_task(task: ImplementTask) -> str | None:
    """Rejection reason for an implement task, or None when its reference passes and its tests are not vacuous."""
    if not all(run_tests(task.solution, task.tests, EXECUTION_TIMEOUT)):
        return "reference_failed"
    if all(run_tests(_stub(task.function_name), task.tests, EXECUTION_TIMEOUT)):
        return "vacuous_tests"
    return None


def _parse_implement(response: dict[str, Any], seen: set[str]) -> tuple[dict[str, str | None], str | None]:
    try:
        task = IMPLEMENT_TOOL.parse(response["response"]["body"])
    except (UnicodeError, ValidationError, ValueError):
        return {}, "invalid_task"
    fields: dict[str, str | None] = {
        "problem": implement_problem(task.specification),
        "answer": json.dumps(task.tests),
        "reference_code": task.solution,
        "claimed_output": None,
    }
    normalized = " ".join(task.specification.split())
    if normalized in seen:
        return fields, "duplicate_task"
    seen.add(normalized)
    return fields, verify_implement_task(task)


def _parse_trace(response: dict[str, Any], seen: set[str]) -> tuple[dict[str, str | None], str | None]:
    try:
        task = TRACE_TOOL.parse(response["response"]["body"])
    except (UnicodeError, ValidationError, ValueError):
        return {}, "invalid_task"
    fields: dict[str, str | None] = {
        "problem": trace_problem(task.code, task.call),
        "answer": None,
        "reference_code": task.code,
        "claimed_output": task.output,
    }
    lines = [line for line in task.code.strip().splitlines() if line.strip()]
    if not TRACE_MIN_LINES <= len(lines) <= TRACE_MAX_LINES:
        return fields, "code_length"
    normalized = " ".join(f"{task.code} {task.call}".split())
    if normalized in seen:
        return fields, "duplicate_task"
    seen.add(normalized)
    fields["answer"], reason = trace_value(task.code, task.call, EXECUTION_TIMEOUT)
    return fields, reason


def parse_code_task_batch(
    raw_output: str, config: GenerateCodeTasksConfig, packet: dict[str, Any]
) -> list[dict[str, Any]]:
    """Execute and verify every generated task; keep rejection accounting for the rest."""
    responses = _responses_by_id(raw_output, {f"task-{index:05d}" for index in range(config.requested)})
    parse = _parse_implement if config.kind is TaskKind.IMPLEMENT else _parse_trace
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index in range(config.requested):
        request_id = f"task-{index:05d}"
        assignment = code_task_assignment(packet, index)
        fields: dict[str, str | None] = {}
        reason = _failure_reason(responses[request_id])
        if reason is None:
            fields, reason = parse(responses[request_id], seen)
        records.append(
            {
                "request_id": request_id,
                "capability_id": config.capability_id,
                "kind": str(config.kind),
                "facet_id": assignment.facet_id,
                "topic": assignment.topic,
                "problem": fields.get("problem"),
                "answer": fields.get("answer"),
                "reference_code": fields.get("reference_code"),
                "claimed_output": fields.get("claimed_output"),
                "accepted": reason is None,
                "rejection_reason": reason,
            }
        )
    return records


def generate_tasks(config: GenerateCodeTasksConfig) -> Artifact:
    """Write verified task Parquet, exact GLM responses, and a manifest."""
    catalog = TaskCurriculumCatalogArtifact(path=config.catalog_path).read_catalog()
    packet = capability_packet(catalog, config.capability_id)
    requests = [code_task_request(config, packet, index) for index in range(config.requested)]
    raw_output = _run_batch(
        _glm_client(config.relay_job), requests, f"curriculum-code-{config.kind}-{config.capability_id}.jsonl"
    )
    records = parse_code_task_batch(raw_output, config, packet)

    output = StoragePath(config.output_path)
    output.mkdirs()
    write_table(output, PROBLEMS_FILENAME, records, CODE_TASK_SCHEMA)
    (output / RAW_RESPONSES_FILENAME).write_text(raw_output)
    reasons: dict[str, int] = {}
    for record in records:
        key = record["rejection_reason"] or "accepted"
        reasons[key] = reasons.get(key, 0) + 1
    manifest = {
        "catalog_version": catalog.catalog_version,
        "capability_id": config.capability_id,
        "kind": str(config.kind),
        "generator": GLM_MODEL,
        "requested": len(records),
        "outcomes": reasons,
        "verified": (
            "reference passes all tests and a None stub fails one"
            if config.kind is TaskKind.IMPLEMENT
            else "answer is the executed value, identical over two runs"
        ),
        "problems": PROBLEMS_FILENAME,
        "raw_responses": RAW_RESPONSES_FILENAME,
    }
    (output / MANIFEST_FILENAME).write_text(json.dumps(manifest, indent=2) + "\n")
    logger.info("code tasks for %s (%s): %s", config.capability_id, config.kind, reasons)
    return Artifact(path=config.output_path)


def generate_code_tasks(
    capability_id: str,
    *,
    kind: TaskKind,
    version: str,
    requested: int,
    seed: int,
    max_completion_tokens: int,
    catalog: ArtifactStep[TaskCurriculumCatalogArtifact] = TASK_CURRICULUM,
) -> ArtifactStep[Artifact]:
    """Build one GLM code-task generation step; run it on `cw-us-east-08a`, where the GLM relay is reachable."""

    def build_config(ctx: StepContext) -> GenerateCodeTasksConfig:
        return GenerateCodeTasksConfig(
            catalog_path=ctx.artifact_path(catalog),
            output_path=ctx.output_path,
            capability_id=capability_id,
            kind=kind,
            requested=requested,
            seed=seed,
            max_completion_tokens=max_completion_tokens,
            relay_job=DEFAULT_GLM_RELAY_JOB,
        )

    return ArtifactStep(
        name=user_owned_name(f"documents/curriculum-sft/{capability_id}/{kind}-tasks"),
        version=version,
        artifact_type=Artifact,
        run=generate_tasks,
        build_config=build_config,
        deps=(catalog,),
    )
