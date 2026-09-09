# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The Marin verifier contract for converted tasks.

A converted task's ``task.toml`` carries a ``[verifier]`` table with a ``kind`` discriminator and
one kind-specific subtable. Every verifier implementation lives in the task image as the
``tasktrove-verify`` tool; a task ships only data. ``tests/test.sh`` is the same three lines for
every task and hands off to that tool. The tool writes ``/logs/verifier/reward.json`` as
``{"reward": float, "status": "scored" | "infra_error", "detail": {...}}``.
"""

import tomllib
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any

import tomlkit

VERIFY_TEST_SH = """#!/bin/bash
set -euo pipefail
exec tasktrove-verify /task/task.toml
"""

VERIFY_TOOL_REF = "git+https://github.com/marin-community/marin@main#subdirectory=lib/tasktrove-verify"
UV_IMAGE = "ghcr.io/astral-sh/uv:0.8"


class VerifierKind(StrEnum):
    ANSWER = "answer"
    JUDGE = "judge"
    SCHEMA = "schema"
    CHECKER = "checker"
    STDIO = "stdio"
    TESTS = "tests"
    SCRIPT = "script"


class AnswerType(StrEnum):
    MCQ = "mcq"
    MATH = "math"
    NUMERIC = "numeric"
    EXACT = "exact"
    LIST = "list"


class MathType(StrEnum):
    SCALAR = "scalar"
    INTERVAL = "interval"
    SET = "set"
    TUPLE = "tuple"
    EQUATION = "equation"


class TestReport(StrEnum):
    PYTEST_JSON = "pytest-json"
    JUNIT_XML = "junit-xml"
    GO_JSON = "go-json"
    EXIT_CODE = "exit-code"


class ImageTier(StrEnum):
    """Shared base images. Tasks in the same tier get byte-identical Dockerfiles."""

    ANSWER = "answer"
    PYTEST = "pytest"
    POLYGLOT = "polyglot"
    SWE = "swe"


@dataclass(frozen=True)
class AnswerSpec:
    type: AnswerType
    expected: str
    math_type: MathType | None = None
    options: int | None = None
    tolerance_abs: float | None = None
    tolerance_rel: float | None = None


@dataclass(frozen=True)
class JudgeSpec:
    references: tuple[str, ...]
    rubric: str
    model: str
    exact_gate: bool = True


@dataclass(frozen=True)
class SchemaSpec:
    format: str
    schema: dict


@dataclass(frozen=True)
class CheckerSpec:
    name: str
    params: dict = field(default_factory=dict)


@dataclass(frozen=True)
class StdioSpec:
    cases: str = "tests/cases"
    compare: str = "exact"
    special_judge: str | None = None
    per_case_timeout: float = 10.0
    min_cases: int = 5


@dataclass(frozen=True)
class TestsSpec:
    command: str
    report: TestReport
    must_pass: tuple[str, ...] = ()
    must_not_break: tuple[str, ...] = ()
    restore: tuple[str, ...] = ()
    setup: str | None = None
    workdir: str = "/app"


@dataclass(frozen=True)
class ScriptSpec:
    path: str


@dataclass(frozen=True)
class VerifierSpec:
    kind: VerifierKind
    output: str = "/app/answer.txt"
    network: bool = False
    answer: AnswerSpec | None = None
    judge: JudgeSpec | None = None
    schema: SchemaSpec | None = None
    checker: CheckerSpec | None = None
    stdio: StdioSpec | None = None
    tests: TestsSpec | None = None
    script: ScriptSpec | None = None

    def __post_init__(self) -> None:
        body = getattr(self, self.kind.value)
        if body is None:
            raise ValueError(f"verifier kind {self.kind} requires a [verifier.{self.kind}] table")
        others = [k for k in VerifierKind if k != self.kind and getattr(self, k.value) is not None]
        if others:
            raise ValueError(f"verifier kind {self.kind} cannot also carry {others}")
        if self.network and self.kind != VerifierKind.JUDGE:
            raise ValueError("only judge verifiers may enable network access")

    def to_table(self) -> dict[str, Any]:
        table: dict[str, Any] = {"kind": self.kind.value, "output": self.output, "network": self.network}
        body = {k: v for k, v in asdict(getattr(self, self.kind.value)).items() if v is not None}
        table[self.kind.value] = body
        return table


def render_task_toml(agent_timeout: float, verifier_timeout: float, verifier: VerifierSpec, metadata: dict) -> str:
    """Render the converted task.toml: Harbor's top-level tables plus our ``[verifier]`` table."""
    doc = {
        "version": "1.0",
        "agent": {"timeout_sec": agent_timeout},
        "verifier": {"timeout_sec": verifier_timeout, "restart_environment": False, **verifier.to_table()},
        "metadata": metadata,
    }
    return tomlkit.dumps(doc)


_BODY_TYPES = {
    VerifierKind.ANSWER: AnswerSpec,
    VerifierKind.JUDGE: JudgeSpec,
    VerifierKind.SCHEMA: SchemaSpec,
    VerifierKind.CHECKER: CheckerSpec,
    VerifierKind.STDIO: StdioSpec,
    VerifierKind.TESTS: TestsSpec,
    VerifierKind.SCRIPT: ScriptSpec,
}


def parse_verifier_spec(task_toml: str) -> VerifierSpec:
    """Reconstruct the typed spec from a converted task.toml; raises on a malformed table."""
    table = tomllib.loads(task_toml)["verifier"]
    kind = VerifierKind(table["kind"])
    if kind.value not in table:
        raise ValueError(f"[verifier] kind={kind} but no [verifier.{kind}] table")
    body = {k: tuple(v) if isinstance(v, list) else v for k, v in table[kind.value].items()}
    return VerifierSpec(
        kind=kind,
        output=table["output"],
        network=table["network"],
        **{kind.value: _BODY_TYPES[kind](**body)},
    )


_TIER_DOCKERFILES = {
    ImageTier.ANSWER: (
        f"""FROM python:3.11-slim-bookworm
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 UV_OFFLINE=1
RUN mkdir -p /app /tests /logs/verifier && chmod 755 /app /tests
COPY --from={UV_IMAGE} /uv /usr/local/bin/uv
RUN UV_OFFLINE=0 uv tool install "tasktrove-verify[answer] @ {VERIFY_TOOL_REF}"
WORKDIR /app
"""
    ),
    ImageTier.PYTEST: (
        f"""FROM python:3.11-slim-bookworm
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 UV_OFFLINE=1
RUN apt-get update && apt-get install -y --no-install-recommends build-essential git && rm -rf /var/lib/apt/lists/*
RUN mkdir -p /app /tests /logs/verifier && chmod 755 /app /tests
RUN pip install --no-cache-dir pytest pytest-timeout pytest-json-report numpy scipy sympy networkx pandas requests
COPY --from={UV_IMAGE} /uv /usr/local/bin/uv
RUN UV_OFFLINE=0 uv tool install "tasktrove-verify[tests] @ {VERIFY_TOOL_REF}"
WORKDIR /app
"""
    ),
    ImageTier.POLYGLOT: (
        f"""FROM ubuntu:24.04
ENV DEBIAN_FRONTEND=noninteractive UV_OFFLINE=1
RUN apt-get update && apt-get install -y --no-install-recommends \\
    bash git curl ca-certificates build-essential cmake libgtest-dev \\
    python3 python3-pip python3-pytest golang-go openjdk-17-jdk maven gradle \\
    nodejs npm ruby-full php-cli composer rustc cargo \\
    && rm -rf /var/lib/apt/lists/*
RUN mkdir -p /app /tests /logs/verifier && chmod 755 /app /tests
COPY --from={UV_IMAGE} /uv /usr/local/bin/uv
RUN UV_OFFLINE=0 uv tool install "tasktrove-verify[tests] @ {VERIFY_TOOL_REF}"
WORKDIR /app
"""
    ),
    ImageTier.SWE: (
        f"""FROM python:3.10-bookworm
ENV DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC UV_OFFLINE=1
RUN apt-get update && apt-get install -y --no-install-recommends \\
    git curl wget jq build-essential libffi-dev libssl-dev libtiff-dev locales locales-all tzdata pkg-config \\
    && rm -rf /var/lib/apt/lists/*
RUN mkdir -p /logs/verifier /testbed /output && chmod 777 /output
COPY --from={UV_IMAGE} /uv /usr/local/bin/uv
RUN UV_OFFLINE=0 uv tool install "tasktrove-verify[tests] @ {VERIFY_TOOL_REF}"
WORKDIR /testbed
"""
    ),
}


def tier_dockerfile(tier: ImageTier, setup: str | None = None) -> str:
    """The canonical Dockerfile for a tier, plus an optional per-repository setup block.

    Every task in a tier without ``setup`` gets a byte-identical Dockerfile so image builds cache
    across tasks. SWE tasks append their repository setup as a trailing ``RUN`` block, so tasks
    from the same repository still share one image.
    """
    text = _TIER_DOCKERFILES[tier]
    if setup:
        text += "\n# --- repository setup ---\n" + setup.rstrip() + "\n"
    return text
