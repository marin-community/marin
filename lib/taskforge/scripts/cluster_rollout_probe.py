# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run TaskSpecs through RolloutEngine inside an Iris task, with GLM-5.3 resolved in-task.

Submit from the worktree root as one CPU job federated to a cluster that hosts a GLM relay and
starts gVisor pods (``cw-rno2a``: relay ``/muchanem/glm53-relay-rno2a``)::

    lib/taskforge/.venv/bin/iris --cluster=marin job run --no-wait --no-sync \
      --job-name taskforge-cluster-rollout-probe-NN --target-cluster cw-rno2a --priority interactive \
      --cpu 2 --memory 3GB --timeout 5400 -e GLM_API_TOKEN "$GLM_API_TOKEN" -- \
      bash -c 'cd lib/taskforge && uv run --frozen python scripts/cluster_rollout_probe.py \
        --relay-job /muchanem/glm53-relay-rno2a'

``GLM_API_TOKEN`` travels in the job's environment (``-e``), so the controller stores it with the
job and the task's ``IRIS_JOB_ENV`` carries it. Iris copies ``IRIS_JOB_ENV`` into every child job,
and each docker machine is a child job the model controls, so ``main`` removes the token (and the
submitter keys Iris forwards) from this process's environment before anything submits a child.

Phases, each ``k`` trials through ``taskforge.validate.trials.run_trials``:

- ``math``: a null-environment numeric task (no machine).
- ``docker_shipped``: a docker task from a digest-pinned public image on
  ``machine_factories(MachineHost.IRIS, controller_url)`` exactly as shipped, with the task's
  ``IRIS_CONTROLLER_URL``.
- ``docker_readiness_fix``: the same task and factory with a readiness poll that
  compares against ``iris`` ``TaskState``, patched into this process (``apply_readiness_fix``),
  then one machine that lists the environment variable names a sandbox receives. The run exits
  non-zero after writing its results when any of those names looks like a credential
  (``secret_names``): on CoreWeave, cluster ``task_env`` object-store keys reach every
  sandbox, which a model-controlled machine with network must not see.

The docker task asks for ``network: true``: the shipped Iris factory refuses ``NetworkPolicy.DENY``.

Everything goes to ``--results-dir`` (default ``$IRIS_OUTPUT_DIR/cluster_rollout_probe``, which
Iris archives with the attempt): ``summary.json``, the trial evidence ``run_trials`` writes, and
``ledger/<item>.jsonl``. Each file is printed as ``RESULT_FILE <path> <index>/<count> <chunk>``
lines, where ``<chunk>`` is a JSON string of at most ``PRINT_CHUNK`` characters, and the summary as
``RESULT_SUMMARY``. ``--reassemble <logs> --results-dir <dir>`` rebuilds the directory from saved
``iris job logs`` output: chunks are grouped by path, ordered by index, ``json.loads``-ed and
concatenated. The directory is also copied to ``--upload-prefix`` (default
``$MARIN_PREFIX/taskforge/cluster_rollout_probe/<run>``) only when the task already holds
credentials for that store; this script adds none.
"""

import argparse
import asyncio
import json
import os
import time
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from iris.client.workload import TaskState
from iris.rpc import job_pb2
from rigging.filesystem.storage_path import StoragePath, prefix_join
from rigging.timing import ExponentialBackoff
from shellbox.backends.iris import machine as iris_backend
from shellbox.image import RegistryImage as ShellboxRegistryImage
from shellbox.machine import Command, Machine, MachineFactory, MachineSpec, NetworkPolicy
from taskcompendium.environment import EnvironmentKind, RegistryImage, StdoutReward
from taskcompendium.execution import TaskExecution
from taskcompendium.grading import numeric_answer
from taskcompendium.models import AnswerType, Source, TaskSpec
from taskcompendium.submission import PlainText

from taskforge.ledger.jsonl import JsonlLedger, ledger_files, read_entries
from taskforge.ledger.records import entry_to_json
from taskforge.llm.client import GlmClient, Pool, endpoint_in_task
from taskforge.llm.policy import LLMPolicy
from taskforge.llm.rollout_model import GlmRolloutModel
from taskforge.sandbox.factories import IRIS_DOCKER, SHELLSIM, MachineHost, machine_factories
from taskforge.spec.draft import assemble, environment, file, shell_verifier
from taskforge.validate.evidence import Complete, Evidence
from taskforge.validate.outcome import Graded, Outcome, TrialKind
from taskforge.validate.trials import EngineSettings, TrialPlan, run_trials

GLM_TOKEN_ENV = "GLM_API_TOKEN"
IRIS_CONTROLLER_URL_ENV = "IRIS_CONTROLLER_URL"
# Iris copies these from the submitting process into every child job (iris.cluster.types.EnvironmentSpec).
SUBMITTER_KEYS = ("HF_TOKEN", "WANDB_API_KEY")
IRIS_JOB_ENV = "IRIS_JOB_ENV"
IMAGE = "docker.io/library/python@sha256:02108f5d322dd89f1c9e552442c25acb0543dfdbc455693a5599624f20d9155d"
IMAGE_TAG = "python:3.12-slim (OCI index digest, resolved 2026-10-05)"
MATH_ANSWER = "395"
NUMBERS = "12\n7\n30\n11\n"
NUMBERS_SUM = 60
CHECK_SCRIPT = f'v=$(tr -d " \\n" < /workspace/sum.txt)\nif [ "$v" = {NUMBERS_SUM} ]; then echo 1; else echo 0; fi\n'
# Prints names only; `env | cut` would leak fragments of multi-line values into evidence.
ENV_NAMES_SCRIPT = (
    f"awk 'BEGIN{{for(k in ENVIRON) print k}}' | sort | tr '\\n' ' '; echo; printenv {GLM_TOKEN_ENV} | wc -c"
)
# A sandbox environment name is treated as a credential when one of its ``_``-separated words is in
# SECRET_WORDS or it ends with one of SECRET_SUFFIXES (so GPG_KEY and TOKENIZERS_* are not).
SECRET_WORDS = frozenset({"SECRET", "TOKEN", "PASSWORD"})
SECRET_SUFFIXES = ("KEY_ID", "API_KEY", "ACCESS_KEY")
POLICY = LLMPolicy(max_continuations=0)
MAX_TURNS = 12
COMMAND_TIMEOUT = 120
CLEANUP_TIMEOUT = 120
EXECUTION = TaskExecution()
# The probe measures the shipped Iris backend, so it does not refuse docker tasks up front: the
# DOCKER row is the backend's own create-time checks (registry images, network ALLOW only).
PROBE_CAPABILITIES = {
    EnvironmentKind.SHELLSIM: SHELLSIM,
    EnvironmentKind.DOCKER: replace(IRIS_DOCKER, network=frozenset({NetworkPolicy.ALLOW}), unavailable=None),
}
# Characters of a result file per log line, so log storage keeps each line whole.
PRINT_CHUNK = 4000


class Phase(StrEnum):
    MATH = "math"
    DOCKER_SHIPPED = "docker_shipped"
    DOCKER_READINESS_FIX = "docker_readiness_fix"


@dataclass
class TimedFactory:
    """Delegates to ``factory`` and records how long each ``create`` took and how it ended."""

    factory: MachineFactory
    creates: list[dict[str, Any]] = field(default_factory=list)

    async def create(self, spec: MachineSpec) -> Machine:
        started = time.monotonic()
        try:
            machine = await self.factory.create(spec)
        except Exception as error:
            self.creates.append({"ok": False, "seconds": time.monotonic() - started, "error": repr(error)[:1000]})
            raise
        self.creates.append({"ok": True, "seconds": time.monotonic() - started})
        return machine


def scrub_child_environment() -> str:
    """Return the GLM token and remove it, and the forwarded submitter keys, from what children inherit."""
    token = os.environ.pop(GLM_TOKEN_ENV)
    job_env = json.loads(os.environ.get(IRIS_JOB_ENV) or "{}")
    for name in (GLM_TOKEN_ENV, *SUBMITTER_KEYS):
        job_env.pop(name, None)
        os.environ.pop(name, None)
    os.environ[IRIS_JOB_ENV] = json.dumps(job_env)
    return token


def apply_readiness_fix() -> None:
    """Make the shipped ``IrisMachineFactory._create_sync`` see ``RUNNING``, in this process only.

    The shipped poll compares ``Task.status().state`` (``iris.client.workload.TaskState``) with
    ``job_pb2.TASK_STATE_*`` ints, so it never matches. Rebinding those names in the backend module
    to ``TaskState`` members is the patch's readiness change; nothing else in the backend changes.
    It covers only the RUNNING and pending comparisons: the backend's failure branch still reads the
    nonexistent ``status.error``, and any other ``job_pb2`` name it touches would raise
    ``AttributeError``, so failure reporting still needs the upstream patch. Probe only.
    """
    states = {
        "TASK_STATE_RUNNING": TaskState.RUNNING,
        "TASK_STATE_PENDING": TaskState.PENDING,
        "TASK_STATE_BUILDING": TaskState.BUILDING,
        "TASK_STATE_ASSIGNED": TaskState.ASSIGNED,
    }
    patched = SimpleNamespace(CONTAINER_PROFILE_GVISOR=job_pb2.CONTAINER_PROFILE_GVISOR, **states)
    setattr(iris_backend, "job_pb2", patched)  # noqa: B010 - module rebinding for this probe only


def source(row: str) -> Source:
    return Source(dataset="taskforge-cluster-rollout-probe", revision="1", row=row, importer_revision="1")


def math_task() -> TaskSpec:
    return assemble(
        "cluster-math",
        "What is 17 * 23 + 4? Reply with only the number, nothing else.",
        AnswerType.NUMBER,
        environment(EnvironmentKind.NULL),
        numeric_answer(MATH_ANSWER, tolerance_abs=0, tolerance_rel=0),
        source("math"),
        execution=EXECUTION,
    )


def docker_task() -> TaskSpec:
    return assemble(
        "cluster-docker-file",
        "The file /workspace/numbers.txt holds one integer per line. Use the shell tool to write their sum, "
        "as a single integer, to /workspace/sum.txt. Say when you are done.",
        AnswerType.FILE,
        environment(
            EnvironmentKind.DOCKER,
            image=RegistryImage(reference=IMAGE),
            files=(file("/workspace/numbers.txt", NUMBERS),),
            network=True,
        ),
        shell_verifier(
            ("sh", "/grader/check.sh"), StdoutReward(), timeout=60, files=(file("/grader/check.sh", CHECK_SCRIPT),)
        ),
        source("docker-file"),
        execution=EXECUTION,
    )


def shell_calls(outcome: Outcome) -> list[str]:
    rollout = outcome.rollout
    if rollout is None:
        return []
    return [
        call["function"]["arguments"]
        for message in rollout.messages
        if message["role"] == "assistant"
        for call in message.get("tool_calls") or []
    ]


def final_message(outcome: Outcome) -> str | None:
    rollout = outcome.rollout
    if rollout is None:
        return None
    replies = [message.get("content") for message in rollout.messages if message["role"] == "assistant"]
    return replies[-1] if replies else None


def outcome_summary(outcome: Outcome) -> dict[str, Any]:
    rollout = outcome.rollout
    common: dict[str, Any] = {
        "turns": 0 if rollout is None else len(rollout.steps),
        "stop_reason": None if rollout is None else rollout.stop_reason,
        "response_tokens": 0 if rollout is None else rollout.loss_mask.count(1),
        "shell_calls": shell_calls(outcome),
        "final_message": final_message(outcome),
    }
    if isinstance(outcome, Graded):
        grade = outcome.grade
        return {
            "outcome": "graded",
            "reward": outcome.reward,
            "grade": {
                "status": str(grade.status),
                "reward": grade.reward,
                "passed": grade.passed,
                "error": grade.error,
                "failure": None if grade.failure is None else str(grade.failure),
                "diagnostics": grade.diagnostics,
            },
            **common,
        }
    return {
        "outcome": "ungraded",
        "cause": str(outcome.cause),
        "retryable": outcome.retryable,
        "detail": outcome.detail[-3000:],
        **common,
    }


def evidence_summary(outcomes: list[Outcome]) -> dict[str, Any]:
    evidence = Evidence({TrialKind.SOLVER: tuple(outcomes)})
    status = evidence.status
    stats = evidence.reward_stats(TrialKind.SOLVER)
    return {
        "status": "complete" if isinstance(status, Complete) else {"incomplete": dict(status.causes)},
        "graded": stats.graded,
        "mean_reward": stats.mean_reward,
        "solved": stats.solved,
    }


async def run_phase(
    phase: Phase,
    task: TaskSpec,
    factories: dict[EnvironmentKind, MachineFactory],
    model: GlmRolloutModel,
    results: Path,
    k: int,
    max_retries: int,
) -> dict[str, Any]:
    directory = results / str(phase)
    timed = {kind: TimedFactory(factory) for kind, factory in factories.items()}
    settings = EngineSettings(
        factories=timed,
        capabilities=PROBE_CAPABILITIES,
        max_turns=MAX_TURNS,
        command_timeout=COMMAND_TIMEOUT,
        cleanup_timeout=CLEANUP_TIMEOUT,
        convention=PlainText(id="plain"),
    )
    plan = TrialPlan(
        item_id=task.id,
        round=0,
        kind=TrialKind.SOLVER,
        k=k,
        max_retries=max_retries,
        retry_backoff=ExponentialBackoff(initial=5, maximum=60),
        evidence_dir=directory,
        ledger=JsonlLedger(directory / "ledger"),
    )
    print(f"PHASE_START {phase} task={task.id} k={k} max_retries={max_retries}", flush=True)
    started = time.monotonic()
    outcomes = await run_trials(task, EXECUTION, plan, settings, model)
    wall_time = time.monotonic() - started
    summary = {
        "phase": str(phase),
        "task_id": task.id,
        "wall_time": wall_time,
        "trials": [outcome_summary(outcome) for outcome in outcomes],
        "evidence": evidence_summary(outcomes),
        "machine_creates": {str(kind): factory.creates for kind, factory in timed.items() if factory.creates},
        "ledger": [entry_to_json(entry) for path in ledger_files(directory / "ledger") for entry in read_entries(path)],
    }
    print(f"PHASE_END {phase} wall={wall_time:.1f}s evidence={json.dumps(summary['evidence'])}", flush=True)
    return summary


async def sandbox_environment(factory: MachineFactory) -> dict[str, Any]:
    """The environment variable names one Iris machine receives, and the GLM token's length there."""
    timed = TimedFactory(factory)
    machine = await timed.create(MachineSpec(source=ShellboxRegistryImage(IMAGE), network=NetworkPolicy.ALLOW))
    try:
        result = await machine.run(Command(argv=("sh", "-c", ENV_NAMES_SCRIPT), timeout=60))
    finally:
        await machine.close()
    names, token_bytes = result.stdout.decode().strip().splitlines()
    return {
        "create": timed.creates,
        "names": names.split(),
        "secret_names": secret_names(names.split()),
        "glm_token_bytes_in_sandbox": int(token_bytes),
    }


def secret_names(names: list[str]) -> list[str]:
    return [name for name in names if SECRET_WORDS & set(name.split("_")) or name.endswith(SECRET_SUFFIXES)]


def upload(results: Path, prefix: str) -> str:
    """Copy ``results`` under ``prefix`` when this task already holds credentials for that store."""
    if prefix.startswith("s3://") and not os.environ.get("AWS_ACCESS_KEY_ID"):
        return "skipped: no AWS credentials in the task environment"
    if not prefix.startswith(("s3://", "gs://")):
        return f"skipped: unsupported prefix {prefix}"
    destination = StoragePath(prefix)
    for path in sorted(p for p in results.rglob("*") if p.is_file()):
        (destination / path.relative_to(results).as_posix()).upload_from(str(path))
    return f"uploaded to {prefix}"


def print_results(results: Path) -> None:
    """Print every file as ``RESULT_FILE <path> <index>/<count> <JSON string chunk>`` lines."""
    for path in sorted(p for p in results.rglob("*") if p.is_file()):
        text = path.read_text()
        chunks = [text[offset : offset + PRINT_CHUNK] for offset in range(0, len(text), PRINT_CHUNK)] or [""]
        for index, chunk in enumerate(chunks):
            print(f"RESULT_FILE {path.relative_to(results)} {index}/{len(chunks)} {json.dumps(chunk)}", flush=True)


def reassemble(logs: Path, results: Path) -> None:
    """Rebuild the results directory from ``RESULT_FILE`` lines in saved ``iris job logs`` output."""
    chunks: dict[str, dict[int, str]] = {}
    counts: dict[str, int] = {}
    for line in logs.read_text().splitlines():
        if "RESULT_FILE " not in line:
            continue
        name, position, payload = line.split("RESULT_FILE ", 1)[1].split(" ", 2)
        index, count = (int(part) for part in position.split("/"))
        chunks.setdefault(name, {})[index] = json.loads(payload)
        counts[name] = count
    for name, parts in chunks.items():
        if sorted(parts) != list(range(counts[name])):
            raise ValueError(f"{name}: have chunks {sorted(parts)} of {counts[name]}")
        target = results / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("".join(parts[index] for index in range(counts[name])))
    print(f"reassembled {len(chunks)} files into {results}")


async def probe(args: argparse.Namespace, token: str, controller_url: str, results: Path) -> dict[str, Any]:
    endpoint = endpoint_in_task(args.relay_job, token, Pool.HIGH)
    print(f"GLM_ENDPOINT relay={args.relay_job} base_url={endpoint.base_url}", flush=True)
    summary: dict[str, Any] = {
        "relay_job": args.relay_job,
        "glm_base_url": endpoint.base_url,
        "image": IMAGE,
        "image_tag": IMAGE_TAG,
        "network": "docker task network=true; shipped IrisMachineFactory refuses NetworkPolicy.DENY",
        "controller_url": controller_url,
        "task_id": os.environ.get("IRIS_TASK_ID"),
        "phases": [],
    }
    async with GlmClient(endpoint) as client:
        model = GlmRolloutModel(client, POLICY)
        summary["phases"].append(await run_phase(Phase.MATH, math_task(), {}, model, results, args.k, 2))
        factories = dict(machine_factories(MachineHost.IRIS, controller_url))
        summary["phases"].append(
            await run_phase(Phase.DOCKER_SHIPPED, docker_task(), factories, model, results, args.k, 1)
        )
        apply_readiness_fix()
        factories = dict(machine_factories(MachineHost.IRIS, controller_url))
        summary["phases"].append(
            await run_phase(Phase.DOCKER_READINESS_FIX, docker_task(), factories, model, results, args.k, 2)
        )
        summary["sandbox_environment"] = await sandbox_environment(factories[EnvironmentKind.DOCKER])
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--relay-job", help="Iris job that registers the glm-5.3 endpoint")
    mode.add_argument("--reassemble", type=Path, help="saved `iris job logs` output to rebuild results from")
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--results-dir", type=Path, default=None)
    parser.add_argument("--upload-prefix", default=None)
    args = parser.parse_args()
    if args.reassemble is not None:
        if args.results_dir is None:
            raise SystemExit("--reassemble needs --results-dir")
        reassemble(args.reassemble, args.results_dir)
        return
    controller_url = os.environ.get(IRIS_CONTROLLER_URL_ENV)
    if not controller_url:
        raise SystemExit(f"{IRIS_CONTROLLER_URL_ENV} is unset; the probe runs only inside an Iris task")
    token = scrub_child_environment()
    run = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    results = args.results_dir
    if results is None:
        output_dir = os.environ.get("IRIS_OUTPUT_DIR")
        if output_dir is None:
            raise SystemExit("--results-dir is required outside an Iris task with IRIS_OUTPUT_DIR")
        results = Path(output_dir) / "cluster_rollout_probe"
    results.mkdir(parents=True, exist_ok=True)
    marin_prefix = os.environ.get("MARIN_PREFIX")
    prefix = args.upload_prefix or (marin_prefix and prefix_join(marin_prefix, f"taskforge/cluster_rollout_probe/{run}"))
    started = time.monotonic()
    summary = asyncio.run(probe(args, token, controller_url, results))
    summary["wall_time"] = time.monotonic() - started
    summary["run"] = run
    (results / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    print_results(results)
    print("RESULT_SUMMARY " + json.dumps(summary, default=str), flush=True)
    print("UPLOAD " + (upload(results, prefix) if prefix else "skipped: no prefix"), flush=True)
    leaked = summary["sandbox_environment"]["secret_names"]
    if leaked:
        raise SystemExit(f"SANDBOX_SECRETS credential-like names reach model-controlled sandboxes: {leaked}")


if __name__ == "__main__":
    main()
