# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run TaskSpecs through RolloutEngine inside an Iris task, with GLM-5.3 resolved in-task.

Submit from the worktree root as one CPU job federated to a cluster that hosts a GLM relay and
starts gVisor pods (``cw-rno2a``). ``--relay-job`` names the Iris job that registers the glm-5.3
endpoint on that cluster; it has no default::

    lib/taskforge/.venv/bin/iris --cluster=marin job run --no-wait --no-sync \
      --job-name taskforge-cluster-rollout-probe-NN --target-cluster cw-rno2a --priority interactive \
      --cpu 2 --memory 3GB --timeout 5400 -e GLM_API_TOKEN "$GLM_API_TOKEN" -- \
      bash -c 'cd lib/taskforge && uv run --frozen python scripts/cluster_rollout_probe.py \
        --relay-job <relay-job>'

``GLM_API_TOKEN`` travels in the job's environment (``-e``), so the controller stores it with the
job and the task's ``IRIS_JOB_ENV`` carries it. Iris copies ``IRIS_JOB_ENV`` into every child job,
and each docker machine is a child job the model controls, so ``main`` removes the token (and the
submitter keys Iris forwards) from this process's environment before anything submits a child.

Phases, each ``k`` trials through ``taskforge.validate.trials.run_trials``:

- ``math``: a null-environment numeric task (no machine).
- ``docker``: a task on a digest-pinned public image on
  ``machine_factories(MachineHost.IRIS, controller_url, image_cache=None)`` as shipped, with the
  task's ``IRIS_CONTROLLER_URL``. Its script grader runs in a separate verifier machine started from
  the same image, which receives the agent's ``/workspace/sum.txt`` as an artifact.

Then one machine lists the environment variable names a sandbox receives. The run exits non-zero
after writing its results when any of those names looks like a credential (``secret_names``): on
CoreWeave, cluster ``task_env`` object-store keys reach every sandbox, which a model-controlled
machine with network must not see.

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
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from rigging.filesystem.storage_path import StoragePath, prefix_join
from rigging.timing import ExponentialBackoff
from rolloutengine.spec import LoweredTaskSpec
from shellbox.image import RegistryImage
from shellbox.machine import Backend, Command, Machine, MachineFactory, MachineSpec, NetworkPolicy
from taskcompendium.models import (
    AnswerType,
    ArtifactKind,
    MissingArtifactPolicy,
    PlainText,
    Source,
    StdoutReward,
    VerifierArtifact,
)
from verifyit.spec import NumericSpec

from taskforge.ledger.jsonl import JsonlLedger, ledger_files, read_entries
from taskforge.llm.client import GlmClient, Pool, endpoint_in_task
from taskforge.llm.policy import LLMPolicy
from taskforge.llm.recording import CallLedger
from taskforge.llm.rollout_model import GlmRolloutModel
from taskforge.queue.job import (
    GLM_TOKEN_KEY,
    IRIS_CONTROLLER_URL_ENV,
    IRIS_OUTPUT_DIR_ENV,
    SUBMITTER_KEYS,
    scrub_child_environment,
    secret_names,
)
from taskforge.sandbox.factories import MachineHost, factory_capabilities, machine_factories
from taskforge.spec.draft import (
    answer_grader,
    assemble,
    file,
    grader_environment,
    lower,
    machine,
    requirements,
    script_grader,
    session,
)
from taskforge.validate.evidence import Complete, Evidence
from taskforge.validate.outcome import Graded, Outcome, TrialKind
from taskforge.validate.trials import Deadlines, EngineSettings, TrialPlan, run_trials

IMAGE = "docker.io/library/python@sha256:02108f5d322dd89f1c9e552442c25acb0543dfdbc455693a5599624f20d9155d"
IMAGE_TAG = "python:3.12-slim (OCI index digest, resolved 2026-10-05)"
MATH_ANSWER = "395"
NUMBERS = "12\n7\n30\n11\n"
NUMBERS_SUM = 60
SUM_PATH = "/workspace/sum.txt"
CHECK_SCRIPT = f'v=$(tr -d " \\n" < {SUM_PATH})\nif [ "$v" = {NUMBERS_SUM} ]; then echo 1; else echo 0; fi\n'
# Prints names only; `env | cut` would leak fragments of multi-line values into evidence.
ENV_NAMES_SCRIPT = (
    f"awk 'BEGIN{{for(k in ENVIRON) print k}}' | sort | tr '\\n' ' '; echo; printenv {GLM_TOKEN_KEY} | wc -c"
)
POLICY = LLMPolicy(max_continuations=0)
MAX_TURNS = 12
COMMAND_TIMEOUT = 120
TOOL_TURN_TIMEOUT = 240
MODEL_TURN_TIMEOUT = 600
CLEANUP_TIMEOUT = 120
STARTUP_TIMEOUT = 900
VERIFIER_TIMEOUT = 300
DEADLINES = Deadlines(total_turn_timeout=1800, attempt_timeout=2400)
TOKEN_CONTRACT_RETRIES = 2
# The builder's session; EngineSettings and DEADLINES replace all but the verifier deadline.
SESSION = session(
    max_turns=MAX_TURNS,
    model_turn_timeout=MODEL_TURN_TIMEOUT,
    command_timeout=COMMAND_TIMEOUT,
    tool_turn_timeout=TOOL_TURN_TIMEOUT,
    total_turn_timeout=DEADLINES.total_turn_timeout,
    attempt_timeout=DEADLINES.attempt_timeout,
    verifier_timeout=VERIFIER_TIMEOUT,
    cleanup_timeout=CLEANUP_TIMEOUT,
)
# Characters of a result file per log line, so log storage keeps each line whole.
PRINT_CHUNK = 4000


class Phase(StrEnum):
    MATH = "math"
    DOCKER = "docker"


@dataclass
class TimedFactory:
    """Delegates to ``factory`` and records how long each ``create`` took and how it ended."""

    factory: MachineFactory
    creates: list[dict[str, Any]] = field(default_factory=list)

    @property
    def backend(self) -> Backend:
        return self.factory.backend

    async def create(self, spec: MachineSpec) -> Machine:
        started = time.monotonic()
        try:
            machine = await self.factory.create(spec)
        except Exception as error:
            self.creates.append({"ok": False, "seconds": time.monotonic() - started, "error": repr(error)[:1000]})
            raise
        self.creates.append({"ok": True, "seconds": time.monotonic() - started})
        return machine


def source(row: str) -> Source:
    return Source(dataset="taskforge-cluster-rollout-probe", revision="1", row=row, importer_revision="1")


def math_task(factories: dict[str, MachineFactory]) -> LoweredTaskSpec:
    task = assemble(
        "cluster-math",
        "What is 17 * 23 + 4? Reply with only the number, nothing else.",
        AnswerType.NUMBER,
        PlainText(),
        answer_grader(NumericSpec(expected=MATH_ANSWER, tolerance_abs=0, tolerance_rel=0)),
        source("math"),
        environment=None,
    )
    return lower(
        task, host=MachineHost.IRIS, task_machine=None, verifier_machine=None, session=SESSION, factories=factories
    )


def docker_task(factories: dict[str, MachineFactory]) -> LoweredTaskSpec:
    """A file task on ``IMAGE`` whose script grader runs in a verifier machine from the same image."""
    task = assemble(
        "cluster-docker-file",
        "The file /workspace/numbers.txt holds one integer per line. Use the shell tool to write their sum, "
        f"as a single integer, to {SUM_PATH}. Say when you are done.",
        AnswerType.FILE,
        PlainText(),
        script_grader(
            ("sh", "/tests/check.sh"),
            StdoutReward(),
            environment=grader_environment(IMAGE),
            answer_path=None,
            timeout=VERIFIER_TIMEOUT,
            files=(file("check.sh", CHECK_SCRIPT),),
            artifacts=(
                VerifierArtifact(
                    source=SUM_PATH, target=SUM_PATH, kind=ArtifactKind.FILE, missing=MissingArtifactPolicy.SKIP
                ),
            ),
        ),
        source("docker-file"),
        environment=requirements(image=IMAGE),
        files=(file("workspace/numbers.txt", NUMBERS),),
    )
    settings = machine(startup_timeout=STARTUP_TIMEOUT)
    return lower(
        task,
        host=MachineHost.IRIS,
        task_machine=settings,
        verifier_machine=settings,
        session=SESSION,
        factories=factories,
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
        "timed_out": stats.timed_out,
    }


async def run_phase(
    phase: Phase,
    lowered: LoweredTaskSpec,
    factories: dict[str, MachineFactory],
    client: GlmClient,
    results: Path,
    k: int,
    max_retries: int,
) -> dict[str, Any]:
    directory = results / str(phase)
    task = lowered.task
    ledger = JsonlLedger(directory / "ledger")
    model = GlmRolloutModel(client, POLICY, CallLedger(ledger, task.id, 0, str(TrialKind.SOLVER)))
    timed = {backend: TimedFactory(factory) for backend, factory in factories.items()}
    settings = EngineSettings(
        factories=timed,
        capabilities=factory_capabilities(MachineHost.IRIS),
        max_turns=MAX_TURNS,
        command_timeout=COMMAND_TIMEOUT,
        tool_turn_timeout=TOOL_TURN_TIMEOUT,
        model_turn_timeout=MODEL_TURN_TIMEOUT,
        cleanup_timeout=CLEANUP_TIMEOUT,
    )
    plan = TrialPlan(
        item_id=task.id,
        round=0,
        kind=TrialKind.SOLVER,
        k=k,
        deadlines=DEADLINES,
        max_retries=max_retries,
        token_contract_retries=TOKEN_CONTRACT_RETRIES,
        retry_backoff=ExponentialBackoff(initial=5, maximum=60),
        evidence_dir=directory,
        ledger=ledger,
        first_attempt=0,
    )
    print(f"PHASE_START {phase} task={task.id} k={k} max_retries={max_retries}", flush=True)
    started = time.monotonic()
    outcomes = await run_trials(lowered, plan, settings, model)
    wall_time = time.monotonic() - started
    summary = {
        "phase": str(phase),
        "task_id": task.id,
        "wall_time": wall_time,
        "trials": [outcome_summary(outcome) for outcome in outcomes],
        "evidence": evidence_summary(outcomes),
        "machine_creates": {backend: factory.creates for backend, factory in timed.items() if factory.creates},
        "ledger": [entry.to_json() for path in ledger_files(directory / "ledger") for entry in read_entries(path)],
    }
    print(f"PHASE_END {phase} wall_time={wall_time:.1f}s evidence={json.dumps(summary['evidence'])}", flush=True)
    return summary


async def sandbox_environment(factory: MachineFactory) -> dict[str, Any]:
    """The environment variable names one Iris machine receives, and the GLM token's length there."""
    timed = TimedFactory(factory)
    machine = await timed.create(MachineSpec(source=RegistryImage(IMAGE), network=NetworkPolicy.ALLOW))
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
        "network": "docker task and verifier machines NetworkPolicy.DENY; environment probe machine ALLOW",
        "controller_url": controller_url,
        "task_id": os.environ.get("IRIS_TASK_ID"),
        "phases": [],
    }
    async with GlmClient(endpoint) as client:
        factories = dict(machine_factories(MachineHost.IRIS, controller_url, image_cache=None))
        summary["phases"].append(
            await run_phase(Phase.MATH, math_task(factories), factories, client, results, args.k, 2)
        )
        summary["phases"].append(
            await run_phase(Phase.DOCKER, docker_task(factories), factories, client, results, args.k, 2)
        )
        summary["sandbox_environment"] = await sandbox_environment(factories[Backend.GVISOR.value])
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
    token = scrub_child_environment((GLM_TOKEN_KEY, *SUBMITTER_KEYS))[GLM_TOKEN_KEY]
    run = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    results = args.results_dir
    if results is None:
        output_dir = os.environ.get(IRIS_OUTPUT_DIR_ENV)
        if output_dir is None:
            raise SystemExit(f"--results-dir is required outside an Iris task with {IRIS_OUTPUT_DIR_ENV}")
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
