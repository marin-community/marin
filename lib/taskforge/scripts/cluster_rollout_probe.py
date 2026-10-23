# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run TaskSpecs through RolloutEngine inside an Iris task, with GLM-5.3 resolved in-task.

Submit from the worktree root as one CPU job federated to a cluster that hosts a GLM relay and
starts gVisor pods (``cw-rno2a``). ``--relay-job`` names the Iris job that registers the glm-5.3
endpoint on that cluster; it has no default::

    uv run iris --cluster=marin job run --no-wait --max-retries 1 \
      --job-name taskforge-cluster-rollout-probe-NN --target-cluster cw-rno2a --priority interactive \
      --cpu 2 --memory 3GB --timeout 5400 -e GLM_HIGH_TOKEN "$GLM_HIGH_TOKEN" -- \
      uv run --package marin-taskforge --frozen lib/taskforge/scripts/cluster_rollout_probe.py \
        --relay-job <relay-job>

``GLM_HIGH_TOKEN``, the high pool's bearer token, travels in the job's environment (``-e``), so the
controller stores it with the job and the task's ``IRIS_JOB_ENV`` carries it. Iris copies
``IRIS_JOB_ENV`` into every child job, and each docker machine is a child job the model controls,
so ``main`` removes the token (and the submitter keys Iris forwards) from this process's
environment before anything submits a child.

Phases, each ``k`` trials through ``taskforge.stage50_validate.trials.run_trials``:

- ``math``: a null-environment numeric task (no machine).
- ``docker``: a task on a digest-pinned public image on
  ``machine_factories((Backend.GVISOR,), controller_url)`` as shipped, with the
  task's ``IRIS_CONTROLLER_URL``. Its script grader runs in a separate verifier machine started from
  the same image, which receives the agent's ``/workspace/sum.txt`` as an artifact.

Then one machine lists the environment variable names a sandbox receives. The run exits non-zero
after writing its results when any of those names looks like a credential (``secret_names``): on
CoreWeave, cluster ``task_env`` object-store keys reach every sandbox, which a model-controlled
machine with network must not see.

Everything goes to ``$IRIS_OUTPUT_DIR/cluster_rollout_probe``: ``summary.json``, and under
``<phase>/`` the trial evidence ``run_trials`` writes and ``ledger/<workpiece>.jsonl``. Iris
archives that directory once per task attempt, and the archive is how the results come back:
``iris attempt describe`` names it (``lib/iris/docs/task-outputs.md``). The summary is also
printed as one ``RESULT_SUMMARY`` log line.
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

from rolloutengine.models import TaskRolloutSpec
from shellbox.image import RegistryImage
from shellbox.machine import (
    Backend,
    Command,
    ExecutionRequirements,
    Machine,
    MachineFactory,
    MachineSpec,
    NetworkPolicy,
)
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
from taskforge.llm.glm_client import GLM_MODEL, GlmClient, Pool, RelayEndpoint
from taskforge.llm.policy import LLMPolicy, LLMProviderConfig
from taskforge.llm.recording import CallLedger
from taskforge.llm.rollout_model import GlmRolloutModel
from taskforge.sandbox.factories import MachineHost, machine_factories
from taskforge.spec.draft import (
    answer_grader,
    assemble,
    file,
    grader_environment,
    requirements,
    rollout_limits,
    script_grader,
)
from taskforge.stage50_validate.evidence import Evidence
from taskforge.stage50_validate.lowering import lower
from taskforge.stage50_validate.outcome import Graded, Outcome, TrialKind
from taskforge.stage50_validate.trials import EngineSettings, RetryBackoff, TrialPlan, rollout_steps, run_trials

GLM_TOKEN_ENV = "GLM_HIGH_TOKEN"
IRIS_CONTROLLER_URL_ENV = "IRIS_CONTROLLER_URL"
# Iris copies these from the submitting process into every child job (iris.cluster.types.EnvironmentSpec).
SUBMITTER_KEYS = ("HF_TOKEN", "WANDB_API_KEY")
IRIS_JOB_ENV = "IRIS_JOB_ENV"
IRIS_OUTPUT_DIR_ENV = "IRIS_OUTPUT_DIR"
RESULTS_DIR = "cluster_rollout_probe"
IMAGE = "docker.io/library/python@sha256:02108f5d322dd89f1c9e552442c25acb0543dfdbc455693a5599624f20d9155d"
IMAGE_TAG = "python:3.12-slim (OCI index digest, resolved 2026-10-05)"
MATH_ANSWER = "395"
NUMBERS = "12\n7\n30\n11\n"
NUMBERS_SUM = 60
SUM_PATH = "/workspace/sum.txt"
CHECK_SCRIPT = f'v=$(tr -d " \\n" < {SUM_PATH})\nif [ "$v" = {NUMBERS_SUM} ]; then echo 1; else echo 0; fi\n'
# Prints names only; `env | cut` would leak fragments of multi-line values into evidence.
ENV_NAMES_SCRIPT = (
    f"awk 'BEGIN{{for(k in ENVIRON) print k}}' | sort | tr '\\n' ' '; echo; printenv {GLM_TOKEN_ENV} | wc -c"
)
# A sandbox environment name is treated as a credential when one of its ``_``-separated words is in
# SECRET_WORDS or it ends with one of SECRET_SUFFIXES (so GPG_KEY and TOKENIZERS_* are not).
SECRET_WORDS = frozenset({"SECRET", "TOKEN", "PASSWORD"})
SECRET_SUFFIXES = ("KEY_ID", "API_KEY", "ACCESS_KEY")
POLICY = LLMPolicy(max_tokens=None, max_continuations=None)
PROVIDER = LLMProviderConfig(
    model=GLM_MODEL,
    max_attempts=8,
    hold_timeout=3600.0,
    backoff_initial=1.0,
    backoff_maximum=60.0,
    backoff_factor=2.0,
    stall_timeout=600.0,
    max_connections=512,
    structured_repairs=None,
    context_window=262144,
)
COMMAND_TIMEOUT = 120
TOOL_TURN_TIMEOUT = 240
MODEL_TURN_TIMEOUT = 600
CLEANUP_TIMEOUT = 120
VERIFIER_TIMEOUT = 300
LIMITS = rollout_limits(
    max_turns=None,
    model_turn_timeout=MODEL_TURN_TIMEOUT,
    tool_turn_timeout=TOOL_TURN_TIMEOUT,
    total_turn_timeout=1800,
    attempt_timeout=2400,
    verifier_timeout=VERIFIER_TIMEOUT,
    cleanup_timeout=CLEANUP_TIMEOUT,
)


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

    def check_environment(self, requirements: ExecutionRequirements, spec: MachineSpec) -> None:
        self.factory.check_environment(requirements, spec)

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


def source(row: str) -> Source:
    return Source(dataset="taskforge-cluster-rollout-probe", revision="1", row=row, importer_revision="1")


def math_task(factories: dict[str, MachineFactory]) -> TaskRolloutSpec:
    task = assemble(
        "cluster-math",
        "What is 17 * 23 + 4? Reply with only the number, nothing else.",
        AnswerType.NUMBER,
        PlainText(),
        answer_grader(NumericSpec(expected=MATH_ANSWER, tolerance_abs=0, tolerance_rel=0)),
        source("math"),
        environment=None,
    )
    return lower(task, MachineHost.IRIS, factories, LIMITS)


def docker_task(factories: dict[str, MachineFactory]) -> TaskRolloutSpec:
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
    return lower(task, MachineHost.IRIS, factories, LIMITS)


def assistant_messages(outcome: Outcome) -> list[dict[str, Any]]:
    """The model's messages across the rollout's conversations; none without a rollout."""
    rollout = outcome.rollout
    if rollout is None:
        return []
    return [step.turn.message for step in rollout_steps(rollout)]


def shell_calls(outcome: Outcome) -> list[str]:
    return [
        call["function"]["arguments"]
        for message in assistant_messages(outcome)
        for call in message.get("tool_calls") or []
    ]


def final_message(outcome: Outcome) -> str | None:
    replies = [message.get("content") for message in assistant_messages(outcome)]
    return replies[-1] if replies else None


def response_tokens(outcome: Outcome) -> int:
    """The completion tokens the rollout's turns report in their usage; 0 without a rollout."""
    rollout = outcome.rollout
    if rollout is None:
        return 0
    return sum(step.turn.usage.completion_tokens for step in rollout_steps(rollout) if step.turn.usage is not None)


def outcome_summary(outcome: Outcome) -> dict[str, Any]:
    rollout = outcome.rollout
    common: dict[str, Any] = {
        "turns": 0 if rollout is None else len(rollout_steps(rollout)),
        "stop_reason": None if rollout is None else rollout.stop_reason,
        "response_tokens": response_tokens(outcome),
        "shell_calls": shell_calls(outcome),
        "final_message": final_message(outcome),
    }
    if isinstance(outcome, Graded):
        grade = outcome.grade
        return {
            "outcome": "graded",
            "reward": outcome.reward,
            "candidate_code_error": outcome.candidate_code_error,
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
    ungraded = evidence.ungraded
    stats = evidence.reward_stats(TrialKind.SOLVER)
    return {
        "status": "complete" if not ungraded else {"incomplete": dict(ungraded)},
        "graded": stats.graded,
        "mean_reward": stats.mean_reward,
        "solved": stats.solved,
        "timed_out": stats.timed_out,
        "candidate_code_errors": stats.candidate_code_errors,
    }


async def run_phase(
    phase: Phase,
    spec: TaskRolloutSpec,
    factories: dict[str, MachineFactory],
    client: GlmClient,
    results: Path,
    k: int,
    max_retries: int,
) -> dict[str, Any]:
    directory = results / str(phase)
    task = spec.task
    ledger = JsonlLedger(directory / "ledger")
    model = GlmRolloutModel(client, POLICY, CallLedger(ledger, task.id, 0, str(TrialKind.SOLVER)))
    timed = {backend: TimedFactory(factory) for backend, factory in factories.items()}
    settings = EngineSettings(factories=timed, command_timeout=COMMAND_TIMEOUT, limits=LIMITS)
    plan = TrialPlan(
        workpiece_id=task.id,
        round=0,
        kind=TrialKind.SOLVER,
        k=k,
        max_retries=max_retries,
        retry_backoff=RetryBackoff(initial=5, maximum=60, factor=2, jitter=0.1),
        evidence_dir=directory,
        ledger=ledger,
        first_attempt=0,
    )
    print(f"PHASE_START {phase} task={task.id} k={k} max_retries={max_retries}", flush=True)
    started = time.monotonic()
    outcomes = await run_trials(spec, plan, settings, model)
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


def secret_names(names: list[str]) -> list[str]:
    return [name for name in names if SECRET_WORDS & set(name.split("_")) or name.endswith(SECRET_SUFFIXES)]


async def probe(args: argparse.Namespace, token: str, controller_url: str, results: Path) -> dict[str, Any]:
    relay = RelayEndpoint(args.relay_job, token, Pool.HIGH)
    endpoint = relay.current()
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
    async with GlmClient(relay, PROVIDER) as client:
        factories = dict(machine_factories((Backend.GVISOR,), controller_url))
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
    parser.add_argument("--relay-job", required=True, help=f"Iris job that registers the {GLM_MODEL} endpoint")
    parser.add_argument("--k", type=int, default=3)
    args = parser.parse_args()
    controller_url = os.environ.get(IRIS_CONTROLLER_URL_ENV)
    output_dir = os.environ.get(IRIS_OUTPUT_DIR_ENV)
    if not controller_url or not output_dir:
        raise SystemExit(
            f"{IRIS_CONTROLLER_URL_ENV} or {IRIS_OUTPUT_DIR_ENV} is unset; the probe runs only inside an Iris task"
        )
    token = scrub_child_environment()
    run = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    results = Path(output_dir) / RESULTS_DIR
    results.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    summary = asyncio.run(probe(args, token, controller_url, results))
    summary["wall_time"] = time.monotonic() - started
    summary["run"] = run
    (results / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    print("RESULT_SUMMARY " + json.dumps(summary, default=str), flush=True)
    leaked = summary["sandbox_environment"]["secret_names"]
    if leaked:
        raise SystemExit(f"SANDBOX_SECRETS credential-like names reach model-controlled sandboxes: {leaked}")


if __name__ == "__main__":
    main()
