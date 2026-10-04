# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Check tool transport under explicit instruction with two fixed shell probes."""

import json
import logging
from collections.abc import Awaitable, Callable
from dataclasses import asdict
from itertools import pairwise

import httpx
from rigging.filesystem.storage_path import StoragePath, prefix_join
from rolloutengine.contracts import ModelRequest, ModelTurn, RolloutContractError
from rolloutengine.engine import ShellboxRolloutEngine
from shellbox.backends.shellsim.machine import ShellSimMachineFactory
from taskcompendium.environment import EnvironmentFile, EnvironmentKind, EnvironmentSpec
from taskcompendium.grading import numeric_answer
from taskcompendium.models import AnswerType, ConversationInput, EnvironmentRequirements, Source, TaskSpec, TextMessage
from taskcompendium.submission import AnswerFormat, SubmissionConvention

logger = logging.getLogger(__name__)

PREFLIGHT_INSTRUCTION = (
    "Use the shell tool before you answer. First, call shell with command "
    "`cat /workspace/preflight-value.txt`. After the tool response, reply with only the file contents. "
    "Do not guess the contents."
)
PREFLIGHT_PROBES = (
    (PREFLIGHT_INSTRUCTION, 48213),
    (
        PREFLIGHT_INSTRUCTION
        + " Your first response must contain a shell function call in the supplied <tool_call> format. "
        "Give the final answer only after the tool response.",
        73961,
    ),
)


def validate_token_preflight(turns: list[dict], rollout: dict) -> None:
    """Require an actual tool exchange that retains all sampled token IDs."""
    if len(turns) < 2 or not any(message.get("role") == "tool" for message in rollout["messages"]):
        raise ValueError("The model did not complete a two-turn shell tool exchange")
    for earlier, later in pairwise(turns):
        prefix = earlier["result"]["prompt_token_ids"] + earlier["result"]["response_token_ids"]
        if later["result"]["prompt_token_ids"][: len(prefix)] != prefix:
            raise RolloutContractError("The production continuation changed sampled token IDs")
    if rollout["grade"]["reward"] != 1:
        raise ValueError("The real Shellsim preflight task did not pass")


def preflight_task(index: int, instruction: str, value: int) -> TaskSpec:
    return TaskSpec(
        id=f"russell-real-token-preflight-{index}",
        context=ConversationInput(
            events=(
                TextMessage(role="system", content=instruction),
                TextMessage(role="user", content="Read the file and return its contents."),
            )
        ),
        environment=EnvironmentSpec(
            kind=EnvironmentKind.SHELLSIM,
            files=(EnvironmentFile(path="/workspace/preflight-value.txt", content=f"{value}\n".encode()),),
        ),
        environment_requirements=EnvironmentRequirements(),
        answer_type=AnswerType.NUMBER,
        verifier=numeric_answer(value, tolerance_abs=0, tolerance_rel=0),
        source=Source(dataset="russell-token-preflight", revision="3", row=str(index), importer_revision="1"),
    )


async def run_preflight_probe(
    turn: Callable[[ModelRequest], Awaitable[ModelTurn]], client: httpx.AsyncClient, task: TaskSpec, evidence_path: str
) -> tuple[dict, Exception | None]:
    requests: list[dict] = []
    turns: list[dict] = []

    async def save_response(response: httpx.Response) -> None:
        await response.aread()
        requests.append(
            {
                "url": str(response.request.url),
                "request": json.loads(response.request.content),
                "status": response.status_code,
                "response": response.text,
            }
        )

    async def record_turn(request: ModelRequest) -> ModelTurn:
        result = await turn(request)
        turns.append({"request": asdict(request), "result": asdict(result)})
        return result

    engine = ShellboxRolloutEngine(
        record_turn,
        {EnvironmentKind.SHELLSIM: ShellSimMachineFactory()},
        max_turns=4,
        command_timeout=10,
        convention=SubmissionConvention(id="russell-dev", answer_format=AnswerFormat.PLAIN),
    )
    evidence: dict = {
        "requests": requests,
        "turns": turns,
        "status": "failed",
        "fixture": task.model_dump(mode="json"),
    }
    client.event_hooks["response"].append(save_response)
    recorded_error: Exception | None = None
    try:
        evidence["rollout"] = asdict(await engine.run(task))
        validate_token_preflight(turns, evidence["rollout"])
        evidence["status"] = "passed"
    except Exception as error:
        recorded_error = error
        evidence["error"] = f"{type(error).__name__}: {error}"
        if isinstance(error, RolloutContractError):
            evidence["contract_failure"] = True
    finally:
        client.event_hooks["response"].remove(save_response)
        StoragePath(evidence_path).write_text(json.dumps(evidence) + "\n")
    return evidence, recorded_error


async def run_token_preflight(
    turn: Callable[[ModelRequest], Awaitable[ModelTurn]], client: httpx.AsyncClient, output_path: str
) -> None:
    """Run both fixed probes and preserve evidence before applying the suite gate."""
    tasks = [preflight_task(index, instruction, value) for index, (instruction, value) in enumerate(PREFLIGHT_PROBES, 1)]
    manifest = {
        "claim": "tool transport under explicit instruction; ordinary tool adherence remains untested",
        "fixtures": [task.model_dump(mode="json") for task in tasks],
    }
    StoragePath(prefix_join(output_path, "token-preflight-suite.json")).write_text(json.dumps(manifest) + "\n")
    attempts: list[dict] = []
    contract_error: RolloutContractError | None = None
    last_error: Exception | None = None
    for index, task in enumerate(tasks, 1):
        evidence, error = await run_preflight_probe(
            turn, client, task, prefix_join(output_path, f"token-preflight-probe-{index}.json")
        )
        if error is not None:
            last_error = error
            logger.warning("Token preflight probe %d failed: %s", index, evidence["error"])
        if isinstance(error, RolloutContractError):
            contract_error = error
        attempts.append(evidence)

    passed = any(attempt["status"] == "passed" for attempt in attempts) and contract_error is None
    result = {**manifest, "attempts": attempts, "status": "passed" if passed else "failed"}
    StoragePath(prefix_join(output_path, "token-preflight.json")).write_text(json.dumps(result) + "\n")
    if contract_error is not None:
        raise RolloutContractError(
            "The preflight suite observed a token or adapter contract failure"
        ) from contract_error
    if not passed:
        raise ValueError("Neither preflight probe completed a two-turn shell tool exchange") from last_error
