# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Check the production model adapter with a real two-turn shell task."""

import json
from collections.abc import Awaitable, Callable
from dataclasses import asdict
from itertools import pairwise

import httpx
from rigging.filesystem.storage_path import StoragePath, prefix_join
from rolloutengine.contracts import ModelRequest, ModelTurn
from rolloutengine.engine import ShellboxRolloutEngine
from shellbox.backends.shellsim.machine import ShellSimMachineFactory
from taskcompendium.environment import EnvironmentFile, EnvironmentKind, EnvironmentSpec
from taskcompendium.grading import numeric_answer
from taskcompendium.models import AnswerType, ConversationInput, EnvironmentRequirements, Source, TaskSpec, TextMessage
from taskcompendium.submission import AnswerFormat, SubmissionConvention


def validate_token_preflight(turns: list[dict], rollout: dict) -> None:
    """Require an actual tool exchange that retains all sampled token IDs."""
    if len(turns) < 2 or not any(message.get("role") == "tool" for message in rollout["messages"]):
        raise ValueError("The model did not complete a two-turn shell tool exchange")
    for earlier, later in pairwise(turns):
        prefix = earlier["result"]["prompt_token_ids"] + earlier["result"]["response_token_ids"]
        if later["result"]["prompt_token_ids"][: len(prefix)] != prefix:
            raise ValueError("The production continuation changed sampled token IDs")
    if rollout["grade"]["reward"] != 1:
        raise ValueError("The real Shellsim preflight task did not pass")


async def run_token_preflight(
    turn: Callable[[ModelRequest], Awaitable[ModelTurn]], client: httpx.AsyncClient, output_path: str
) -> None:
    """Save wire and token evidence and raise if the actual tool exchange fails."""
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

    task = TaskSpec(
        id="russell-real-token-preflight",
        context=ConversationInput(
            events=(
                TextMessage(
                    role="user",
                    content="Read /workspace/preflight-value.txt with the shell tool, then reply with only its content.",
                ),
            )
        ),
        environment=EnvironmentSpec(
            kind=EnvironmentKind.SHELLSIM,
            files=(EnvironmentFile(path="/workspace/preflight-value.txt", content=b"48213\n"),),
        ),
        environment_requirements=EnvironmentRequirements(),
        answer_type=AnswerType.NUMBER,
        verifier=numeric_answer(48213, tolerance_abs=0, tolerance_rel=0),
        source=Source(dataset="russell-token-preflight", revision="2", row="0", importer_revision="1"),
    )
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
    try:
        evidence["rollout"] = asdict(await engine.run(task))
        validate_token_preflight(turns, evidence["rollout"])
        evidence["status"] = "passed"
    except Exception as error:
        evidence["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        client.event_hooks["response"].remove(save_response)
        StoragePath(prefix_join(output_path, "token-preflight.json")).write_text(json.dumps(evidence) + "\n")
