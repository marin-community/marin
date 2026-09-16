# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Collect semantic attempts from independently configured Harbor lowerings."""

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

import msgspec

from taskcompendium.harbor.runner import run_trial
from taskcompendium.models import GradingResult, Outcome


@dataclass(frozen=True)
class GenerationRequest:
    """One lowering and consumer-owned launch, with rollout identity."""

    task_dir: Path
    execution: dict[str, Any]
    instance_id: str
    repetition_id: int


@dataclass(frozen=True)
class GeneratedAttempt:
    """A retained trial, including failures that have no semantic reward."""

    instance_id: str
    repetition_id: int
    task_dir: str
    trial_dir: str
    status: Outcome
    reward: float | None
    step_names: tuple[str, ...]
    step_results: tuple[GradingResult | None, ...]
    step_exceptions: tuple[dict[str, Any] | None, ...]
    messages: tuple[dict[str, Any], ...]
    tools: tuple[dict[str, Any], ...]
    exception: dict[str, Any] | None
    specification_sha256: str | None
    source: dict[str, Any] | None
    model_name: str | None
    chat_template_kwargs: dict[str, Any]
    token_provenance: str = "reconstructed"


async def _generate_attempt(request: GenerationRequest, trials_dir: Path) -> GeneratedAttempt:
    trial_name = uuid4().hex
    trial_dir = trials_dir.resolve() / trial_name
    steps: list[GradingResult | None] = []
    step_exceptions: list[dict[str, Any] | None] = []
    messages: tuple[dict[str, Any], ...] = ()
    tools: tuple[dict[str, Any], ...] = ()
    manifest: dict[str, Any] = {}
    exception = None
    reward = None
    status = Outcome.INFRA_ERROR
    try:
        manifest = json.loads((request.task_dir / "manifest.json").read_text())
        result = await run_trial(request.task_dir, request.execution, trials_dir, trial_name)
        if result.exception_info is not None:
            exception = result.exception_info.model_dump(mode="json")
        names = manifest["step_names"]
        steps = [None] * len(names)
        step_exceptions = [None] * len(names)
        for step in result.step_results or []:
            index = names.index(step.step_name)
            if step.exception_info is not None:
                step_exceptions[index] = step.exception_info.model_dump(mode="json")
                if exception is None:
                    exception = step_exceptions[index]
        roots = [trial_dir / "steps" / name for name in names] if len(names) > 1 else [trial_dir]
        for root in roots:
            transcript_path = root / "agent/transcript.json"
            if transcript_path.exists():
                # Each ordered-step transcript already contains preceding history.
                messages = tuple(json.loads(transcript_path.read_text()))
                tools = tuple(json.loads((root / "agent/tools.json").read_text()))
        contexts = [step.agent_result for step in result.step_results] if result.step_results else [result.agent_result]
        if not messages and contexts[-1] is not None:
            messages = tuple(contexts[-1].metadata.get("all_messages", []))
            tools = tuple(contexts[-1].metadata.get("tools", []))
        for index, root in enumerate(roots):
            grading_path = root / "verifier/taskcompendium-result.json"
            if grading_path.exists():
                steps[index] = msgspec.json.decode(grading_path.read_bytes(), type=GradingResult)
        failures = [
            step for step in steps if step is not None and (step.status != Outcome.GRADED or step.reward is None)
        ]
        if failures:
            status = failures[0].status if failures[0].status != Outcome.GRADED else Outcome.INFRA_ERROR
        elif exception is None and all(step is not None for step in steps) and result.verifier_result is not None:
            reward = result.verifier_result.rewards["reward"]
            status = Outcome.GRADED
        elif exception is None:
            exception = {
                "exception_type": "MissingSemanticResult",
                "step_names": [name for name, step in zip(names, steps, strict=True) if step is None],
            }
    except Exception as error:
        # Retain launch and archive errors alongside trial failures; cancellation
        # still propagates, and no failure is converted into a score.
        exception = {"exception_type": type(error).__name__, "exception_message": str(error)}
    return GeneratedAttempt(
        request.instance_id,
        request.repetition_id,
        str(request.task_dir.resolve()),
        str(trial_dir),
        status,
        reward,
        tuple(manifest.get("step_names", [])),
        tuple(steps),
        tuple(step_exceptions),
        messages,
        tools,
        exception,
        manifest.get("specification_sha256"),
        manifest.get("source"),
        request.execution.get("agent", {}).get("model_name"),
        request.execution.get("agent", {}).get("kwargs", {}).get("chat_template_kwargs") or {},
    )


async def generate_attempts(
    requests: list[GenerationRequest], trials_dir: Path, *, concurrency: int
) -> list[GeneratedAttempt]:
    """Run a mixed batch in request order without starting a training runtime.

    Args:
        requests: Per-lowering launches and unique instance/repetition identities.
        trials_dir: Parent directory for Harbor's retained trial artifacts.
        concurrency: Maximum simultaneous trials.

    Returns:
        Semantic attempt records, including ungraded failures with null reward.
    """
    if concurrency < 1:
        raise ValueError("concurrency must be positive")
    identities = {(request.instance_id, request.repetition_id) for request in requests}
    if len(identities) != len(requests):
        raise ValueError("Generation request identities must be unique")
    if any(not request.instance_id or request.repetition_id < 0 for request in requests):
        raise ValueError("Generation requests require an instance ID and nonnegative repetition ID")
    semaphore = asyncio.Semaphore(concurrency)

    async def generate(request: GenerationRequest) -> GeneratedAttempt:
        async with semaphore:
            return await _generate_attempt(request, trials_dir)

    return list(await asyncio.gather(*(generate(request) for request in requests)))


def write_attempts(attempts: list[GeneratedAttempt], destination: Path) -> None:
    """Write semantic JSONL separately from numeric trainer batches."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as output:
        for attempt in attempts:
            output.write(msgspec.json.encode(attempt) + b"\n")
