# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Convert Harbor trajectory payloads into the normalized FineStore step contract."""

import json
import logging
from dataclasses import dataclass

from finestore.eval import EvaluationStore, StepRecord

logger = logging.getLogger(__name__)


def _content_to_text(value: object) -> str | None:
    if value is None or isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def trajectory_step_rows(trajectory: dict, *, task: str, doc_id: str, trial_id: str) -> list[StepRecord]:
    """Flatten a Harbor trajectory's top-level steps into normalized rows."""
    records = []
    for step in trajectory.get("steps") or []:
        metrics = step.get("metrics") or {}
        tool_calls = step.get("tool_calls")
        observation = step.get("observation")
        records.append(
            StepRecord(
                task=task,
                doc_id=doc_id,
                trial_id=trial_id,
                step_id=step.get("step_id"),
                source=step.get("source"),
                model_name=step.get("model_name"),
                message=_content_to_text(step.get("message")),
                reasoning_content=step.get("reasoning_content"),
                tool_calls_json=json.dumps(tool_calls, ensure_ascii=False) if tool_calls is not None else None,
                observation_json=json.dumps(observation, ensure_ascii=False) if observation is not None else None,
                prompt_tokens=metrics.get("prompt_tokens"),
                completion_tokens=metrics.get("completion_tokens"),
                cost_usd=metrics.get("cost_usd"),
                prompt_token_ids=metrics.get("prompt_token_ids"),
                completion_token_ids=metrics.get("completion_token_ids"),
                logprobs=metrics.get("logprobs"),
            )
        )
    return records


@dataclass(frozen=True)
class StoredTrajectory:
    """The artifact URI and normalized rows written for one Harbor trajectory."""

    uri: str
    steps: tuple[StepRecord, ...]


def archive_trajectory(
    store: EvaluationStore,
    raw: bytes,
    *,
    task: str,
    doc_id: str,
    trial_id: str,
) -> StoredTrajectory:
    """Store a Harbor trajectory artifact and append its normalized top-level steps."""
    uri = store.add_artifact(
        f"{trial_id}/trajectory.json",
        raw,
        metadata={"task": task, "doc_id": doc_id, "trial_id": trial_id},
    )
    try:
        trajectory = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        logger.warning("trajectory for trial %s is unparseable; storing the artifact without steps", trial_id)
        return StoredTrajectory(uri=uri, steps=())
    steps = tuple(trajectory_step_rows(trajectory, task=task, doc_id=doc_id, trial_id=trial_id))
    store.add_steps(steps)
    return StoredTrajectory(uri=uri, steps=steps)
