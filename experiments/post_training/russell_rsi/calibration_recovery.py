# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Complete calibration from pinned original attempts and reviewed recovery evidence."""

import hashlib
import json
import math
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

from fray.types import ResourceConfig
from marin.execution.artifact import Artifact
from marin.execution.lazy import ArtifactStep, artifact_identity
from marin.execution.remote import remote
from marin.training.training import LevanterCheckpoint
from rigging.filesystem.storage_path import StoragePath, prefix_join
from rigging.runtime_bundle import RuntimeBundle

from experiments.post_training.russell_rsi.bootstrap_loop import ATTEMPTS_PER_TASK, CALIBRATION_TEMPERATURE, write_once
from experiments.post_training.russell_rsi.repair_tasks import canonical_sha256, pinned_bytes


@dataclass(frozen=True)
class PinnedFile:
    uri: str
    sha256: str

    def read_bytes(self) -> bytes:
        return pinned_bytes(self.uri, self.sha256)

    def read_json(self) -> dict:
        return json.loads(self.read_bytes())


@dataclass(frozen=True)
class CalibrationRecoveryConfig:
    original_summary: PinnedFile
    original_traces: PinnedFile
    original_tasks: PinnedFile
    original_bank: PinnedFile
    review: PinnedFile
    replays: tuple[PinnedFile, ...]
    replacement_issuance: PinnedFile
    replacement_tasks: PinnedFile
    replacement_summary: PinnedFile
    replacement_traces: PinnedFile
    model_identity: str
    model_uri: str
    tokenizer: str
    tokenizer_revision: str
    bank_identity: str
    runtime_source: str
    runtime_bundle: RuntimeBundle
    runtime_module_hashes: dict[str, str]
    output_path: str


def graded_reward(record: dict) -> float:
    grade = record["grade"]
    reward = grade["reward"]
    if (
        grade["status"] != "graded"
        or grade.get("error") is not None
        or grade.get("failure") is not None
        or type(reward) not in (int, float)
        or not math.isfinite(reward)
        or not 0 <= reward <= 1
    ):
        raise ValueError("Recovery requires a finite grade, without an execution failure")
    return reward


def replay_grade(original: dict, replay: dict, review: dict) -> dict:
    """Compare saved actions and messages, with only exact reviewed observation changes."""
    recovered = replay["rollout"]
    if original["interrupted_operation"] != "grade" or original["grade"]["reward"] is not None:
        raise ValueError("A replay must complete an original grading interruption")
    if replay["model_requests"] != 0 or replay["execution_error"] is not None:
        raise ValueError("A grading replay cannot request another model candidate")
    if recovered["task_id"] != original["task_id"] or len(recovered["steps"]) != len(original["steps"]):
        raise ValueError("Replay task or saved-turn count changed")
    exceptions = {item["step_index"]: item for item in review["observation_exceptions"]}
    if len(exceptions) != len(review["observation_exceptions"]):
        raise ValueError("Duplicate observation exceptions")
    replacements = []
    for index, (before, after) in enumerate(zip(original["steps"], recovered["steps"], strict=True)):
        if before["turn"] != after["turn"]:
            raise ValueError("Replay changed a saved model turn")
        before_observations = before["transition"]["observations"]
        after_observations = after["transition"]["observations"]
        exception = exceptions.pop(index, None)
        if exception is None:
            if before_observations != after_observations:
                raise ValueError("Replay changed an unreviewed observation")
            continue
        calls = before["turn"]["message"]["tool_calls"]
        if (
            before_observations != exception["original"]
            or after_observations != exception["replay"]
            or len(calls) != 1
            or calls[0]["id"] != review["tool_call_id"]
            or json.loads(calls[0]["function"]["arguments"])["command"] != review["exact_command"]
        ):
            raise ValueError("Replay does not match the exact reviewed command and observations")
        replacements.extend(zip(after_observations, before_observations, strict=True))
    if exceptions:
        raise ValueError("Observation exception identifies a missing step")

    def original_messages(messages: list[dict]) -> list[dict]:
        return [next((before for after, before in replacements if message == after), message) for message in messages]

    for before, after in zip(original["steps"], recovered["steps"], strict=True):
        if before["messages"] != original_messages(after["messages"]):
            raise ValueError("Replay changed a model message history")
    if original["messages"] != original_messages(recovered["messages"]):
        raise ValueError("Replay changed the final message history")
    for field in ("prompt_token_ids", "response_token_ids", "loss_mask"):
        if original[field] != recovered[field]:
            raise ValueError(f"Replay changed {field}")
    stop_difference = review.get("stop_reason_difference")
    if stop_difference is None:
        if original["stop_reason"] != recovered["stop_reason"]:
            raise ValueError("Replay changed an unreviewed stop reason")
    elif (original["stop_reason"], recovered["stop_reason"]) != (
        stop_difference["original"],
        stop_difference["replay"],
    ):
        raise ValueError("Replay stop reason does not match the reviewed interruption")
    if graded_reward(recovered) != review["accepted_reward"]:
        raise ValueError("Replay grade differs from the reviewed result")
    return recovered["grade"]


def recover_calibration(config: CalibrationRecoveryConfig) -> None:
    """Write eight grades per task and preserve the source of each completed attempt."""
    from taskcompendium.parquet import read_tasks  # noqa: PLC0415

    summary = config.original_summary.read_json()
    if summary["model_identity"] != config.model_identity or summary["tasks_identity"] != config.bank_identity:
        raise ValueError("Original calibration checkpoint or task-bank identity changed")
    bank = config.original_bank.read_json()
    with TemporaryDirectory() as directory:
        task_path = Path(directory) / "train.parquet"
        task_path.write_bytes(config.original_tasks.read_bytes())
        rows = list(read_tasks(str(task_path)))
        tasks = {task.id: task for task in rows}
    if (
        len(rows) != len(tasks)
        or len(tasks) != len(bank["tasks"])
        or set(tasks) != {row["task_id"] for row in bank["tasks"]}
    ):
        raise ValueError("Original task bank does not identify the pinned task rows")
    for row in bank["tasks"]:
        if canonical_sha256(tasks[row["task_id"]].model_dump(mode="json")) != row["task_sha256"]:
            raise ValueError("Original task content differs from its qualified bank")
    lines = config.original_traces.read_bytes().splitlines(keepends=True)
    originals = [json.loads(line) for line in lines]
    if Counter(item["task_id"] for item in originals) != Counter({task_id: ATTEMPTS_PER_TASK for task_id in tasks}):
        raise ValueError("Original calibration must contain eight logical attempts per task")
    original_rewards: dict[str, list[float]] = {}
    grades = {}
    for index, original in enumerate(originals):
        if original["grade"]["status"] == "graded":
            original_rewards.setdefault(original["task_id"], []).append(graded_reward(original))
            grades[index] = (original["grade"], config.original_traces)
    if (
        original_rewards != summary["task_rewards"]
        or summary["count"] != len(tasks)
        or summary["samples_per_task"] != ATTEMPTS_PER_TASK
    ):
        raise ValueError("Original summary does not match its trace records")
    original_grade_count = len(grades)
    review = config.review.read_json()
    reviewed = {item["replay_sha256"]: item for item in review["replays"]}
    if len(reviewed) != len(review["replays"]) or set(reviewed) != {item.sha256 for item in config.replays}:
        raise ValueError("Replay files do not match the independent review")
    for evidence in config.replays:
        replay = evidence.read_json()
        decision = reviewed[evidence.sha256]
        index = replay["original_line_index"]
        if index in grades or index != decision["original_line_index"]:
            raise ValueError("Recovery duplicates an attempt or changes its reviewed index")
        if (
            replay["original_line_sha256"] != hashlib.sha256(lines[index]).hexdigest()
            or replay["original_line_sha256"] != decision["original_line_sha256"]
            or replay["task_id"] != originals[index]["task_id"]
            or replay["task_sha256"] != hashlib.sha256(tasks[replay["task_id"]].model_dump_json().encode()).hexdigest()
            or replay["source_commit"] != config.runtime_source
            or replay["original_runtime_source"] != config.runtime_source
            or replay["runtime_bundle"] != asdict(config.runtime_bundle)
        ):
            raise ValueError("Replay source, task, or runtime identity changed")
        grades[index] = (replay_grade(originals[index], replay, decision), evidence)

    issuance = config.replacement_issuance.read_json()
    with TemporaryDirectory() as directory:
        task_path = Path(directory) / "replacement.parquet"
        task_path.write_bytes(config.replacement_tasks.read_bytes())
        replacement_tasks = list(read_tasks(str(task_path)))
    replacement_summary = config.replacement_summary.read_json()
    replacements = [json.loads(line) for line in config.replacement_traces.read_bytes().splitlines()]
    index = issuance["original_line_index"]
    original = originals[index]
    evaluation = issuance["evaluation"]
    if index in grades or index != review["startup_replacement"]["original_line_index"]:
        raise ValueError("Startup replacement duplicates an attempt or changes its reviewed index")
    if (
        issuance["original_line_sha256"] != hashlib.sha256(lines[index]).hexdigest()
        or issuance["original_traces_sha256"] != config.original_traces.sha256
        or issuance["original_task_sha256"] != canonical_sha256(tasks[original["task_id"]].model_dump(mode="json"))
        or issuance["tasks_sha256"] != config.replacement_tasks.sha256
        or len(replacement_tasks) != 1
        or replacement_tasks[0] != tasks[original["task_id"]]
        or original["interrupted_operation"] != "start"
        or original["steps"]
        or original["response_token_ids"]
        or len(replacements) != 1
        or replacements[0]["task_id"] != original["task_id"]
        or replacements[0]["interrupted_operation"] is not None
        or replacements[0]["execution_error"] is not None
        or evaluation["model_identity"] != config.model_identity
        or evaluation["model_uri"] != config.model_uri
        or evaluation["tokenizer"] != config.tokenizer
        or evaluation["tokenizer_revision"] != config.tokenizer_revision
        or evaluation["runtime_bundle"] != asdict(config.runtime_bundle)
        or evaluation["temperature"] != CALIBRATION_TEMPERATURE
        or evaluation["samples_per_task"] != 1
        or evaluation["limit"] != 1
        or issuance["runtime_module_hashes"] != config.runtime_module_hashes
        or replacement_summary["model_identity"] != config.model_identity
        or replacement_summary["tasks_identity"] != evaluation["tasks_identity"]
    ):
        raise ValueError("Startup replacement changed its original attempt or sampling protocol")
    replacement_reward = graded_reward(replacements[0])
    if replacement_summary["task_rewards"] != {original["task_id"]: [replacement_reward]}:
        raise ValueError("Startup summary differs from its single trace")
    grades[index] = (replacements[0]["grade"], config.replacement_traces)
    if len(grades) != len(originals):
        raise ValueError("Calibration recovery still has ungraded attempts")
    rewards: dict[str, list[float]] = {task_id: [] for task_id in tasks}
    lineage = []
    for index, original in enumerate(originals):
        grade, evidence = grades[index]
        rewards[original["task_id"]].append(grade["reward"])
        lineage.append(
            {
                "original_line_index": index,
                "original_line_sha256": hashlib.sha256(lines[index]).hexdigest(),
                "task_id": original["task_id"],
                "original_grade": original["grade"],
                "completed_grade": grade,
                "grade_evidence": asdict(evidence),
            }
        )
    passed = sum(reward > 0 for group in rewards.values() for reward in group)
    result = {
        **summary,
        "task_rewards": rewards,
        "categories": {"passed": passed, "incorrect": len(originals) - passed},
        "informative_groups": sum(len(set(group)) > 1 for group in rewards.values()),
        "failed_task_ids": sorted(task_id for task_id, group in rewards.items() if any(reward == 0 for reward in group)),
    }
    destination = StoragePath(config.output_path)
    write_once(destination / "failure_summary.json", result)
    write_once(
        destination / "recovery.json",
        {"inputs": asdict(config), "original_grade_count": original_grade_count, "attempts": lineage},
    )


def calibration_recovery_step(
    value: dict,
    version: str,
    bank: ArtifactStep[Artifact],
    model: ArtifactStep[LevanterCheckpoint],
    runtime_bundle: RuntimeBundle,
    tokenizer: str,
    tokenizer_revision: str,
) -> ArtifactStep[Artifact]:
    """Bind the reviewed recovery files to a separate CPU artifact."""
    fields = {
        name: PinnedFile(**value[name])
        for name in (
            "original_summary",
            "original_traces",
            "review",
            "replacement_issuance",
            "replacement_tasks",
            "replacement_summary",
            "replacement_traces",
        )
    }
    return ArtifactStep(
        name="evals/russell-rsi-initial-calibration-recovery",
        version=version,
        artifact_type=Artifact,
        deps=(bank, model),
        build_config=lambda ctx: CalibrationRecoveryConfig(
            **fields,
            original_tasks=PinnedFile(
                prefix_join(ctx.artifact_path(bank), "train.parquet"), value["original_tasks"]["sha256"]
            ),
            original_bank=PinnedFile(
                prefix_join(ctx.artifact_path(bank), "bank.json"), value["original_bank"]["sha256"]
            ),
            replays=tuple(PinnedFile(**item) for item in value["replays"]),
            model_identity=artifact_identity(model),
            model_uri=ctx.artifact_path(model),
            tokenizer=tokenizer,
            tokenizer_revision=tokenizer_revision,
            bank_identity=artifact_identity(bank),
            runtime_source=value["runtime_source"],
            runtime_bundle=runtime_bundle,
            runtime_module_hashes=value["runtime_module_hashes"],
            output_path=ctx.output_path,
        ),
        run=remote(
            recover_calibration,
            resources=ResourceConfig.with_cpu(cpu=4, ram="16GB", disk="64GB"),
            pip_packages=["./lib/taskcompendium"],
        ),
    )
