# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Read exact Harbor evidence for verifier-selected BFCL recovery preferences."""

import gzip
import json
import zipfile
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypedDict

from rigging.filesystem.storage_path import StoragePath

from experiments.post_training.bfcl_rl.data import DATASET_COMMIT, BFCLPartition
from experiments.post_training.bfcl_rl.preferences import RolloutOutcome, VerifiedRollout, select_pair

SCOREABLE_HARBOR_MODEL_ERRORS = frozenset(
    {"AgentTimeoutError", "ContextLengthExceededError", "NonZeroAgentExitCodeError"}
)


def canonical_native_outcome(trial: Mapping[str, Any]) -> RolloutOutcome:
    """Use verified rewards for policy-scoreable model errors; never invent a timeout score."""
    error = trial["exception_info"]
    if error is not None and error["exception_type"] not in SCOREABLE_HARBOR_MODEL_ERRORS:
        return RolloutOutcome.UNSCORED
    rewards = (trial["verifier_result"] or {}).get("rewards")
    if rewards == {"reward": 1.0}:
        return RolloutOutcome.CORRECT
    if rewards == {"reward": 0.0}:
        return RolloutOutcome.INCORRECT
    return RolloutOutcome.UNSCORED


@dataclass(frozen=True)
class CollectionIdentity:
    """Identities taken from the immutable collection launch and audited data dependency."""

    run_id: str
    model_source_identity: str
    model_revision: str
    harness: str
    dataset_commit: str
    task_root: str


@dataclass(frozen=True)
class TokenStep:
    prompt_token_ids: tuple[int, ...]
    response_token_ids: tuple[int, ...]
    loss_mask: tuple[int, ...]


@dataclass(frozen=True)
class RetainedRollout:
    rollout: VerifiedRollout
    record_id: str
    steps: tuple[TokenStep, ...]


class PretokenizedPreference(TypedDict):
    chosen_input_ids: list[int]
    chosen_assistant_masks: list[int]
    rejected_input_ids: list[int]
    rejected_assistant_masks: list[int]


def retained_rollout(
    record: Mapping[str, Any], *, identity: CollectionIdentity, partition: BFCLPartition, trajectory_uri: str
) -> RetainedRollout:
    """Validate a v6 record against its pinned launch and audited complement dependency.

    The caller must establish that the launch consumed the audited data artifact;
    retained records identify task names, but do not carry hashes of task files.
    """
    if identity.dataset_commit != DATASET_COMMIT or partition.dataset_commit != identity.dataset_commit:
        raise ValueError("collection must use the audited BFCL dataset revision")
    if record["schema_version"] != 6:
        raise ValueError("preference ingestion requires per-step prompt evidence from schema v6")
    if record["run_id"] != identity.run_id:
        raise ValueError("retained record belongs to a different collection run")
    if record["phase"] != "eval" or record["global_step"] != 0:
        raise ValueError("recovery requires generation-only evidence before optimizer updates")
    if record["provenance"]["model_source_identity"] != identity.model_source_identity:
        raise ValueError("retained record belongs to a different model source")
    tasks = {task.name: task for task in partition.complement}
    task_name = record["trajectory"]["instance_id"]
    if task_name not in tasks:
        raise ValueError(f"retained task is outside the BFCL training complement: {task_name}")
    task = tasks[task_name]
    if task.source_id in {item.source_id for item in partition.parity}:
        raise ValueError("audited partition contains a parity task in the training complement")
    data_source = record["trajectory"]["environment_extras"]["data_source"]
    if Path(data_source).name != task.name or str(Path(data_source).parent) != identity.task_root:
        raise ValueError("retained task path differs from the worker's audited data source")
    verdict = record["verification_result"]
    outcome = RolloutOutcome.UNSCORED
    disposition = record["disposition"]
    if verdict is not None and verdict["status"] == "verified":
        score = verdict["score"]
        if score not in (0.0, 1.0) or verdict["score_min"] != 0.0 or verdict["score_max"] != 1.0:
            raise ValueError("BFCL preferences require a binary verifier score")
        if verdict["passed"] is not None and verdict["passed"] != (score == 1.0):
            raise ValueError("BFCL verifier pass flag contradicts its score")
        model_error = disposition["exception_type"] in SCOREABLE_HARBOR_MODEL_ERRORS
        scoreable = disposition["error_treatment"] is None or (model_error and disposition["error_treatment"] == "zero")
        if disposition["server_error"] is None and (disposition["exception_type"] is None or model_error) and scoreable:
            if disposition["error_treatment"] is None and record["reward"]["outcome"] != score:
                raise ValueError("retained outcome differs from the BFCL verifier score")
            if disposition["error_treatment"] == "zero" and record["reward"]["outcome"] != 0.0:
                raise ValueError("zero-treated model error has a nonzero retained reward")
            outcome = RolloutOutcome.CORRECT if score == 1.0 else RolloutOutcome.INCORRECT

    response = record["response"]
    tokens = response["token_ids"]
    masks = response["loss_mask"]
    if len(tokens) != len(masks) or any(type(token) is not int or token < 0 for token in tokens):
        raise ValueError("retained response tokens and loss masks must align")
    if any(mask not in (0, 1) for mask in masks):
        raise ValueError("retained response loss masks must be binary")
    steps = []
    token_start = 0
    for boundary in response["step_boundaries"]:
        token_end = boundary["token_end"]
        if boundary["token_start"] != token_start or not token_start <= token_end <= len(tokens):
            raise ValueError("retained step boundaries must partition the response")
        prompt = boundary["prompt_token_ids"]
        if not prompt or any(type(token) is not int or token < 0 for token in prompt):
            raise ValueError("retained step requires exact prompt token IDs")
        steps.append(TokenStep(tuple(prompt), tuple(tokens[token_start:token_end]), tuple(masks[token_start:token_end])))
        token_start = token_end
    if token_start != len(tokens):
        raise ValueError("retained step evidence does not cover the trajectory")
    if steps and tuple(record["prompt"]["token_ids"]) != steps[0].prompt_token_ids:
        raise ValueError("retained step evidence does not cover the trajectory")
    if not steps and outcome is not RolloutOutcome.UNSCORED:
        raise ValueError("verified rollout requires exact model-token evidence")
    rollout = VerifiedRollout(
        task.source_id,
        task.digest,
        identity.harness,
        record["trajectory"]["repetition_id"],
        identity.model_revision,
        outcome,
        trajectory_uri,
    )
    return RetainedRollout(rollout, record["record_id"], tuple(steps))


def retained_archive_records(archives: Sequence[str]) -> Iterator[tuple[str, dict[str, Any]]]:
    """Read retention records and their archive locations without extracting files."""
    record_ids = set()
    for path in archives:
        with StoragePath(path).open("rb") as source, zipfile.ZipFile(source) as archive:
            for name in sorted(archive.namelist()):
                if not name.startswith("records/") or not name.endswith(".json.gz"):
                    continue
                record = json.loads(gzip.decompress(archive.read(name)))
                if record["record_id"] in record_ids:
                    raise ValueError(f"duplicate retained record: {record['record_id']}")
                record_ids.add(record["record_id"])
                yield f"{path}#{name}", record
    if not record_ids:
        raise ValueError("collection archives contain no retained records")


def read_retained_archives(
    archives: Sequence[str], *, identity: CollectionIdentity, partition: BFCLPartition
) -> tuple[RetainedRollout, ...]:
    """Validate native retention archives without retokenizing."""
    return tuple(
        retained_rollout(record, identity=identity, partition=partition, trajectory_uri=uri)
        for uri, record in retained_archive_records(archives)
    )


def causal_token_sequence(steps: Sequence[TokenStep], *, max_length: int) -> tuple[list[int], list[int]]:
    """Preserve exact causal prefixes and assistant masks; reject context forks and truncation."""
    tokens: list[int] = []
    masks: list[int] = []
    for step in steps:
        prefix = list(step.prompt_token_ids)
        if prefix[: len(tokens)] != tokens:
            raise ValueError("context fork requires a preference reader supporting independent step prefixes")
        masks.extend([0] * (len(prefix) - len(tokens)))
        tokens = prefix + list(step.response_token_ids)
        masks.extend(step.loss_mask)
    if len(tokens) > max_length:
        raise ValueError("preference sequence exceeds the training context; truncation would change the rollout")
    if not any(masks):
        raise ValueError("retained trajectory has no trainable assistant tokens")
    return tokens, masks


def pretokenized_preference(
    chosen: RetainedRollout, rejected: RetainedRollout, *, max_length: int
) -> PretokenizedPreference:
    """Produce the Levanter preference-cache columns from a sole-correct BFCL pair."""
    chosen_rollout, rejected_rollout = chosen.rollout, rejected.rollout
    select_pair(chosen_rollout, rejected_rollout)
    if chosen_rollout.outcome != RolloutOutcome.CORRECT or rejected_rollout.outcome != RolloutOutcome.INCORRECT:
        raise ValueError("preference requires one verifier-correct and one verifier-incorrect rollout")
    if chosen.steps[0].prompt_token_ids != rejected.steps[0].prompt_token_ids:
        raise ValueError("preference trajectories must share the exact initial prompt")
    chosen_ids, chosen_masks = causal_token_sequence(chosen.steps, max_length=max_length)
    rejected_ids, rejected_masks = causal_token_sequence(rejected.steps, max_length=max_length)
    return {
        "chosen_input_ids": chosen_ids,
        "chosen_assistant_masks": chosen_masks,
        "rejected_input_ids": rejected_ids,
        "rejected_assistant_masks": rejected_masks,
    }
