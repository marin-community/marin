"""Global batch layouts and shared trajectory-to-tensor assembly."""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch
from omegaconf import DictConfig

from skyrl_train.config.objective_spec import rollout_logprobs_required
from skyrl_train.dataset.preprocess import collate_response_token_channel, convert_prompts_responses_to_batch_tensors
from skyrl_train.dataset.routed_expert_batch import RoutedExpertRows
from skyrl_train.distillation import validate_distillation_attachment
from skyrl_train.rollouts.buffer import RowFacts
from skyrl_train.training_batch import TrainingInputBatch
from skyrl_train.trajectory_runners.types import BatchFields, TrajectoryBatch
from skyrl_train.batch_metrics import consumed_stop_metrics
from skyrl_train.inference_engines.base import UNSTAMPED_POLICY_STEP
from skyrl_train.utils.algorithm_registry import PolicyLossRegistry


@dataclass(frozen=True)
class BatchPlan:
    """Ordered row identity, global widths, and channel policy for one batch."""

    batch_id: int
    global_step: int
    uids: tuple[str, ...]
    facts: tuple[RowFacts, ...]
    fields: BatchFields
    rollout_staleness: tuple[int, ...] | None
    num_experts: int | None

    @property
    def max_prompt_len(self) -> int:
        return max(int(facts.prompt_len.max()) for facts in self.facts)

    @property
    def max_response_len(self) -> int:
        return max(int(facts.response_len.max()) for facts in self.facts)

    def rank_rows(self, dp: int, dp_size: int) -> range:
        size = len(self.uids) // dp_size
        return range(dp * size, (dp + 1) * size)

    def group_indices(self, dp: int, dp_size: int) -> tuple[int, ...]:
        """Return the ordered groups intersecting the rank's contiguous rows."""
        rows = self.rank_rows(dp, dp_size)
        offset = 0
        indices = []
        for index, facts in enumerate(self.facts):
            end = offset + len(facts.response_len)
            if offset < rows.stop and rows.start < end:
                indices.append(index)
            offset = end
        return tuple(indices)


def fields_from_facts(facts: Sequence[RowFacts]) -> BatchFields:
    """Resolve normalization policy from admitted groups in row order."""
    return BatchFields.from_groups([group.fields for group in facts])


def _validated_distillation_tensors(
    trajectory_batch: TrajectoryBatch,
    response_mask: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Validate optional teacher evidence and return its learner-only tensors."""
    evidence = trajectory_batch.get("teacher_evidence")
    distillation = trajectory_batch.get("distillation")
    if evidence is None and distillation is None:
        return {}
    if evidence is None or distillation is None:
        raise ValueError("trajectory batches must carry teacher_evidence and prepared distillation together")
    trajectory_ids = trajectory_batch.get("trajectory_ids")
    if trajectory_ids is None:
        raise ValueError("teacher evidence requires stable trajectory_ids")
    validate_distillation_attachment(
        evidence,
        distillation,
        trajectory_ids=tuple(trajectory_id.to_string() for trajectory_id in trajectory_ids),
        response_mask=response_mask.to(torch.bool),
    )
    return distillation.training_tensors()


def token_staleness(
    lease_staleness: Sequence[int], policy_steps: torch.Tensor | None, response_mask: torch.Tensor, global_step: int
) -> torch.Tensor:
    """Each response token's staleness: its lease's, unless the engine recorded the policy step that sampled it."""
    lease = torch.tensor(lease_staleness, dtype=torch.int32).unsqueeze(1)
    staleness = lease * response_mask.to(torch.int32)
    if policy_steps is None:
        return staleness
    stamped = (policy_steps != UNSTAMPED_POLICY_STEP) & (response_mask > 0)
    staleness = torch.where(stamped, global_step - policy_steps, staleness)
    assert bool(((staleness >= 0) & (staleness <= lease)).all()), (
        "a token's policy step must fall between its lease's step and the training step"
    )
    return staleness


def assemble_slice(
    plan: BatchPlan,
    rows: range,
    trajectory_batch: TrajectoryBatch,
    *,
    tokenizer,
    algorithm: DictConfig,
) -> TrainingInputBatch:
    """Build normalized rows at the plan's global widths and retain whole-batch metadata."""
    batch = trajectory_batch
    if rows.step != 1 or rows.start < 0 or rows.stop > len(plan.uids) or len(rows) != len(batch["response_ids"]):
        raise ValueError("normalized rows must match their contiguous batch-plan range")
    token_channel = bool(algorithm.get("enable_token_reward_channel", False))
    sequences, attention, response, rewards, losses, logprobs, shaping, tags = (
        convert_prompts_responses_to_batch_tensors(
            tokenizer,
            batch["prompt_token_ids"],
            batch["response_ids"],
            batch["rewards"],
            batch["loss_masks"],
            batch.get("rollout_logprobs"),
            batch.get("token_level_shaping") if token_channel else None,
            batch.get("response_span_tags") if token_channel else None,
            max_prompt_len=plan.max_prompt_len,
            max_response_len=plan.max_response_len,
        )
    )
    if (
        rollout_logprobs_required(algorithm, loss_spec=PolicyLossRegistry.spec(algorithm.policy_loss_type))
        and logprobs is None
    ):
        raise ValueError("rollout_logprobs are required by the configured objective")
    if logprobs is not None:
        assert logprobs.shape == losses.shape, "Logprobs should look like responses"
    route_rows = None
    if "rollout_routed_experts" in plan.fields.present:
        route_rows = RoutedExpertRows(tuple(batch["rollout_routed_experts"]), plan.max_response_len, plan.num_experts)
        if len(route_rows) != len(batch["response_ids"]):
            raise ValueError("routed experts must have one row per response")
    if plan.rollout_staleness is None:
        staleness = torch.zeros_like(response, dtype=torch.int32)
    else:
        policy_steps = collate_response_token_channel(
            batch.get("rollout_policy_steps"),
            response,
            dtype=torch.int32,
            expected_lengths=[len(row) for row in batch["response_ids"]],
        )
        staleness = token_staleness(
            plan.rollout_staleness[rows.start : rows.stop], policy_steps, response, plan.global_step
        )
    training_input = TrainingInputBatch(
        {
            "sequences": sequences,
            "attention_mask": attention,
            "response_mask": response,
            "rewards": rewards,
            "loss_mask": losses,
            "rollout_logprobs": logprobs,
            "rollout_staleness": staleness,
            "is_last_step": (
                torch.tensor(batch["is_last_step"], dtype=torch.bool) if batch.get("is_last_step") is not None else None
            ),
        },
        routed_expert_rows=route_rows,
    )
    training_input.update(_validated_distillation_tensors(batch, response))
    pair_roles = batch.get("pair_roles")
    if pair_roles is not None:
        if len(pair_roles) != len(batch["response_ids"]):
            raise ValueError("pair_roles must carry one role per response row")
        training_input["pair_roles"] = torch.tensor(pair_roles, dtype=torch.float32)
    if shaping is not None:
        training_input["token_level_shaping"] = shaping
    if tags is not None:
        training_input["response_span_tags"] = tags
    loop_advantages = collate_response_token_channel(
        batch.get("loop_advantages"),
        response,
        dtype=torch.float,
        expected_lengths=[len(row) for row in batch["response_ids"]],
    )
    if loop_advantages is not None:
        training_input["loop_advantages"] = loop_advantages
    response_lengths = np.concatenate([facts.response_len for facts in plan.facts])
    training_input.metadata = {
        "uids": list(plan.uids),
        "response_length": plan.max_response_len,
        "avg_response_length": int(response_lengths.sum()) / len(plan.uids),
        "consumed_stop_metrics": consumed_stop_metrics(
            tuple(reason for facts in plan.facts for reason in facts.stop_reasons)
            if "stop_reasons" in plan.fields.first_group_list_fields
            else None,
            len(plan.uids),
        ),
    }
    if "exclude_from_baseline" in plan.fields.present:
        training_input.metadata["exclude_from_baseline"] = np.concatenate(
            [facts.exclude_from_baseline for facts in plan.facts]
        )
    return training_input


def forward_input(data: TrainingInputBatch) -> TrainingInputBatch:
    """Select forward tensors and available control metadata, retaining compact routes."""
    keys = ["sequences", "attention_mask"]
    if data.routed_experts is not None:
        keys.append("rollout_routed_experts")
    return data.select(
        keys=keys, metadata_keys=[key for key in ("response_length", "global_step") if key in data.metadata]
    )
