from __future__ import annotations

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass

import ray
from omegaconf import DictConfig

from skyrl_train.batch_assembly import BatchPlan
from skyrl_train.batch_metrics import reward_metrics, rollout_metrics, rollout_staleness_metrics
from skyrl_train.rollouts.buffer import RowFacts
from skyrl_train.rollouts.context import RolloutBatchMetadata, TrainingContext
from skyrl_train.rollouts.payloads import RolloutReader
from skyrl_train.telemetry import ConsumedWork, record_consumed_staleness
from skyrl_train.training_batch import TrainingOutputBatch
from skyrl_train.trajectory_runners.types import BatchFields
from skyrl_train.utils.utils import moe_router_replay_requested
from skyrl_train.workers.worker import PPORayActorGroup


def require_worker_batch_rows(
    cfg: DictConfig,
    facts: Sequence[RowFacts | None],
    *,
    group_uids: Sequence[str],
    dp_size: int,
    num_experts: int | None = None,
) -> None:
    """Reject unsupported admitted rows with their rollout group identity."""
    for uid, group in zip(group_uids, facts, strict=True):
        if group is None:
            raise ValueError(f"trainer.batch_builder=worker group {uid!r}: checkpoint metadata without row facts")
        restrictions = [
            ("nonzero loop credit", group.loop_credit),
            (
                "teacher evidence",
                bool(
                    group.fields.present
                    & {"teacher_evidence", "distillation", "student_topk_indices", "behavior_topk_logprobs"}
                ),
            ),
            (
                "route dtype without the model expert count",
                moe_router_replay_requested(cfg)
                and num_experts is None
                and "rollout_routed_experts" in group.fields.present,
            ),
        ]
        for reason, unsupported in restrictions:
            if unsupported:
                raise ValueError(f"trainer.batch_builder=worker group {uid!r} does not support {reason}")
    if sum(len(group.response_len) for group in facts) % dp_size:
        raise ValueError(f"trainer.batch_builder=worker groups {tuple(group_uids)!r}: rows requiring DP padding")


@dataclass(frozen=True)
class BatchSummary:
    work: ConsumedWork
    stop_metrics: dict[str, float]
    callback_rewards: tuple[float, ...]
    batch_size: int


class WorkerBatchStep:
    """Share compact admitted groups and retain assembled rows on policy actors."""

    def __init__(
        self,
        *,
        context: TrainingContext,
        policy: PPORayActorGroup,
        cfg: DictConfig,
        pad_token_id: int,
        num_experts: int | None,
    ):
        self.context, self.policy, self.cfg = context, policy, cfg
        self.pad_token_id, self.num_experts = pad_token_id, num_experts
        self._metadata: RolloutBatchMetadata
        self._reader: RolloutReader
        self._plan: BatchPlan | None = None

    @property
    def admitted_plan(self) -> BatchPlan:
        if self._plan is None:
            raise RuntimeError("worker batch is not prepared")
        return self._plan

    async def admit(self, *, stall_timeout: float) -> dict[str, float]:
        self._metadata = await self.context.next_batch_metadata(stall_timeout=stall_timeout)
        self._reader = self.context.reader(self._metadata, stall_timeout=stall_timeout)
        return self._metadata.metrics

    async def prepare(self, *, global_step: int) -> tuple[dict[str, float], float]:
        groups = self._metadata.groups
        facts = tuple(group.row_facts for group in groups)
        require_worker_batch_rows(
            self.cfg,
            facts,
            group_uids=tuple(group.uid for group in groups),
            dp_size=self.policy.actor_infos[0].rank.dp_size,
            num_experts=self.num_experts,
        )
        fields = BatchFields.from_groups([group.fields for group in facts])
        if not moe_router_replay_requested(self.cfg):
            fields = fields.without_routes()
        staleness = [global_step - group.policy_step for group in groups]
        self._plan = BatchPlan(
            global_step=global_step,
            uids=tuple(group.uid for group in groups for _ in group.row_facts.response_len),
            facts=facts,
            fields=fields,
            rollout_staleness=tuple(
                age for group, age in zip(groups, staleness, strict=True) for _ in group.row_facts.response_len
            ),
            num_experts=self.num_experts,
        )
        metrics = rollout_metrics(
            facts, tis_lcs_alert_threshold=float(self.cfg.trainer.algorithm.tis_lcs_alert_threshold)
        )
        metrics.update(
            reward_metrics(facts, self._plan.uids, n_samples_per_prompt=self.cfg.generator.n_samples_per_prompt)
        )
        metrics.update(
            rollout_staleness_metrics(
                staleness, global_step=global_step, max_staleness_steps=self.context.config.max_staleness_steps
            )
        )
        if self.cfg.trainer.training_metrics:
            record_consumed_staleness(
                self._plan.uids,
                self._plan.rollout_staleness,
                [int(value) for group in facts for value in group.response_len],
                step=global_step,
            )
        reader_ref = ray.put(self._reader)
        references = list(
            await self.context.wait(asyncio.gather(*self.policy.share_batch_groups(self._plan, reader_ref)))
        )
        loaded = await self.context.wait(
            asyncio.gather(*self.policy.load_batch(self._plan, reader_ref, references, self.pad_token_id))
        )
        return metrics, max(loaded)

    async def _run_loaded(self, method: str) -> list[TrainingOutputBatch]:
        return await self.context.wait(
            asyncio.gather(*self.policy.async_run_ray_method("pass_through", method, self.admitted_plan.global_step))
        )

    async def prepare_training(self) -> dict[str, float]:
        outputs = await self._run_loaded("prepare_loaded")
        await self.context.wait(asyncio.gather(*self.policy.async_run_ray_method("pass_through", "empty_cache")))
        return outputs[0].metadata["batch_metrics"]

    async def unload(self) -> None:
        await self.context.wait(
            asyncio.gather(
                *self.policy.async_run_ray_method("pass_through", "unload_batch", self.admitted_plan.global_step)
            )
        )

    def summary(self) -> BatchSummary:
        plan = self.admitted_plan
        response_lengths = tuple(int(value) for group in plan.facts for value in group.response_len)
        fields = plan.fields
        return BatchSummary(
            ConsumedWork(
                len(plan.uids),
                sum(response_lengths),
                sum(int(value) for group in plan.facts for value in group.loss_tokens),
            ),
            plan.stop_metrics,
            tuple(float(value) for group in plan.facts for value in group.scalar_values)
            if not fields.token_rewards
            else (),
            len(plan.uids),
        )
