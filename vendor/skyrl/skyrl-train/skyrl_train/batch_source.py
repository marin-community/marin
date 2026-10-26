from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from threading import Lock
from typing import Protocol

import ray
from omegaconf import DictConfig

from marinskyrl.runtime_options import (
    AdvantageEstimator,
    BatchBuilder,
    PolicyLossType,
    reference_model_required,
)
from skyrl_train.batch_assembly import BatchPlan, plan_batch
from skyrl_train.batch_metrics import (
    reward_metrics,
    rollout_metrics,
    rollout_staleness_metrics,
)
from skyrl_train.distributed.dispatch import MeshRank
from skyrl_train.rollouts.buffer import RowFacts
from skyrl_train.rollouts.context import RolloutBatchMetadata, RolloutReader, TrainingContext
from skyrl_train.telemetry import ConsumedWork, critical_phase, record_consumed_staleness
from skyrl_train.timing_observability import StepWallTime
from skyrl_train.training_batch import TrainingOutputBatch
from skyrl_train.trajectory_runners.trajectory_reward_shaping import parse_trajectory_reward_shaping_config
from skyrl_train.trajectory_runners.types import BatchFields
from skyrl_train.utils import Timer
from skyrl_train.utils.advantage_estimators import OUTCOME_ADVANTAGE_ESTIMATORS
from skyrl_train.utils.utils import moe_router_replay_requested
from skyrl_train.workers.worker import PPORayActorGroup


BATCH_CLEANUP_TIMEOUT_SECONDS = 30.0


def require_worker_batch_support(
    cfg: DictConfig,
    *,
    facts: Sequence[RowFacts | None] | None = None,
    policy_mesh: Sequence[MeshRank] | None = None,
    num_experts: int | None = None,
) -> None:
    """Enforce every worker-specific objective, row, and mesh restriction."""
    mode = cfg.trainer.batch_builder
    restrictions = []
    if facts is None:
        algorithm = cfg.trainer.algorithm
        shaping = parse_trajectory_reward_shaping_config(cfg.generator.get("trajectory_reward_shaping"))
        megatron = cfg.trainer.policy.megatron_config
        supported_loss = (
            algorithm.policy_loss_type
            in (
                PolicyLossType.REGULAR,
                PolicyLossType.DUAL_CLIP,
                PolicyLossType.GSPO,
                PolicyLossType.CISPO,
                PolicyLossType.SAPO,
                PolicyLossType.IMPORTANCE_SAMPLING,
                PolicyLossType.BEHAVIOR_CLIP,
            )
            and algorithm.advantage_estimator in OUTCOME_ADVANTAGE_ESTIMATORS
        )
        supported_loss |= (
            algorithm.policy_loss_type == PolicyLossType.SFT
            and algorithm.advantage_estimator == AdvantageEstimator.UNIFORM
        )
        restrictions = [
            (
                "DPO and reference-model objectives",
                reference_model_required(algorithm),
            ),
            ("the configured loss and estimator", not supported_loss),
            ("colocated placement", cfg.trainer.placement.colocate_all),
            ("distillation", algorithm.get("distillation") is not None),
            ("trajectory selection", cfg.trainer.trajectory_selector.type is not None),
            ("step-wise training", cfg.trainer.step_wise_training),
            ("advantage batch normalization", algorithm.advantage_batch_normalize),
            ("loop advantage credit", shaping.enabled and shaping.loop.advantage_penalty_per_token > 0),
            (
                "ETP greater than one",
                (megatron.get("expert_tensor_parallel_size") or megatron.tensor_model_parallel_size) > 1,
            ),
            ("dense batch dumping", mode == BatchBuilder.WORKER and cfg.trainer.dump_data_batch),
        ]
    if facts is not None:
        valid_facts = all(group is not None for group in facts)
        restrictions.append(("checkpoint metadata without row facts", not valid_facts))
        if valid_facts:
            restrictions.extend(
                [
                    ("token-valued outcome credit", any(not group.scalar_rewards for group in facts)),
                    ("nonzero loop credit", any(group.loop_credit for group in facts)),
                    (
                        "teacher evidence",
                        any(
                            group.fields.present
                            & {"teacher_evidence", "distillation", "student_topk_indices", "behavior_topk_logprobs"}
                            for group in facts
                        ),
                    ),
                    (
                        "route dtype without the model expert count",
                        moe_router_replay_requested(cfg)
                        and num_experts is None
                        and any("rollout_routed_experts" in group.fields.present for group in facts),
                    ),
                ]
            )
            if policy_mesh is not None:
                dp_size = policy_mesh[0].dp_size if policy_mesh else 0
                if dp_size < 1 or any(rank.dp_size != dp_size for rank in policy_mesh):
                    raise ValueError(f"trainer.batch_builder={mode} requires a consistent policy mesh")
                owners = [rank.dp for rank in policy_mesh if rank.is_collection_dp_rank()]
                restrictions.extend(
                    [
                        ("rows requiring DP padding", sum(len(group.response_len) for group in facts) % dp_size != 0),
                        ("an incomplete policy collection mesh", sorted(owners) != list(range(dp_size))),
                    ]
                )
    for reason, unsupported in restrictions:
        if unsupported:
            raise ValueError(f"trainer.batch_builder={mode} does not support {reason}")


@dataclass(frozen=True)
class BatchSummary:
    work: ConsumedWork
    stop_metrics: dict[str, float]
    callback_rewards: tuple[float, ...]
    batch_size: int


class BatchSource(Protocol):
    async def admit(self, *, stall_timeout: float) -> dict[str, float]: ...
    async def prepare(self, *, global_step: int, step_wall: StepWallTime | None) -> None: ...
    async def forward(self) -> None: ...
    async def finalize(self, *, step_wall: StepWallTime | None) -> None: ...
    async def train(self, *, step_wall: StepWallTime | None) -> dict: ...
    def summary(self) -> BatchSummary: ...
    async def release(self, error: BaseException | None = None) -> None: ...


class WorkerBatchSource:
    """Share compact admitted groups and retain assembled rows on policy actors."""

    def __init__(
        self,
        *,
        context: TrainingContext,
        policy: PPORayActorGroup,
        cfg: DictConfig,
        pad_token_id: int,
        num_experts: int | None,
        metrics: dict[str, float],
        timings: dict[str, float],
        drain_policy: Callable[[], Awaitable[None]],
        train_policy: Callable[[Callable[[], list[ray.ObjectRef]]], tuple[dict, list[ray.ObjectRef]]],
        finish_update: Callable[[dict], None],
        log_batch: Callable[..., None],
    ):
        self.context, self.policy, self.cfg = context, policy, cfg
        self.pad_token_id, self.num_experts = pad_token_id, num_experts
        self.metrics, self.timings = metrics, timings
        self.drain_policy, self.train_policy = drain_policy, train_policy
        self.finish_update, self.log_batch = finish_update, log_batch
        self._metadata: RolloutBatchMetadata | None = None
        self._reader: RolloutReader | None = None
        self._plan: BatchPlan | None = None
        self._pending = []
        self._dispatch_lock = Lock()
        self._released = False

    @property
    def admitted_reader(self) -> RolloutReader:
        if self._reader is None:
            raise RuntimeError("worker batch is not admitted")
        return self._reader

    @property
    def admitted_plan(self) -> BatchPlan:
        if self._plan is None:
            raise RuntimeError("worker batch is not prepared")
        return self._plan

    async def admit(self, *, stall_timeout: float) -> dict[str, float]:
        with Timer("wait_for_generation_buffer", self.timings) as timer:
            self._metadata = await self.context.next_batch_metadata(stall_timeout=stall_timeout)
        self.wait_duration = timer.duration
        self._reader = self.context.reader(self._metadata, stall_timeout=stall_timeout)
        return self._metadata.metrics

    async def prepare(self, *, global_step: int, step_wall: StepWallTime | None) -> None:
        if step_wall is not None:
            step_wall.start("batch_assembly")
        groups = self._metadata.groups
        facts = tuple(group.row_facts for group in groups)
        require_worker_batch_support(
            self.cfg,
            facts=facts,
            policy_mesh=tuple(actor.rank for actor in self.policy.actor_infos),
            num_experts=self.num_experts,
        )
        fields = BatchFields.from_groups([group.fields for group in facts])
        if not moe_router_replay_requested(self.cfg):
            fields = fields.without_routes()
        staleness = [global_step - group.policy_step for group in groups]
        self._plan = plan_batch(
            batch_id=self._metadata.batch_id,
            global_step=global_step,
            uids=tuple(group.uid for group in groups for _ in group.row_facts.response_len),
            facts=facts,
            fields=fields,
            rollout_staleness=tuple(
                age for group, age in zip(groups, staleness, strict=True) for _ in group.row_facts.response_len
            ),
            num_experts=self.num_experts,
            algorithm=self.cfg.trainer.algorithm,
        )
        staleness_metrics = rollout_staleness_metrics(
            staleness, global_step=global_step, max_staleness_steps=self.context.config.max_staleness_steps
        )
        self.metrics.update(
            rollout_metrics(facts, tis_lcs_alert_threshold=float(self.cfg.trainer.algorithm.tis_lcs_alert_threshold))
        )
        self.metrics.update(
            reward_metrics(facts, self._plan.uids, n_samples_per_prompt=self.cfg.generator.n_samples_per_prompt)
        )
        self.metrics.update(staleness_metrics)
        if self.cfg.trainer.training_metrics:
            record_consumed_staleness(
                self._plan.uids,
                self._plan.rollout_staleness,
                [int(value) for group in facts for value in group.response_len],
                step=global_step,
            )
        reader_ref = ray.put(self._reader)
        self._pending = self.policy.share_batch_groups(self._plan, reader_ref)
        references = list(await self.context.wait(asyncio.gather(*self._pending)))
        self._pending = self.policy.load_batch(self._plan, reader_ref, references, self.pad_token_id)
        loaded = await self.context.wait(asyncio.gather(*self._pending))
        self._pending = []
        self.timings["load_worker_batch"] = max(loaded)
        self.log_batch(
            groups=len(facts),
            sequences=len(self._plan.uids),
            response_tokens=sum(int(group.response_len.sum()) for group in facts),
            duration_seconds=self.wait_duration,
        )

    async def _run_loaded(self, method: str) -> list[TrainingOutputBatch]:
        self._pending = self.policy.async_run_ray_method("pass_through", method, self.admitted_plan.batch_id)
        outputs = await self.context.wait(asyncio.gather(*self._pending))
        self._pending = []
        return outputs

    async def forward(self) -> None:
        await self.drain_policy()
        with Timer("fwd_logprobs_values_reward", self.timings):
            outputs = await self._run_loaded("forward_loaded")
            self.metrics.update(outputs[0].metadata["batch_metrics"])
            await self.context.wait(asyncio.gather(*self.policy.async_run_ray_method("pass_through", "empty_cache")))

    async def finalize(self, *, step_wall: StepWallTime | None) -> None:
        if step_wall is not None:
            step_wall.start("advantages")
        with Timer("prepare_worker_training_input", self.timings):
            outputs = await self._run_loaded("prepare_loaded")
            self.metrics.update(outputs[0].metadata["batch_metrics"])

    async def train(self, *, step_wall: StepWallTime | None) -> dict:
        if step_wall is not None:
            step_wall.start("policy_training")
        with (
            Timer("train_critic_and_policy", self.timings),
            critical_phase("train_step", self.admitted_plan.global_step),
        ):
            status, cache_refs = await asyncio.to_thread(self.train_policy, self._dispatch_train)
            self._pending = []
            await self.context.wait(asyncio.gather(*cache_refs))
        self.finish_update(status)
        return status

    def _dispatch_train(self) -> list[ray.ObjectRef]:
        with self._dispatch_lock:
            if self._released:
                raise RuntimeError("worker batch is released")
            self._pending = self.policy.async_run_ray_method(
                "pass_through", "train_loaded", self.admitted_plan.batch_id
            )
            return self._pending

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

    async def release(self, error: BaseException | None = None) -> None:
        with self._dispatch_lock:
            self._released = True
        try:
            async with asyncio.timeout(BATCH_CLEANUP_TIMEOUT_SECONDS):
                if error is not None:
                    for ref in self._pending:
                        ray.cancel(ref, force=False)
                if self._pending:
                    await asyncio.gather(*self._pending, return_exceptions=error is not None)
                if self._plan is not None:
                    await asyncio.gather(
                        *self.policy.async_run_ray_method("pass_through", "unload_batch", self._plan.batch_id)
                    )
        except Exception as cleanup_error:
            if error is None:
                raise
            error.add_note(f"Loaded-batch cleanup failed: {cleanup_error}")
        finally:
            self._pending = []
            self._plan, self._reader, self._metadata = None, None, None


@asynccontextmanager
async def batch_lifetime(source: BatchSource):
    """Release a source's owned batch while retaining the initiating exception."""
    error = None
    try:
        yield source
    except BaseException as caught:
        error = caught
        raise
    finally:
        await source.release(error)
