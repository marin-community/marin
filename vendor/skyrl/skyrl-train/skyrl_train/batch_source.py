from __future__ import annotations

import asyncio
from dataclasses import dataclass, replace
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Protocol

import ray
from marinskyrl.runtime_options import (
    BatchBuilder,
    PolicyLossType,
    AdvantageEstimator,
    parse_batch_builder,
    reference_model_required,
)
from skyrl_train.batch_assembly import BatchPlan, fields_from_facts, plan_batch
from skyrl_train.trajectory_runners.types import BatchFields
from skyrl_train.rollouts.context import RolloutBatchMetadata, RolloutReader
from skyrl_train.utils.advantage_estimators import OUTCOME_ADVANTAGE_ESTIMATORS
from skyrl_train.trajectory_runners.trajectory_reward_shaping import parse_trajectory_reward_shaping_config
from skyrl_train.utils.utils import moe_router_replay_requested
from skyrl_train.distributed.dispatch import collect_actor_results

from marinskyrl.distillation import DistillationRewardMode
from skyrl_train.batch_metrics import (
    consumed_work,
    install_correction,
    consumed_stop_metrics,
    rollout_metrics,
    reward_metrics,
)
from skyrl_train.rollouts.buffer import RolloutGroup
from skyrl_train.rollouts.context import TrainingContext
from skyrl_train.telemetry import ConsumedWork, critical_phase, record_rollout_staleness, record_consumed_staleness
from skyrl_train.timing_observability import StepWallTime
from skyrl_train.training_batch import TrainingInputBatch
from skyrl_train.utils import Timer
from skyrl_train.utils.importance_ratio_diagnostics import LocalReduction

if TYPE_CHECKING:
    from skyrl_train.trainer import RayPPOTrainer


BATCH_CLEANUP_TIMEOUT_SECONDS = 30.0


class UnsupportedWorkerBatchError(ValueError):
    """A requested worker or verify configuration has no equivalent assembly path."""


def require_worker_batch_support(cfg, *, trainer_type, facts=None, policy_mesh=None, num_experts=None) -> None:
    """Enforce every worker-specific objective, row, and mesh restriction."""
    from skyrl_train.trainer import RayPPOTrainer

    algorithm = cfg.trainer.algorithm
    mode = parse_batch_builder(cfg.trainer.batch_builder)
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
        algorithm.policy_loss_type == PolicyLossType.SFT and algorithm.advantage_estimator == AdvantageEstimator.UNIFORM
    )
    restrictions = [
        (
            "DPO and reference-model objectives",
            algorithm.policy_loss_type == PolicyLossType.DPO or reference_model_required(algorithm),
        ),
        ("the configured loss and estimator", not supported_loss),
        ("custom trainer hooks", trainer_type is not RayPPOTrainer),
        ("critic", cfg.trainer.critic.model.path is not None),
        ("distillation", algorithm.get("distillation") is not None),
        ("trajectory selection", cfg.trainer.trajectory_selector.type is not None),
        ("step-wise training", cfg.trainer.step_wise_training),
        ("advantage batch normalization", algorithm.advantage_batch_normalize),
        ("loop advantage credit", shaping.enabled and shaping.loop.advantage_penalty_per_token > 0),
        (
            "ETP greater than one",
            (megatron.get("expert_tensor_parallel_size") or megatron.tensor_model_parallel_size) > 1,
        ),
        ("dense batch dumping", mode is BatchBuilder.WORKER and cfg.trainer.dump_data_batch),
    ]
    if facts is not None:
        valid_facts = all(group is not None and isinstance(group.fields, BatchFields) for group in facts)
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
                    raise UnsupportedWorkerBatchError(f"trainer.batch_builder={mode} requires a consistent policy mesh")
                owners = [rank.dp for rank in policy_mesh if rank.is_collection_dp_rank()]
                restrictions.extend(
                    [
                        ("rows requiring DP padding", sum(len(group.response_len) for group in facts) % dp_size != 0),
                        ("an incomplete policy collection mesh", sorted(owners) != list(range(dp_size))),
                    ]
                )
    for reason, unsupported in restrictions:
        if unsupported:
            raise UnsupportedWorkerBatchError(f"trainer.batch_builder={mode} does not support {reason}")


@dataclass(frozen=True)
class BatchSummary:
    work: ConsumedWork
    group_count: int
    stop_metrics: dict[str, float]
    callback_rewards: tuple[float, ...]


@dataclass
class DriverBatchState:
    batch: TrainingInputBatch
    summary: BatchSummary | None
    metrics: dict[str, float]
    timings: dict[str, float]
    empty: bool


class BatchSource(Protocol):
    async def admit(self, *, stall_timeout: float) -> dict[str, float]: ...
    async def prepare(self, *, global_step: int, step_wall: StepWallTime | None) -> None: ...
    async def forward(self, *, step_wall: StepWallTime | None) -> None: ...
    async def finalize(self, *, step_wall: StepWallTime | None) -> None: ...
    async def train(self, *, step_wall: StepWallTime | None) -> dict: ...
    def summary(self) -> BatchSummary: ...
    async def release(self, error: BaseException | None = None) -> None: ...


class DriverBatchSource:
    """Own an admitted driver batch through preparation, forward, and training."""

    def __init__(self, trainer: RayPPOTrainer, context: TrainingContext):
        self.trainer = trainer
        self.context = context
        self.state: DriverBatchState | None = None
        self.groups: list[RolloutGroup] = []

    @property
    def batch(self) -> TrainingInputBatch:
        if self.state is None:
            raise RuntimeError("driver batch is not prepared")
        return self.state.batch

    async def admit(self, *, stall_timeout: float) -> dict[str, float]:
        self.groups, metrics = await self.context.next_batch(
            stall_timeout=stall_timeout, on_admitted=self.trainer._submit_admitted_groups_for_teacher_scoring
        )
        return metrics

    async def prepare(self, *, global_step: int, step_wall: StepWallTime | None) -> None:
        scored = None
        if self.trainer._distillation_runtime is not None:
            with Timer("wait_for_teacher_evidence", self.trainer.all_timings):
                scored = await self.trainer._await_admitted_teacher_evidence(self.groups)
        if step_wall is not None:
            step_wall.start("batch_assembly")
        batch = await asyncio.to_thread(self.trainer.convert_rollout_groups_to_training_input, self.groups)
        if scored is not None:
            self.trainer._attach_teacher_evidence(batch, scored)
        self.prepare_from_batch(batch, summary=self._summarize(batch))
        responses = [response for group in self.groups for response in group.trajectory_batch["response_ids"]]
        self.trainer._log_rollout_batch_completed(
            groups=len(self.groups),
            sequences=len(responses),
            response_tokens=sum(map(len, responses)),
            duration_seconds=self.trainer.all_timings["wait_for_generation_buffer"],
        )

    def _summarize(self, batch: TrainingInputBatch) -> BatchSummary:
        return BatchSummary(
            consumed_work(batch),
            len(self.groups),
            batch.metadata["consumed_stop_metrics"],
            tuple(self.trainer._current_step_rewards),
        )

    def prepare_from_batch(self, batch: TrainingInputBatch, *, summary: BatchSummary | None = None) -> None:
        """Install a complete state for an admitted batch or a dense replay prefix."""
        self.state = DriverBatchState(
            batch,
            summary,
            self.trainer.all_metrics,
            self.trainer.all_timings,
            bool(
                "ftpo_chosen_mask" in batch
                and not batch["loss_mask"].any()
                and not batch.get("ftpo_weights", batch["loss_mask"]).any()
            ),
        )

    async def forward(self, *, step_wall: StepWallTime | None) -> None:
        if self.state.empty:
            return
        await self.trainer._drain_policy_event_loops()
        with Timer("fwd_logprobs_values_reward", self.state.timings):
            self.state.batch = await asyncio.to_thread(self.trainer.fwd_logprobs_values_reward, self.batch)

    async def finalize(self, *, step_wall: StepWallTime | None) -> None:
        if self.state.empty:
            return
        training_input = self.batch
        if (
            self.trainer.distillation_plan is not None
            and self.trainer.distillation_plan.reward_mode is DistillationRewardMode.REPLACE
        ):
            if training_input.get("teacher_valid_mask") is None:
                raise ValueError("distillation reward_mode=replace requires teacher evidence on every training batch")
            training_input["loss_mask"] = training_input["loss_mask"] * training_input["teacher_valid_mask"]

        self.trainer.all_metrics.update(
            install_correction(training_input, self.trainer.cfg.trainer.algorithm, reduction=LocalReduction())
        )

        if self.trainer.cfg.trainer.algorithm.use_kl_in_reward:
            with Timer("apply_reward_kl_penalty", self.trainer.all_timings):
                training_input = self.trainer.apply_reward_kl_penalty(training_input)

        # Batch replay runs this prefix outside an optimizer step, without a step wall clock.
        if step_wall is not None:
            step_wall.start("advantages")
        with Timer("compute_advantages_and_returns", self.trainer.all_timings):
            training_input = self.trainer.compute_advantages_and_returns(training_input)
            training_input = self.trainer.finalize_advantages_for_training(training_input)

        self.state.batch = training_input

    def summary(self) -> BatchSummary:
        if self.state.summary is None:
            raise RuntimeError("dense replay has no admission summary")
        return replace(self.state.summary, work=consumed_work(self.batch))

    async def train(self, *, step_wall: StepWallTime | None) -> dict:
        if self.state.empty:
            self.state.timings["train_critic_and_policy"] = 0.0
            self.state.metrics.update({"policy/policy_update_steps": 0.0, "ftpo/empty_batch": 1.0})
            return {"policy_update_steps": 0.0}
        training_input = self.batch
        if self.trainer.cfg.trainer.dump_data_batch:
            if step_wall is not None:
                step_wall.start("training_preparation")
            with Timer("dump_data_batch", self.trainer.all_timings):
                self.trainer.dump_data(
                    training_input, file_name=f"global_step_{self.trainer.global_step}_training_input"
                )

        if step_wall is not None:
            step_wall.start("policy_training")
        with (
            Timer("train_critic_and_policy", self.trainer.all_timings),
            critical_phase("train_step", self.trainer.global_step),
        ):
            status = await asyncio.to_thread(self.trainer.train_critic_and_policy, training_input)
        self.trainer.finish_policy_update(status)
        return status

    async def release(self, error: BaseException | None = None) -> None:
        self.state = None
        self.groups = []


class WorkerBatchSource:
    """Share compact admitted groups and retain assembled rows on policy actors."""

    def __init__(self, trainer: RayPPOTrainer):
        self.trainer = trainer
        self.context, self.policy, self.cfg = trainer.context, trainer.policy_model, trainer.cfg
        self.pad_token_id = trainer.tokenizer.pad_token_id
        self.num_experts = trainer._resolve_num_experts() if moe_router_replay_requested(self.cfg) else None
        self.metrics, self.timings = trainer.all_metrics, trainer.all_timings
        self._metadata: RolloutBatchMetadata | None = None
        self._reader: RolloutReader | None = None
        self._plan: BatchPlan | None = None
        self._pending = []
        self._references = []

    @property
    def admitted_plan(self) -> BatchPlan:
        if self._plan is None:
            raise RuntimeError("worker batch is not prepared")
        return self._plan

    async def admit(self, *, stall_timeout: float) -> dict[str, float]:
        self._metadata = await self.context.next_batch_metadata(stall_timeout=stall_timeout)
        self._reader = self.context.reader(self._metadata, stall_timeout=stall_timeout)
        return self._metadata.metrics

    async def prepare(self, *, global_step: int, step_wall: StepWallTime | None) -> None:
        if step_wall is not None:
            step_wall.start("batch_assembly")
        groups = self._metadata.groups
        facts = tuple(group.row_facts for group in groups)
        require_worker_batch_support(
            self.cfg,
            trainer_type=type(self.trainer),
            facts=facts,
            policy_mesh=tuple(actor.rank for actor in self.policy.actor_infos),
            num_experts=self.num_experts,
        )
        fields = fields_from_facts(facts)
        if not moe_router_replay_requested(self.cfg):
            fields = fields.without_routes()
        self._plan = plan_batch(
            batch_id=self._metadata.batch_id,
            global_step=global_step,
            uids=tuple(group.uid for group in groups for _ in group.row_facts.response_len),
            facts=facts,
            fields=fields,
            rollout_staleness=tuple(
                global_step - group.policy_step for group in groups for _ in group.row_facts.response_len
            ),
            num_experts=self.num_experts,
            algorithm=self.cfg.trainer.algorithm,
        )
        staleness = [global_step - group.policy_step for group in groups]
        record_rollout_staleness(staleness, global_step)
        assert max(staleness) <= self.context.config.max_staleness_steps
        self.metrics.update(
            rollout_metrics(facts, tis_lcs_alert_threshold=float(self.cfg.trainer.algorithm.tis_lcs_alert_threshold))
        )
        self.metrics.update(
            reward_metrics(facts, self._plan.uids, n_samples_per_prompt=self.cfg.generator.n_samples_per_prompt)
        )
        self.metrics.update(
            {
                "async/staleness_mean": sum(staleness) / len(staleness),
                "async/staleness_max": max(staleness),
                "async/staleness_min": min(staleness),
                "async/staleness_ratio": sum(value > 0 for value in staleness) / len(staleness),
            }
        )
        if self.cfg.trainer.training_metrics:
            record_consumed_staleness(
                self._plan.uids,
                self._plan.rollout_staleness,
                [int(value) for group in facts for value in group.response_len],
                step=global_step,
            )
        self._pending = self.policy.read_batch_groups(self._plan, self._reader)
        self._references = list(await self.context.wait(asyncio.gather(*self._pending)))
        self._pending = self.policy.load_batch(self._plan, self._reader, self._references, self.pad_token_id)
        loaded = await self.context.wait(asyncio.gather(*self._pending))
        self._pending = []
        self.timings["load_worker_batch"] = max(loaded)
        self.trainer._log_rollout_batch_completed(
            groups=len(facts),
            sequences=len(self._plan.uids),
            response_tokens=sum(int(group.response_len.sum()) for group in facts),
            duration_seconds=self.timings["wait_for_generation_buffer"],
        )

    async def _run_loaded(self, method: str):
        self._pending = self.policy.async_run_ray_method("pass_through", method, self.admitted_plan.batch_id)
        outputs = await self.context.wait(asyncio.gather(*self._pending))
        self._pending = []
        return outputs

    async def forward(self, *, step_wall: StepWallTime | None) -> None:
        await self.trainer._drain_policy_event_loops()
        with Timer("fwd_logprobs_values_reward", self.timings):
            if self.cfg.trainer.placement.colocate_all:
                await asyncio.to_thread(self.policy.backload_to_gpu, backload_optimizer=False, backload_model=True)
            outputs = await self._run_loaded("forward_loaded")
            self.metrics.update(outputs[0].metadata["batch_metrics"])
            if self.cfg.trainer.placement.colocate_all:
                await asyncio.to_thread(self.policy.offload_to_cpu, offload_optimizer=False, offload_model=True)
            else:
                await self.context.wait(
                    asyncio.gather(*self.policy.async_run_ray_method("pass_through", "empty_cache"))
                )

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
            if self.cfg.trainer.placement.colocate_all:
                await asyncio.to_thread(self.policy.backload_to_gpu)
            elif self.cfg.trainer.offload_optimizer_during_rollouts:
                with Timer("backload_policy_optimizer_to_gpu", self.timings):
                    await asyncio.to_thread(self.policy.backload_to_gpu, backload_optimizer=True, backload_model=False)
            with Timer("policy_train", self.timings):
                self._pending = self.policy.async_run_ray_method(
                    "pass_through", "train_loaded", self.admitted_plan.batch_id
                )
                outputs = await asyncio.to_thread(
                    collect_actor_results, self.policy.actor_infos, self._pending, operation="policy train_loaded"
                )
                self._pending = []
            status = outputs[0].metadata["train_status"]
            self.metrics.update({f"policy/{key}": value for key, value in status.items()})
            await self.context.wait(asyncio.gather(*self.policy.async_run_ray_method("pass_through", "empty_cache")))
        self.trainer.finish_policy_update(status)
        return status

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
            len(plan.facts),
            consumed_stop_metrics(
                tuple(value for group in plan.facts for value in group.stop_reasons)
                if "stop_reasons" in fields.first_group_list_fields
                else None,
                len(plan.uids),
            ),
            tuple(float(value) for group in plan.facts for value in group.scalar_values)
            if not fields.token_rewards
            else (),
        )

    async def release(self, error: BaseException | None = None) -> None:
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
            self._pending, self._references = [], []
            self._plan, self._reader, self._metadata = None, None, None


def make_batch_source(mode: BatchBuilder, *, driver: DriverBatchSource) -> BatchSource:
    """Select the configured batch lifetime with lazy worker dependencies."""
    if mode is BatchBuilder.DRIVER:
        return driver
    require_worker_batch_support(driver.trainer.cfg, trainer_type=type(driver.trainer))
    from skyrl_train.batch_verification import VerifyBatchSource

    worker = WorkerBatchSource(driver.trainer)
    if mode is BatchBuilder.WORKER:
        return worker
    return VerifyBatchSource(
        worker,
        context=driver.context,
        policy=driver.trainer.policy_model,
        cfg=driver.trainer.cfg,
        tokenizer=driver.trainer.tokenizer,
        timings=driver.trainer.all_timings,
    )


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
