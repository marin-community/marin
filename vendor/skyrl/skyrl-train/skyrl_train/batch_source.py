from __future__ import annotations

import asyncio
from dataclasses import dataclass, replace
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING


from marinskyrl.distillation import DistillationRewardMode
from skyrl_train.batch_metrics import (
    consumed_work,
    install_correction,
)
from skyrl_train.rollouts.buffer import RolloutGroup
from skyrl_train.rollouts.context import TrainingContext
from skyrl_train.telemetry import ConsumedWork, critical_phase
from skyrl_train.timing_observability import StepWallTime
from skyrl_train.training_batch import TrainingInputBatch
from skyrl_train.utils import Timer
from skyrl_train.utils.importance_ratio_diagnostics import LocalReduction

if TYPE_CHECKING:
    from skyrl_train.trainer import RayPPOTrainer


BATCH_CLEANUP_TIMEOUT_SECONDS = 30.0


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


@asynccontextmanager
async def batch_lifetime(source: DriverBatchSource):
    """Release a source's owned batch while retaining the initiating exception."""
    error = None
    try:
        yield source
    except BaseException as caught:
        error = caught
        raise
    finally:
        await source.release(error)
