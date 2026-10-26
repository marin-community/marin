import asyncio

from loguru import logger

from skyrl_train.batch_assembly import assemble_slice, forward_input
from skyrl_train.batch_digest import BatchInputPhase, digest
from skyrl_train.batch_metrics import advantage_metrics, install_correction
from skyrl_train.batch_source import BatchSummary, WorkerBatchSource
from skyrl_train.distributed.dispatch import concatenate_outputs_after_mesh_dispatch
from skyrl_train.group_admission import GroupAdvantageInvariant
from skyrl_train.training_batch import TrainingInputBatch, dump_training_batch
from skyrl_train.trajectory_runners.trajectory_processing import concatenate_trajectory_batches, token_reward_rows
from skyrl_train.timing_observability import StepWallTime
from skyrl_train.utils import Timer
from skyrl_train.utils.advantage_estimators import (
    apply_loop_advantages,
    batch_advantages,
    drop_advantage_inputs,
)
from skyrl_train.utils.importance_ratio_diagnostics import LocalReduction


class VerifyBatchSource:
    """Compare isolated driver inputs before worker forward and optimization."""

    def __init__(self, worker: WorkerBatchSource):
        self.worker = worker
        self.context, self.policy, self.cfg = worker.context, worker.policy, worker.cfg
        self.pad_token_id, self.timings = worker.pad_token_id, worker.timings
        self.reference: TrainingInputBatch | None = None

    async def admit(self, *, stall_timeout: float) -> dict[str, float]:
        return await self.worker.admit(stall_timeout=stall_timeout)

    async def prepare(self, *, global_step: int, step_wall: StepWallTime | None) -> None:
        await self.worker.prepare(global_step=global_step, step_wall=step_wall)
        plan = self.worker.admitted_plan
        reader = self.worker.admitted_reader
        groups = await self.context.wait(reader.read(plan.batch_id, tuple(group.index for group in reader.selected)))
        with Timer("verify_driver_preparation", self.timings):
            trajectories = concatenate_trajectory_batches(
                [group.trajectory_batch for group in groups],
                tis_lcs_alert_threshold=float(self.cfg.trainer.algorithm.tis_lcs_alert_threshold),
            )
            trajectories["rewards"] = token_reward_rows(trajectories["rewards"], trajectories["response_ids"])
            batch = assemble_slice(
                plan,
                range(len(plan.uids)),
                trajectories,
                pad_token_id=self.pad_token_id,
                algorithm=self.cfg.trainer.algorithm,
            )
            batch.metadata["pad_size"] = 0
            self.reference = batch

    async def _compare(self, phase: BatchInputPhase) -> None:
        plan = self.worker.admitted_plan
        expected_batch = forward_input(self.reference) if phase is BatchInputPhase.FORWARD else self.reference
        expected_batch = TrainingInputBatch(dict(expected_batch), routed_expert_rows=expected_batch.routed_expert_rows)
        expected_batch.metadata = dict(self.reference.metadata)
        expected_batch.metadata["global_step"] = plan.global_step
        dp_size = self.policy.actor_infos[0].rank.dp_size
        expected = [digest(chunk) for chunk in expected_batch.chunk(expected_batch.batch_size // dp_size)]
        observed = await self.context.wait(
            asyncio.gather(*self.policy.async_run_ray_method("pass_through", "digest_loaded", plan.batch_id, phase))
        )
        for actor, actual in zip(self.policy.actor_infos, observed, strict=True):
            if actual != expected[actor.rank.dp]:
                raise ValueError(f"worker {phase} input digest mismatch on rank {actor.rank}")
        logger.info("Worker {} input digests verified on {} ranks", phase, len(observed))

    async def forward(self) -> None:
        await self._compare(BatchInputPhase.FORWARD)
        await self.context.wait(asyncio.gather(*self.policy.async_run_ray_method("pass_through", "barrier_all")))
        with Timer("verify_driver_forward", self.timings):
            data = forward_input(self.reference)
            data.metadata["global_step"] = self.worker.admitted_plan.global_step
            outputs = await self.context.wait(
                asyncio.gather(*self.policy.async_run_ray_method("mesh", "forward", data))
            )
            batch = self.reference
            batch["action_log_probs"] = concatenate_outputs_after_mesh_dispatch(self.policy.actor_infos, outputs)[
                "output"
            ]
            batch["base_action_log_probs"], batch["values"] = None, None
            batch.pop("rollout_staleness")
        await self.worker.forward()

    async def finalize(self, *, step_wall: StepWallTime | None) -> None:
        with Timer("verify_driver_finalization", self.timings):
            batch, algorithm = self.reference, self.cfg.trainer.algorithm
            install_correction(batch, algorithm, reduction=LocalReduction())
            invariant = GroupAdvantageInvariant.from_config(algorithm.resolved_group_advantage)
            batch["advantages"], batch["returns"] = batch_advantages(batch, algorithm, invariant=invariant)
            advantage_metrics(batch, algorithm=algorithm, reduction=LocalReduction(), step_wise=False)
            self.reference = drop_advantage_inputs(apply_loop_advantages(batch))
        await self.worker.finalize(step_wall=step_wall)
        await self._compare(BatchInputPhase.TRAIN)

    async def train(self, *, step_wall: StepWallTime | None) -> dict:
        if self.cfg.trainer.dump_data_batch:
            dump_training_batch(
                self.reference,
                str(self.cfg.trainer.export_path),
                f"global_step_{self.worker.admitted_plan.global_step}_training_input",
            )
        return await self.worker.train(step_wall=step_wall)

    def summary(self) -> BatchSummary:
        return self.worker.summary()

    async def release(self, error: BaseException | None = None) -> None:
        try:
            await self.worker.release(error)
        finally:
            self.reference = None
