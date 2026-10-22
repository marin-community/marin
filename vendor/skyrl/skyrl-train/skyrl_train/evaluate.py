import asyncio
from contextlib import asynccontextmanager, nullcontext
import torch
from skyrl_train.utils.progress import tqdm
from typing import Dict, List, Protocol
from loguru import logger
from collections import defaultdict
from dataclasses import dataclass, replace

from skyrl_train.utils import Timer
from skyrl_train.evaluation_pool import EvaluationPool
from skyrl_train.io import io
from marinskyrl.resource_locator import join_resource_path

from skyrl_train.trajectory_runners.trajectory_processing import (
    concatenate_trajectory_batches,
    get_metrics_from_trajectory_batch,
    normalized_verifier_scores,
    prepare_trajectory_request,
    verifier_score_summary,
)
from skyrl_train.trajectory_runners.base import (
    ConversationType,
    TrajectoryBatch,
    TrajectoryRequestBatch,
    TrajectoryRunner,
)
from skyrl_train.utils.trainer_utils import (
    calculate_per_dataset_metrics,
    dump_per_dataset_eval_results,
    evaluation_response_metrics,
)
from skyrl_train.trajectory_runners.trajectory_processing import validate_trajectory_batch
from skyrl_train.inference_engines.utils import get_sampling_params_for_backend
from skyrl_train.utils.logging_utils import log_example
from skyrl_train.trajectory_runners.trajectory_retention import (
    RetentionSink,
    make_trajectory_sink,
)

from omegaconf import DictConfig
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import AutoTokenizer


@dataclass
class _EvaluationRollouts:
    batch: TrajectoryBatch
    env_classes: List[str]
    data_sources: List[str | None]
    uids: List[str]
    example_prompt: ConversationType
    example_batch: TrajectoryBatch


class _EvaluationAccumulator(Protocol):
    env_classes: List[str]
    data_sources: List[str | None]
    uids: List[str]

    def record(self, request: TrajectoryRequestBatch, batch: TrajectoryBatch, uids: List[str]) -> None: ...


def evaluation_dump_dir(export_path: str, global_step: int | None) -> str:
    """Return the directory containing one evaluation session's persisted results."""
    session = "eval_only" if global_step is None else f"global_step_{global_step}_evals"
    return join_resource_path(export_path, "dumped_evals", session)


@dataclass
class _WholeTrajectoryAccumulator:
    env_classes: List[str]
    data_sources: List[str | None]
    uids: List[str]

    def record(self, request: TrajectoryRequestBatch, batch: TrajectoryBatch, uids: List[str]) -> None:
        validate_trajectory_batch(len(request.metadata), batch)
        self.env_classes.extend(row.metric_class for row in request.metadata)
        self.data_sources.extend(row.data_source for row in request.metadata)
        self.uids.extend(uids)


@dataclass
class _StepWiseAccumulator:
    env_classes: List[str]
    data_sources: List[str | None]
    uids: List[str]

    def record(self, request: TrajectoryRequestBatch, batch: TrajectoryBatch, uids: List[str]) -> None:
        del uids
        trajectory_ids = request.trajectory_ids
        output_ids = batch.get("trajectory_ids")
        if trajectory_ids is None or output_ids is None:
            raise ValueError("step-wise evaluation requires trajectory IDs")
        inputs_by_id = {
            trajectory_id.instance_id: (row.metric_class, row.data_source)
            for trajectory_id, row in zip(trajectory_ids, request.metadata, strict=True)
        }
        for trajectory_id in output_ids:
            if trajectory_id.instance_id not in inputs_by_id:
                raise ValueError(f"Trajectory ID {trajectory_id.instance_id} not found in input")
            env_class, data_source = inputs_by_id[trajectory_id.instance_id]
            self.env_classes.append(env_class)
            self.data_sources.append(data_source)
            self.uids.append(trajectory_id.instance_id)


async def _collect_evaluation_rollouts(
    eval_dataloader: StatefulDataLoader,
    trajectory_runner: TrajectoryRunner,
    cfg: DictConfig,
    global_step: int | None,
    val_set_name: str | None,
    accumulator: _EvaluationAccumulator,
    pool: EvaluationPool,
) -> _EvaluationRollouts:
    """Collect every selected row and repetition within the shared evaluation pool."""
    trajectory_batches: List[TrajectoryBatch] = []
    last_request = None
    last_batch = None
    pbar = None
    try:
        rows = [row for prompts in eval_dataloader for row in prompts]
        pbar = tqdm(total=len(rows) * cfg.generator.eval_n_samples_per_prompt, initial=0, desc="Evaluation Progress")

        async def collect(row, repetition):
            nonlocal last_request, last_batch
            request, uids = prepare_trajectory_request(
                [row],
                1,
                get_sampling_params_for_backend(cfg.generator.backend, cfg.generator.eval_sampling_params),
                "eval",
                global_step,
            )
            request = replace(
                request,
                trajectory_ids=tuple(
                    replace(
                        item,
                        repetition_id=repetition,
                        instance_id=f"eval-{global_step}-{len(val_set_name or '')}-{val_set_name or ''}-{item.instance_id}",
                    )
                    for item in request.trajectory_ids
                ),
            )
            source = "native" if val_set_name is None else f"native/{val_set_name}"
            batch = await pool.run(source, lambda: trajectory_runner.run(request))
            trajectory_batches.append(batch)
            last_request, last_batch = request, batch

            accumulator.record(request, batch, uids)
            pbar.update(1)

        async with asyncio.TaskGroup() as tasks:
            for repetition in range(cfg.generator.eval_n_samples_per_prompt):
                for row in rows:
                    tasks.create_task(collect(row, repetition))
    finally:
        if pbar is not None:
            pbar.close()

    if last_request is None or last_batch is None:
        raise ValueError("evaluation dataloader produced no batches")
    if len(trajectory_batches) != len(rows) * cfg.generator.eval_n_samples_per_prompt:
        raise RuntimeError("evaluation did not return every selected row and repetition")
    return _EvaluationRollouts(
        batch=concatenate_trajectory_batches(
            trajectory_batches,
            tis_lcs_alert_threshold=float(cfg.trainer.algorithm.tis_lcs_alert_threshold),
        ),
        env_classes=accumulator.env_classes,
        data_sources=accumulator.data_sources,
        uids=accumulator.uids,
        example_prompt=last_request.public_prompts[0],
        example_batch=last_batch,
    )


@asynccontextmanager
async def evaluation_session(trajectory_runner, cfg, global_step, sink):
    """Reserve the whole rollout pool once for all validation sources."""
    trajectory_runner.set_trajectory_sink(sink)
    await trajectory_runner.start_eval_session(
        run_name=getattr(cfg.trainer, "run_name", None) or "eval",
        eval_step=global_step if global_step is not None else 0,
        val_set_name=None,
    )
    try:
        concurrency = cfg.trajectory_runner.eval_max_concurrent_tasks
        if concurrency is None:
            concurrency = cfg.trajectory_runner.max_concurrent_tasks
        async with EvaluationPool(concurrency) as pool:
            yield pool
    finally:
        await trajectory_runner.stop_eval_session()


def _calculate_eval_metrics(
    batch: TrajectoryBatch,
    uids: List[str],
    data_sources: List[str | None],
    samples_per_prompt: int,
) -> Dict[str, float]:
    metrics = calculate_per_dataset_metrics(batch, uids, data_sources, samples_per_prompt)
    overall_avg_score, overall_pass_at_n = get_metrics_from_trajectory_batch(batch, uids)
    metrics.update(
        {
            "eval/all/num_attempted": float(len(batch["response_ids"])),
            "eval/all/avg_score": overall_avg_score,
            f"eval/all/pass_at_{samples_per_prompt}": overall_pass_at_n,
        }
    )
    verifier_scores = normalized_verifier_scores(batch)
    if verifier_scores is not None:
        coverage, average = verifier_score_summary(verifier_scores)
        metrics["eval/all/num_scored"] = float(sum(score is not None for score in verifier_scores))
        metrics["eval/all/verifier_score_coverage"] = coverage
        if average is not None:
            metrics["eval/all/avg_verifier_score"] = average
    metrics.update({f"eval/all/{key}": value for key, value in evaluation_response_metrics(batch).items()})
    return metrics


def _dump_eval_results(
    cfg: DictConfig,
    global_step: int | None,
    tokenizer: AutoTokenizer,
    rollouts: _EvaluationRollouts,
    data_sources: List[str | None],
    metrics: Dict[str, float],
) -> None:
    if not cfg.trainer.dump_eval_results:
        return
    with Timer("dump_eval_results"):
        data_save_dir = evaluation_dump_dir(str(cfg.trainer.export_path), global_step)
        io.makedirs(data_save_dir, exist_ok=True)
        dump_per_dataset_eval_results(
            data_save_dir,
            tokenizer,
            rollouts.batch,
            data_sources,
            rollouts.env_classes,
            metrics,
            rollouts.uids,
        )


@torch.no_grad()
async def evaluate(
    eval_dataloader: StatefulDataLoader,
    trajectory_runner: TrajectoryRunner,
    cfg: DictConfig,
    global_step: int | None,
    tokenizer: AutoTokenizer,
    val_set_name: str | None = None,
    trajectory_sink: RetentionSink | None = None,
    evaluation_pool: EvaluationPool | None = None,
) -> Dict[str, float]:
    """Runs generation and evaluation of trajectories.

    Args:
        eval_dataloader (StatefulDataLoader): dataloader of the eval dataset
        trajectory_runner (TrajectoryRunner): runner to use
        cfg (DictConfig): config
        global_step (int | None): current global step, or
            `None` to indicate a non-training context (e.g., eval-only)
        tokenizer (AutoTokenizer): tokenizer to use
        val_set_name (str | None): optional name of the validation set being evaluated,
            used for unique orchestrator naming
        trajectory_sink (RetentionSink | None): trainer-owned retention sink; a standalone evaluation builds one from cfg

    Returns:
        Dict[str, float]: evaluation metrics
    """
    owns_sink = trajectory_sink is None
    active_sink = trajectory_sink or make_trajectory_sink(cfg.generator, tokenizer)
    try:
        accumulator = _WholeTrajectoryAccumulator([], [], [])
        session = (
            nullcontext(evaluation_pool)
            if evaluation_pool is not None
            else evaluation_session(trajectory_runner, cfg, global_step, active_sink)
        )
        async with session as pool:
            rollouts = await _collect_evaluation_rollouts(
                eval_dataloader, trajectory_runner, cfg, global_step, val_set_name, accumulator, pool
            )
        concatenated_batch = rollouts.batch
        concat_data_sources = rollouts.data_sources
        vis = tokenizer.decode(rollouts.example_batch["response_ids"][0])
        log_example(
            logger,
            prompt=rollouts.example_prompt,
            response=vis,
            reward=rollouts.example_batch["rewards"][0],
        )

        eval_metrics = _calculate_eval_metrics(
            concatenated_batch,
            rollouts.uids,
            concat_data_sources,
            cfg.generator.eval_n_samples_per_prompt,
        )
        _dump_eval_results(cfg, global_step, tokenizer, rollouts, concat_data_sources, eval_metrics)

        return eval_metrics
    finally:
        if owns_sink:
            active_sink.close()


@torch.no_grad()
async def evaluate_step_wise(
    eval_dataloader: StatefulDataLoader,
    trajectory_runner: TrajectoryRunner,
    cfg: DictConfig,
    global_step: int | None,
    tokenizer: AutoTokenizer,
    trajectory_sink: RetentionSink,
    val_set_name: str | None = None,
    evaluation_pool: EvaluationPool | None = None,
) -> Dict[str, float]:
    """Runs generation and evaluation of trajectories for step-wise training.

    Currently assumes that the rewards are assigned to the last step of each trajectory.

    Args:
        eval_dataloader (StatefulDataLoader): dataloader of the eval dataset
        trajectory_runner (TrajectoryRunner): runner to use
        cfg (DictConfig): config
        global_step (int | None): current global step, or
            `None` to indicate a non-training context (e.g., eval-only)
        tokenizer (AutoTokenizer): tokenizer to use
        val_set_name (str | None): optional name of the validation set being evaluated,
            used for unique orchestrator naming
        trajectory_sink (RetentionSink): trainer-owned retention sink

    Returns:
        Dict[str, float]: evaluation metrics
    """
    accumulator = _StepWiseAccumulator([], [], [])
    session = (
        nullcontext(evaluation_pool)
        if evaluation_pool is not None
        else evaluation_session(trajectory_runner, cfg, global_step, trajectory_sink)
    )
    async with session as pool:
        rollouts = await _collect_evaluation_rollouts(
            eval_dataloader, trajectory_runner, cfg, global_step, val_set_name, accumulator, pool
        )
    concatenated_batch = rollouts.batch
    concat_data_sources = rollouts.data_sources
    vis = tokenizer.decode(rollouts.example_batch["response_ids"][0])
    logger.info(f"Eval output example: {vis}")

    # Only use the final step metrics
    trajectory_batch_last_step = defaultdict(list)
    is_last_step_mask = concatenated_batch["is_last_step"]
    for key in concatenated_batch:
        if isinstance(concatenated_batch[key], list):
            assert len(concatenated_batch[key]) == len(is_last_step_mask)
            trajectory_batch_last_step[key] = [
                val for val, is_last_step in zip(concatenated_batch[key], is_last_step_mask) if is_last_step
            ]
    uids_last_step = [uid for uid, is_last_step in zip(rollouts.uids, is_last_step_mask) if is_last_step]
    data_sources_last_step = [
        data_source for data_source, is_last_step in zip(concat_data_sources, is_last_step_mask) if is_last_step
    ]

    eval_metrics = _calculate_eval_metrics(
        trajectory_batch_last_step,
        uids_last_step,
        data_sources_last_step,
        cfg.generator.eval_n_samples_per_prompt,
    )
    _dump_eval_results(cfg, global_step, tokenizer, rollouts, concat_data_sources, eval_metrics)

    return eval_metrics
