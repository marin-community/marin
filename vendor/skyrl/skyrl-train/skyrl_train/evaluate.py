import asyncio
import json
from contextlib import asynccontextmanager, nullcontext
import torch
from skyrl_train.utils.progress import tqdm
from typing import Dict, List, Protocol
from loguru import logger
from collections import defaultdict
from dataclasses import dataclass, replace
from collections.abc import Sequence
from math import fsum
import numpy as np

from finestore.rl.run_records import (
    EvaluationAnswerRow,
    EvaluationTaskRow,
    EvaluationTraceRow,
    EvaluationTraceVerification,
    write_evaluations,
)
from finestore.rl.strata import TaskEvaluationIdentity
from skyrl_train.task_inputs import TrajectoryMetadata
from skyrl_train.trajectory_runners.types import TaskRequestBatch

from skyrl_train.evaluation_pool import EvaluationPool
from skyrl_train.grader_reporting import GraderStats, write_grader_report
from skyrl_train.metric_names import EVAL_OUTCOME_METRIC_PREFIX, GENERATE_OUTCOME_METRIC_PREFIX

from skyrl_train.trajectory_runners.trajectory_processing import (
    concatenate_trajectory_batches,
    get_metrics_from_trajectory_batch,
    get_trajectory_passes,
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
from skyrl_train.utils.trainer_utils import calculate_per_dataset_metrics, evaluation_response_metrics
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

TRANSCRIPT_FIELDS = (
    "prompt",
    "response.messages",
    "response.text",
    "response.token_ids",
    "response.step_boundaries",
)


@dataclass
class _EvaluationRollouts:
    batch: TrajectoryBatch
    uids: List[str]
    metadata: List[TrajectoryMetadata]
    ordinals: List[int]
    example_prompt: ConversationType
    example_batch: TrajectoryBatch

    @property
    def data_sources(self) -> List[str | None]:
        return [row.data_source for row in self.metadata]


class _EvaluationAccumulator(Protocol):
    uids: List[str]
    metadata: List[TrajectoryMetadata]

    def record(self, request: TrajectoryRequestBatch, batch: TrajectoryBatch, uids: List[str]) -> None: ...


@dataclass
class _WholeTrajectoryAccumulator:
    uids: List[str]
    metadata: List[TrajectoryMetadata]

    def record(self, request: TrajectoryRequestBatch, batch: TrajectoryBatch, uids: List[str]) -> None:
        validate_trajectory_batch(len(request.metadata), batch)
        self.uids.extend(uids)
        self.metadata.extend(request.metadata)
        if isinstance(request, TaskRequestBatch):
            expected = [row.spec.task.id for row in request.rows]
            if batch.get("task_ids") != expected:
                raise ValueError("evaluation output task IDs differ from request TaskSpecs")


@dataclass
class _StepWiseAccumulator:
    uids: List[str]
    metadata: List[TrajectoryMetadata]

    def record(self, request: TrajectoryRequestBatch, batch: TrajectoryBatch, uids: List[str]) -> None:
        del uids
        trajectory_ids = request.trajectory_ids
        output_ids = batch.get("trajectory_ids")
        if trajectory_ids is None or output_ids is None:
            raise ValueError("step-wise evaluation requires trajectory IDs")
        inputs_by_id = {
            trajectory_id.instance_id: row for trajectory_id, row in zip(trajectory_ids, request.metadata, strict=True)
        }
        expected_ids = (
            {
                trajectory_id.instance_id: row.spec.task.id
                for trajectory_id, row in zip(trajectory_ids, request.rows, strict=True)
            }
            if isinstance(request, TaskRequestBatch)
            else None
        )
        if expected_ids is not None and batch.get("task_ids") != [
            expected_ids[item.instance_id] for item in output_ids
        ]:
            raise ValueError("step-wise evaluation output task IDs differ from request TaskSpecs")
        for trajectory_id in output_ids:
            if trajectory_id.instance_id not in inputs_by_id:
                raise ValueError(f"Trajectory ID {trajectory_id.instance_id} not found in input")
            self.metadata.append(inputs_by_id[trajectory_id.instance_id])
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
        batch=concatenate_trajectory_batches(trajectory_batches),
        uids=accumulator.uids,
        metadata=accumulator.metadata,
        ordinals=list(range(len(accumulator.uids))),
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
    metrics.update(
        {
            EVAL_OUTCOME_METRIC_PREFIX + key.removeprefix(GENERATE_OUTCOME_METRIC_PREFIX): value
            for key, value in (batch.get("rollout_metrics") or {}).items()
            if key.startswith(GENERATE_OUTCOME_METRIC_PREFIX)
        }
    )
    return metrics


def _evaluation_task_rows(
    rollouts: _EvaluationRollouts,
    *,
    attempt_id: str,
    step: int,
    evaluation: str,
    model: str | None,
    model_source_identity: str | None,
    starting_checkpoint_identity: str | None,
) -> list[EvaluationTaskRow]:
    batch = rollouts.batch
    count = len(batch["response_ids"])
    normalized = normalized_verifier_scores(batch) or [None] * count
    passes = get_trajectory_passes(batch)
    groups: dict[str, list[EvaluationAnswerRow]] = defaultdict(list)
    identities: dict[str, TaskEvaluationIdentity] = {}
    sources: dict[str, str | None] = {}
    for index, (metadata, uid, ordinal) in enumerate(
        zip(rollouts.metadata, rollouts.uids, rollouts.ordinals, strict=True)
    ):
        task_id = (batch.get("task_ids") or [None] * count)[index]
        if task_id is None:
            continue
        identity = metadata.evaluation_identity or TaskEvaluationIdentity()
        if task_id in identities and identities[task_id] != identity:
            raise ValueError(f"evaluation task has conflicting panel strata: {task_id}")
        identities[task_id] = identity
        sources[task_id] = metadata.data_source
        verification = (batch.get("verification_results") or [None] * count)[index]
        reward = batch["rewards"][index]
        groups[task_id].append(
            EvaluationAnswerRow(
                ordinal=ordinal,
                native_uid=uid,
                optimization_reward=fsum(reward) if isinstance(reward, list) else float(reward),
                verifier_score=normalized[index],
                raw_verifier_score=verification.score if verification is not None else None,
                score_min=verification.score_min if verification is not None else None,
                score_max=verification.score_max if verification is not None else None,
                verification_status=verification.status.value if verification is not None else None,
                passed=passes[index],
                outcome=batch["task_outcomes"][index].value,
                error_type=(batch.get("exception_types") or [None] * count)[index],
                error_treatment=(batch.get("error_treatments") or [None] * count)[index],
                response_token_count=len(batch["response_ids"][index]),
            )
        )
    rows = []
    for task_id, answers in groups.items():
        identity = identities[task_id]
        scores = [answer.verifier_score for answer in answers if answer.verifier_score is not None]
        outcomes = {answer.outcome for answer in answers}
        errors = {answer.error_type for answer in answers}
        rows.append(
            EvaluationTaskRow(
                attempt_id=attempt_id,
                step=step,
                evaluation=evaluation,
                panel_name=identity.panel_name,
                panel_version=identity.panel_version,
                task_id=task_id,
                data_source=sources[task_id],
                strata=identity.strata,
                task_digest=identity.task_digest,
                score=float(np.mean(scores)) if scores else None,
                outcome=next(iter(outcomes)) if len(outcomes) == 1 else "mixed",
                error_type=next(iter(errors)) if len(errors) == 1 else "multiple",
                answers=answers,
                model=model,
                model_source_identity=model_source_identity,
                starting_checkpoint_identity=starting_checkpoint_identity,
            )
        )
    return rows


def _has_redaction(redact_fields: Sequence[str], paths: tuple[str, ...]) -> bool:
    return any(
        field == path or field.startswith(f"{path}.") or path.startswith(f"{field}.")
        for field in redact_fields
        for path in paths
    )


def _evaluation_trace_rows(
    rollouts: _EvaluationRollouts,
    tokenizer: AutoTokenizer,
    *,
    attempt_id: str,
    step: int,
    evaluation: str,
    redact_fields: Sequence[str],
) -> list[EvaluationTraceRow]:
    batch = rollouts.batch
    count = len(rollouts.uids)
    missing = (None,) * count
    task_ids = batch.get("task_ids")
    identity_disposition = "task_spec" if task_ids else "preference_input"
    last_steps = batch.get("is_last_step") or (True,) * count
    # A turn's prompt and messages can repeat content from earlier turns.
    keep_transcript = not _has_redaction(redact_fields, TRANSCRIPT_FIELDS)
    messages = (batch.get("evidence_messages") or missing) if keep_transcript else missing
    keep_verification = not _has_redaction(redact_fields, ("verification_result",))
    verifications = (batch.get("verification_results") or missing) if keep_verification else missing
    stop_reasons = batch.get("stop_reasons") or missing
    exceptions = batch.get("exception_types") or missing
    treatments = batch.get("error_treatments") or missing
    rows = []
    for index, (metadata, uid, ordinal) in enumerate(
        zip(rollouts.metadata, rollouts.uids, rollouts.ordinals, strict=True)
    ):
        prompt_ids = batch["prompt_token_ids"][index] if keep_transcript else None
        response_ids = batch["response_ids"][index] if keep_transcript else None
        reward = batch["rewards"][index]
        verification = verifications[index]
        verification_details = None
        if verification is not None:
            verification_details = EvaluationTraceVerification(
                status=verification.status.value,
                reason=verification.reason,
                score=verification.score,
                passed=verification.passed,
                score_min=verification.score_min,
                score_max=verification.score_max,
            )
        rows.append(
            EvaluationTraceRow(
                attempt_id=attempt_id,
                step=step,
                evaluation=evaluation,
                trace_index=index,
                ordinal=ordinal,
                native_uid=uid,
                task_id=task_ids[index] if task_ids is not None else None,
                data_source=metadata.data_source,
                is_last_step=last_steps[index],
                identity_disposition=identity_disposition,
                prompt_token_ids=prompt_ids,
                response_token_ids=response_ids,
                input_prompt=tokenizer.decode(prompt_ids) if prompt_ids is not None else None,
                output_response=tokenizer.decode(response_ids) if response_ids is not None else None,
                score=fsum(reward) if isinstance(reward, list) else reward,
                token_rewards=reward if isinstance(reward, list) else None,
                stop_reason=stop_reasons[index],
                exception_type=exceptions[index],
                error_treatment=treatments[index],
                env_class=metadata.metric_class,
                subset_id=metadata.subset_id,
                verification=verification_details,
                messages_json=json.dumps(messages[index], ensure_ascii=False) if messages[index] is not None else None,
            )
        )
    return rows


def _persist_evaluation(
    cfg: DictConfig,
    global_step: int | None,
    tokenizer: AutoTokenizer,
    rollouts: _EvaluationRollouts,
    *,
    val_set_name: str | None,
    records_export_root: str | None,
    final_rollouts: _EvaluationRollouts | None = None,
) -> None:
    attempt_id = str(cfg.runtime.attempt_id)
    step = global_step if global_step is not None else -1
    evaluation = "eval/" if val_set_name is None else f"eval/{val_set_name}/"
    rows = _evaluation_task_rows(
        final_rollouts if final_rollouts is not None else rollouts,
        attempt_id=attempt_id,
        step=step,
        evaluation=evaluation,
        model=cfg.runtime.get("model"),
        model_source_identity=cfg.runtime.get("model_source_identity"),
        starting_checkpoint_identity=cfg.runtime.get("starting_checkpoint_identity"),
    )
    traces = []
    if cfg.trainer.dump_eval_results:
        retention = cfg.generator.get("trajectory_retention") or {}
        traces = _evaluation_trace_rows(
            rollouts,
            tokenizer,
            attempt_id=attempt_id,
            step=step,
            evaluation=evaluation,
            redact_fields=retention.get("redact_fields", ()),
        )
    write_evaluations(records_export_root or str(cfg.trainer.export_path), rows, traces=traces)
    if cfg.trainer.get("dump_grader_report", False):
        stats = GraderStats()
        stats.add(rollouts.batch, rollouts.metadata)
        summary = stats.summary()
        key = "eval_only" if global_step is None else str(global_step)
        if val_set_name is not None:
            key = f"{val_set_name}/{key}"
        write_grader_report(
            str(cfg.trainer.grader_report_dir),
            key,
            {"evaluation": summary["subsets"], "evaluation_tasks": summary["tasks"]},
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
    *,
    records_export_root: str | None = None,
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
        accumulator = _WholeTrajectoryAccumulator([], [])
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
            rollouts.data_sources,
            cfg.generator.eval_n_samples_per_prompt,
        )
        _persist_evaluation(
            cfg, global_step, tokenizer, rollouts, val_set_name=val_set_name, records_export_root=records_export_root
        )

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
    *,
    records_export_root: str | None = None,
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
    accumulator = _StepWiseAccumulator([], [])
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
    trajectory_batch_last_step["rollout_metrics"] = concatenated_batch.get("rollout_metrics") or {}
    final_rollouts = replace(
        rollouts,
        batch=trajectory_batch_last_step,
        uids=uids_last_step,
        metadata=[row for row, final in zip(rollouts.metadata, is_last_step_mask, strict=True) if final],
        ordinals=[ordinal for ordinal, final in zip(rollouts.ordinals, is_last_step_mask, strict=True) if final],
    )
    eval_metrics = _calculate_eval_metrics(
        final_rollouts.batch, final_rollouts.uids, final_rollouts.data_sources, cfg.generator.eval_n_samples_per_prompt
    )
    _persist_evaluation(
        cfg,
        global_step,
        tokenizer,
        rollouts,
        val_set_name=val_set_name,
        records_export_root=records_export_root,
        final_rollouts=final_rollouts,
    )

    return eval_metrics
