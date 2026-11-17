"""Build teacher-scoring work and schedule it on bounded per-teacher queues."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

import torch
from torch.nn.utils.rnn import pad_sequence

from marinskyrl.distillation import TeacherEvidenceKind
from skyrl_train.distillation import (
    ChosenTokenTeacherEvidence,
    PreparedTeacherInput,
    INVALID_TOPK_INDEX,
    ChosenTokenTeacherInput,
    TeacherTopKInput,
    StudentSelectedTeacherEvidence,
    StudentTopKInput,
    TeacherEvidenceBatch,
    TeacherScoreRequest,
    TopKTeacherEvidence,
    prepare_sampled_reverse_kl,
    prepare_teacher_topk,
    prepare_student_topk_policy_surrogate,
)
from skyrl_train.teacher_oracle import TeacherOracleCollection
from skyrl_train.teacher_routing import RoutedTrajectoryBatch, TeacherRoute
from skyrl_train.trajectory_runners.types import TrajectoryBatch


async def _gather_or_cancel(tasks: list[asyncio.Future[Any]]) -> list[Any]:
    """Gather tasks, cancelling and draining every sibling if one fails."""
    try:
        return list(await asyncio.gather(*tasks))
    except BaseException:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise


@dataclass(frozen=True)
class TeacherScoringWork:
    """One transport-neutral request plus its learner-side objective weights."""

    request: TeacherScoreRequest
    coefficient: float
    route_weights: torch.Tensor


@dataclass(frozen=True)
class ScoredDistillationBatch:
    """Validated evidence and the minimal payload consumed by the learner."""

    evidence: TeacherEvidenceBatch
    distillation: PreparedTeacherInput


@dataclass(frozen=True)
class RoutedTeacherScoringPartition:
    """One logical-teacher request and its original mixed-batch coordinates."""

    original_indices: tuple[int, ...]
    work: TeacherScoringWork


@dataclass(frozen=True)
class RoutedTeacherScoringWork:
    """Teacher-homogeneous requests derived from one mixed learner batch."""

    trajectory_ids: tuple[str, ...]
    routes: tuple[TeacherRoute, ...]
    response_lengths: tuple[int, ...]
    plan_version: str
    partitions: tuple[RoutedTeacherScoringPartition, ...]


@dataclass(frozen=True)
class RoutedScoredDistillationBatch:
    """Reassembled multi-teacher evidence in original learner row order."""

    trajectory_ids: tuple[str, ...]
    routes: tuple[TeacherRoute, ...]
    teacher_revisions: tuple[str, ...]
    plan_version: str
    evidence: tuple[TeacherEvidenceBatch, ...]
    distillation: PreparedTeacherInput


def _routed_assembly_shape(
    work: RoutedTeacherScoringWork,
    scored_partitions: tuple[ScoredDistillationBatch, ...],
) -> tuple[int, int]:
    if len(scored_partitions) != len(work.partitions):
        raise ValueError("scored teacher partitions do not match routed work")
    if not scored_partitions:
        raise ValueError("routed scoring requires at least one teacher partition")
    row_count = len(work.trajectory_ids)
    aligned_metadata = {"routes": len(work.routes), "response_lengths": len(work.response_lengths)}
    if any(length != row_count for length in aligned_metadata.values()):
        raise ValueError(f"routed scoring metadata must align with {row_count} trajectories: {aligned_metadata}")
    original_indices = sorted(index for partition in work.partitions for index in partition.original_indices)
    if original_indices != list(range(row_count)):
        raise ValueError("routed scoring partitions must cover every original trajectory exactly once")
    return row_count, max(work.response_lengths)


def _assemble_student_selected_inputs(
    indexed_inputs: Sequence[tuple[tuple[int, ...], PreparedTeacherInput]],
    shape: tuple[int, int],
    valid_mask: torch.Tensor,
    loss_weights: torch.Tensor,
) -> StudentTopKInput:
    if not all(isinstance(distillation, StudentTopKInput) for _, distillation in indexed_inputs):
        raise ValueError("routed teacher partitions must return one distillation objective kind")
    payloads = tuple(cast(StudentTopKInput, distillation) for _, distillation in indexed_inputs)
    widths = {payload.student_topk_indices.shape[-1] for payload in payloads}
    if len(widths) != 1:
        raise ValueError("routed student-selected partitions must use one top-K width")
    topk = widths.pop()
    indices = torch.full((*shape, topk), INVALID_TOPK_INDEX, dtype=torch.long)
    behavior = torch.full((*shape, topk), torch.nan, dtype=torch.float32)
    teacher = torch.full((*shape, topk), torch.nan, dtype=torch.float32)
    for (original_indices, _), payload in zip(indexed_inputs, payloads, strict=True):
        rows = torch.tensor(original_indices, dtype=torch.long)
        width = payload.valid_mask.shape[1]
        indices[rows, :width] = payload.student_topk_indices
        behavior[rows, :width] = payload.behavior_topk_logprobs
        teacher[rows, :width] = payload.teacher_on_student_logprobs
    return StudentTopKInput(indices, behavior, teacher, valid_mask, loss_weights)


def assemble_distillation_inputs(
    indexed_inputs: Sequence[tuple[tuple[int, ...], PreparedTeacherInput]],
    shape: tuple[int, int],
) -> PreparedTeacherInput:
    """Scatter homogeneous learner inputs into an explicitly sized batch."""
    if not indexed_inputs:
        raise ValueError("distillation assembly requires at least one input")
    valid_mask = torch.zeros(shape, dtype=torch.bool)
    loss_weights = torch.zeros(shape, dtype=torch.float32)
    first = indexed_inputs[0][1]
    for original_indices, distillation in indexed_inputs:
        width = distillation.valid_mask.shape[1]
        if width > shape[1]:
            raise ValueError("teacher evidence is wider than the destination response batch")
        indices = torch.tensor(original_indices, dtype=torch.long)
        if len(indices) != distillation.valid_mask.shape[0]:
            raise ValueError("distillation row indices must match their learner input")
        valid_mask[indices, :width] = distillation.valid_mask
        loss_weights[indices, :width] = distillation.loss_weights

    if isinstance(first, ChosenTokenTeacherInput):
        if not all(isinstance(distillation, ChosenTokenTeacherInput) for _, distillation in indexed_inputs):
            raise ValueError("routed teacher partitions must return one distillation objective kind")
        teacher_logprobs = torch.full(shape, torch.nan, dtype=torch.float32)
        for original_indices, distillation in indexed_inputs:
            payload = cast(ChosenTokenTeacherInput, distillation)
            indices = torch.tensor(original_indices, dtype=torch.long)
            teacher_logprobs[indices, : payload.valid_mask.shape[1]] = payload.teacher_action_log_probs
        return ChosenTokenTeacherInput(teacher_logprobs, valid_mask, loss_weights)

    if isinstance(first, StudentTopKInput):
        return _assemble_student_selected_inputs(indexed_inputs, shape, valid_mask, loss_weights)

    if not isinstance(first, TeacherTopKInput) or not all(
        isinstance(distillation, TeacherTopKInput) for _, distillation in indexed_inputs
    ):
        raise ValueError("routed teacher partitions must return one distillation objective kind")
    payloads = tuple(cast(TeacherTopKInput, distillation) for _, distillation in indexed_inputs)
    topk_values = {payload.teacher_topk_indices.shape[-1] for payload in payloads}
    if len(topk_values) != 1:
        raise ValueError("routed sparse teacher partitions must use one top-K width")
    topk = topk_values.pop()
    teacher_indices = torch.full((*shape, topk), INVALID_TOPK_INDEX, dtype=torch.long)
    teacher_logprobs = torch.full((*shape, topk), torch.nan, dtype=torch.float32)
    retained_mass = torch.full(shape, torch.nan, dtype=torch.float32)
    for (original_indices, _), payload in zip(indexed_inputs, payloads, strict=True):
        indices = torch.tensor(original_indices, dtype=torch.long)
        width = payload.valid_mask.shape[1]
        teacher_indices[indices, :width] = payload.teacher_topk_indices
        teacher_logprobs[indices, :width] = payload.teacher_topk_logprobs
        retained_mass[indices, :width] = payload.retained_mass
    return TeacherTopKInput(teacher_indices, teacher_logprobs, retained_mass, valid_mask, loss_weights)


def _pad_token_rows(token_rows: list[list[int]]) -> tuple[torch.Tensor, torch.Tensor]:
    rows = [torch.tensor(tokens, dtype=torch.long) for tokens in token_rows]
    padded = pad_sequence(rows, batch_first=True, padding_value=0)
    mask = torch.arange(padded.shape[1]).unsqueeze(0) < torch.tensor([len(tokens) for tokens in token_rows]).unsqueeze(
        1
    )
    return padded, mask


def collate_student_selected_rollout(
    trajectory_batch: TrajectoryBatch,
    response_token_ids: list[list[int]],
    response_mask: torch.Tensor,
    top_k: int | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    index_rows = trajectory_batch.get("student_topk_indices")
    behavior_rows = trajectory_batch.get("behavior_topk_logprobs")
    loss_masks = trajectory_batch.get("loss_masks")
    if index_rows is None or behavior_rows is None or loss_masks is None:
        raise ValueError("student-selected scoring requires rollout top-K indices, behavior logprobs, and loss masks")
    batch_size = len(response_token_ids)
    if len(index_rows) != batch_size or len(behavior_rows) != batch_size or len(loss_masks) != batch_size:
        raise ValueError("student-selected rollout fields must align with trajectories")
    if any(len(mask) != len(response) for mask, response in zip(loss_masks, response_token_ids, strict=True)):
        raise ValueError("student-selected loss masks must align with response tokens")
    if any(value not in (0, 1) for mask in loss_masks for value in mask):
        raise ValueError("student-selected loss masks must contain only 0 or 1")
    if top_k is None or top_k <= 0:
        raise ValueError("student-selected scoring requires a positive top_k")
    padded_loss_masks, _ = _pad_token_rows(loss_masks)
    if padded_loss_masks.shape != response_mask.shape:
        raise ValueError("student-selected loss masks must align with response tokens")
    selected_mask = padded_loss_masks.to(torch.bool)
    indices = torch.full((*response_mask.shape, top_k), INVALID_TOPK_INDEX, dtype=torch.long)
    behavior = torch.full((*response_mask.shape, top_k), torch.nan, dtype=torch.float32)
    for row, (selected, scores, response) in enumerate(zip(index_rows, behavior_rows, response_token_ids, strict=True)):
        if len(selected) != len(response) or len(scores) != len(response):
            raise ValueError("student-selected rollout fields must align with response tokens")
        indices[row, : len(response)] = torch.from_numpy(selected)
        behavior[row, : len(response)] = torch.from_numpy(scores)
    indices.masked_fill_(~selected_mask.unsqueeze(-1), INVALID_TOPK_INDEX)
    behavior.masked_fill_(~selected_mask.unsqueeze(-1), torch.nan)
    return indices, behavior, selected_mask


def build_teacher_scoring_work(
    trajectory_batch: TrajectoryBatch,
    *,
    route_ids: tuple[str, ...],
    teacher_id: str,
    tokenizer_fingerprint: str,
    plan_version: str,
    coefficient: float,
    route_weights: tuple[float, ...],
    evidence: TeacherEvidenceKind = TeacherEvidenceKind.CHOSEN_TOKEN,
    top_k: int | None = None,
) -> TeacherScoringWork:
    """Collate exact admitted rollout tokens into the shared oracle contract."""
    trajectory_ids = trajectory_batch.get("trajectory_ids")
    if trajectory_ids is None:
        raise ValueError("teacher scoring requires stable trajectory_ids")
    prompt_token_ids = trajectory_batch["prompt_token_ids"]
    response_token_ids = trajectory_batch["response_ids"]
    batch_size = len(trajectory_ids)
    if batch_size == 0:
        raise ValueError("teacher scoring requires at least one trajectory")
    aligned_fields = {
        "prompt_token_ids": len(prompt_token_ids),
        "response_ids": len(response_token_ids),
        "route_ids": len(route_ids),
        "route_weights": len(route_weights),
    }
    if any(length != batch_size for length in aligned_fields.values()):
        raise ValueError(f"teacher scoring fields must align with {batch_size} trajectories: {aligned_fields}")

    padded_prompts, prompt_mask = _pad_token_rows(prompt_token_ids)
    padded_responses, response_mask = _pad_token_rows(response_token_ids)
    per_trajectory_weights = torch.tensor(route_weights, dtype=torch.float32).unsqueeze(1)
    loss_weights = per_trajectory_weights.expand_as(padded_responses).masked_fill(~response_mask, 0)

    selected_indices = None
    behavior_logprobs = None
    selected_mask = None
    if evidence is TeacherEvidenceKind.STUDENT_SELECTED_TOPK:
        selected_indices, behavior_logprobs, selected_mask = collate_student_selected_rollout(
            trajectory_batch, response_token_ids, response_mask, top_k
        )

    request = TeacherScoreRequest(
        trajectory_ids=tuple(trajectory_id.to_string() for trajectory_id in trajectory_ids),
        route_ids=route_ids,
        teacher_id=teacher_id,
        tokenizer_fingerprint=tokenizer_fingerprint,
        plan_version=plan_version,
        prompt_token_ids=padded_prompts,
        prompt_mask=prompt_mask,
        response_token_ids=padded_responses,
        response_mask=response_mask,
        evidence=evidence,
        top_k=top_k,
        student_topk_indices=selected_indices,
        behavior_topk_logprobs=behavior_logprobs,
        student_selected_mask=selected_mask,
    )
    return TeacherScoringWork(request=request, coefficient=coefficient, route_weights=loss_weights)


def build_routed_teacher_scoring_work(
    routed_batch: RoutedTrajectoryBatch,
    *,
    tokenizer_fingerprints: Mapping[str, str],
    top_k_by_teacher: Mapping[str, int] | None = None,
) -> RoutedTeacherScoringWork:
    """Build one exact request per logical-teacher partition."""
    missing_fingerprints = sorted(
        {partition.teacher_id for partition in routed_batch.partitions} - tokenizer_fingerprints.keys()
    )
    if missing_fingerprints:
        raise ValueError(f"missing tokenizer fingerprints for teachers: {', '.join(missing_fingerprints)}")
    top_k_by_teacher = top_k_by_teacher or {}
    if routed_batch.evidence in {TeacherEvidenceKind.TOPK_DISTRIBUTION, TeacherEvidenceKind.STUDENT_SELECTED_TOPK}:
        missing_top_k = sorted(
            {partition.teacher_id for partition in routed_batch.partitions} - top_k_by_teacher.keys()
        )
        if missing_top_k:
            raise ValueError(f"missing top_k for teachers: {', '.join(missing_top_k)}")

    partitions = tuple(
        RoutedTeacherScoringPartition(
            original_indices=partition.original_indices,
            work=build_teacher_scoring_work(
                partition.trajectory_batch,
                route_ids=tuple(route.route_id for route in partition.routes),
                teacher_id=partition.teacher_id,
                tokenizer_fingerprint=tokenizer_fingerprints[partition.teacher_id],
                plan_version=routed_batch.plan_version,
                coefficient=routed_batch.coefficient,
                route_weights=tuple(route.weight for route in partition.routes),
                evidence=routed_batch.evidence,
                top_k=top_k_by_teacher.get(partition.teacher_id),
            ),
        )
        for partition in routed_batch.partitions
    )
    return RoutedTeacherScoringWork(
        trajectory_ids=routed_batch.trajectory_ids,
        routes=routed_batch.routes,
        response_lengths=routed_batch.response_lengths,
        plan_version=routed_batch.plan_version,
        partitions=partitions,
    )


class TeacherEvidenceCoordinator:
    """Score teacher work and prepare the matching learner objective input."""

    def __init__(self, oracle_owner: TeacherOracleCollection) -> None:
        self._oracle_owner = oracle_owner

    async def score(self, work: TeacherScoringWork) -> ScoredDistillationBatch:
        evidence = await self._oracle_owner.score(work.request.teacher_id, work.request)
        if isinstance(evidence, ChosenTokenTeacherEvidence):
            distillation = prepare_sampled_reverse_kl(
                work.request,
                evidence,
                coefficient=work.coefficient,
                route_weights=work.route_weights,
            )
        elif isinstance(evidence, TopKTeacherEvidence):
            distillation = prepare_teacher_topk(
                work.request,
                evidence,
                coefficient=work.coefficient,
                route_weights=work.route_weights,
            )
        elif isinstance(evidence, StudentSelectedTeacherEvidence):
            distillation = prepare_student_topk_policy_surrogate(
                work.request,
                evidence,
                coefficient=work.coefficient,
                route_weights=work.route_weights,
            )
        else:
            raise TypeError(f"unsupported teacher evidence type: {type(evidence).__name__}")
        return ScoredDistillationBatch(evidence=evidence, distillation=distillation)

    @staticmethod
    def assemble_routed(
        work: RoutedTeacherScoringWork,
        scored_partitions: tuple[ScoredDistillationBatch, ...],
    ) -> RoutedScoredDistillationBatch:
        """Restore independently scored partitions to the original mixed-batch coordinates."""
        shape = _routed_assembly_shape(work, scored_partitions)
        teacher_revisions = [""] * shape[0]
        indexed_inputs = []
        for partition, scored in zip(work.partitions, scored_partitions, strict=True):
            if scored.evidence.trajectory_ids != partition.work.request.trajectory_ids:
                raise ValueError("scored teacher partition does not match its routed trajectories")
            indexed_inputs.append((partition.original_indices, scored.distillation))
            for index in partition.original_indices:
                teacher_revisions[index] = scored.evidence.teacher_revision
        distillation = assemble_distillation_inputs(indexed_inputs, shape)
        return RoutedScoredDistillationBatch(
            trajectory_ids=work.trajectory_ids,
            routes=work.routes,
            teacher_revisions=tuple(teacher_revisions),
            plan_version=work.plan_version,
            evidence=tuple(scored.evidence for scored in scored_partitions),
            distillation=distillation,
        )

    async def close(self) -> None:
        await self._oracle_owner.close()


@dataclass(frozen=True)
class AsyncTeacherScoreTicket:
    """A submitted score whose result gates learner batch assembly."""

    _result: asyncio.Future[ScoredDistillationBatch]

    async def result(self) -> ScoredDistillationBatch:
        return await self._result


@dataclass(frozen=True)
class AsyncRoutedTeacherScoreTicket:
    """Per-teacher queue tickets reassembled only when a mixed group is ready."""

    _work: RoutedTeacherScoringWork
    _partitions: tuple[AsyncTeacherScoreTicket, ...]
    _coordinator: TeacherEvidenceCoordinator

    async def result(self) -> RoutedScoredDistillationBatch:
        scored = await asyncio.gather(*(partition.result() for partition in self._partitions))
        return self._coordinator.assemble_routed(self._work, tuple(scored))


@dataclass(frozen=True)
class _QueuedTeacherScore:
    work: TeacherScoringWork
    result: asyncio.Future[ScoredDistillationBatch]


@dataclass(frozen=True)
class AsyncTeacherQueueLimits:
    """Bound queued and active score requests for one logical teacher."""

    max_queued: int
    workers: int

    def __post_init__(self) -> None:
        if (
            isinstance(self.max_queued, bool)
            or not isinstance(self.max_queued, int)
            or self.max_queued <= 0
            or isinstance(self.workers, bool)
            or not isinstance(self.workers, int)
            or self.workers <= 0
        ):
            raise ValueError("teacher queue and worker limits must be positive")


class AdmittedGroupDistillationAdapter:
    """Bound teacher work per logical teacher between rollout admission and learner batch assembly."""

    def __init__(
        self,
        coordinator: TeacherEvidenceCoordinator,
        *,
        teacher_limits: Mapping[str, AsyncTeacherQueueLimits],
    ) -> None:
        if not teacher_limits:
            raise ValueError("distillation requires at least one teacher queue")
        self._coordinator = coordinator
        self._queues = {
            teacher_id: asyncio.Queue[_QueuedTeacherScore](maxsize=limits.max_queued)
            for teacher_id, limits in teacher_limits.items()
        }
        self._worker_counts = {teacher_id: limits.workers for teacher_id, limits in teacher_limits.items()}
        self._workers: list[asyncio.Task[None]] = []
        self._accepting = False
        self._closed = False
        self._submission_locks = {teacher_id: asyncio.Lock() for teacher_id in teacher_limits}

    async def start(self) -> None:
        if self._closed:
            raise RuntimeError("distillation adapter is closed")
        if self._workers:
            return
        self._accepting = True
        self._workers = [
            asyncio.create_task(self._score_worker(queue))
            for teacher_id, queue in self._queues.items()
            for _ in range(self._worker_counts[teacher_id])
        ]

    async def submit_before_batch_assembly(self, work: TeacherScoringWork) -> AsyncTeacherScoreTicket:
        """Apply queue backpressure while admitting a generated group for scoring."""
        try:
            queue = self._queues[work.request.teacher_id]
            submission_lock = self._submission_locks[work.request.teacher_id]
        except KeyError as error:
            raise ValueError(f"no scoring queue configured for teacher {work.request.teacher_id!r}") from error
        async with submission_lock:
            if not self._accepting:
                raise RuntimeError("distillation adapter is not accepting work")
            result = asyncio.get_running_loop().create_future()
            await queue.put(_QueuedTeacherScore(work=work, result=result))
        return AsyncTeacherScoreTicket(result)

    async def score_before_batch_assembly(self, work: TeacherScoringWork) -> ScoredDistillationBatch:
        """Return only once evidence is ready to travel with a generated group."""
        ticket = await self.submit_before_batch_assembly(work)
        return await ticket.result()

    async def submit_routed_before_batch_assembly(
        self,
        work: RoutedTeacherScoringWork,
    ) -> AsyncRoutedTeacherScoreTicket:
        """Submit every logical teacher concurrently so one full queue cannot head-of-line block another."""
        tickets = await _gather_or_cancel(
            [asyncio.create_task(self.submit_before_batch_assembly(partition.work)) for partition in work.partitions]
        )
        return AsyncRoutedTeacherScoreTicket(work, tuple(tickets), self._coordinator)

    async def score_routed_before_batch_assembly(
        self,
        work: RoutedTeacherScoringWork,
    ) -> RoutedScoredDistillationBatch:
        """Score and reassemble a mixed-domain group before learner-batch assembly."""
        ticket = await self.submit_routed_before_batch_assembly(work)
        return await ticket.result()

    async def close(self) -> None:
        if self._closed:
            return
        self._accepting = False
        self._closed = True
        for submission_lock in self._submission_locks.values():
            async with submission_lock:
                pass
        if not self._workers:
            await self._coordinator.close()
            return
        await asyncio.gather(*(queue.join() for queue in self._queues.values()))
        for worker in self._workers:
            worker.cancel()
        await asyncio.gather(*self._workers, return_exceptions=True)
        self._workers.clear()
        await self._coordinator.close()

    async def __aenter__(self) -> AdmittedGroupDistillationAdapter:
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_value, traceback) -> None:
        await self.close()

    async def _score_worker(
        self,
        queue: asyncio.Queue[_QueuedTeacherScore],
    ) -> None:
        while True:
            queued = await queue.get()
            try:
                scored = await self._coordinator.score(queued.work)
            except Exception as error:
                if not queued.result.done():
                    queued.result.set_exception(error)
            else:
                if not queued.result.done():
                    queued.result.set_result(scored)
            finally:
                queue.task_done()
