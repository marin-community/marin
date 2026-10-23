"""Trainer-facing orchestration for configured distillation."""

from __future__ import annotations

from collections.abc import Sequence

from marinskyrl.distillation import DistillationPlan
from skyrl_train.distillation_adapters import (
    AdmittedGroupDistillationAdapter,
    AsyncRoutedTeacherScoreTicket,
    AsyncTeacherQueueLimits,
    RoutedScoredDistillationBatch,
    RoutedTeacherScoringWork,
    TeacherEvidenceCoordinator,
    assemble_distillation_inputs,
    build_routed_teacher_scoring_work,
)
from skyrl_train.teacher_oracle import TeacherOracleCollection
from skyrl_train.teacher_routing import PlanTeacherRouter, route_trajectory_batch
from skyrl_train.domain_gradient_balance import DomainGradientBalancer
from skyrl_train.training_batch import TrainingInputBatch
from skyrl_train.trajectory_runners.types import TrajectoryBatch


class _RoutedDistillationPlanner:
    """Compile trajectory batches into backend-neutral, routed teacher work."""

    def __init__(self, plan: DistillationPlan, *, tokenizer_fingerprints: dict[str, str]) -> None:
        self._router = PlanTeacherRouter(plan)
        self._tokenizer_fingerprints = dict(tokenizer_fingerprints)
        self._top_k_by_teacher = {teacher.id: teacher.top_k for teacher in plan.teachers if teacher.top_k is not None}
        self._default_route_key = plan.routing.routes[0].key if len(plan.routing.routes) == 1 else None

    def build_work(self, trajectory_batch: TrajectoryBatch) -> RoutedTeacherScoringWork:
        route_keys = trajectory_batch.get("teacher_route_keys")
        if route_keys is None:
            if self._default_route_key is None:
                raise ValueError("a multi-route distillation plan requires teacher_route_keys on the trajectory batch")
            route_keys = [self._default_route_key] * len(trajectory_batch["response_ids"])
        routed = route_trajectory_batch(
            trajectory_batch,
            route_keys=tuple(route_keys),
            router=self._router,
        )
        return build_routed_teacher_scoring_work(
            routed,
            tokenizer_fingerprints=self._tokenizer_fingerprints,
            top_k_by_teacher=self._top_k_by_teacher,
        )


class DistillationRuntime:
    """Score admitted rollout groups before learner batch assembly.

    Each group is submitted when the rollout buffer admits it, so teacher scoring overlaps generation. The
    trainer attaches the scored groups, in learner row order, once it assembles the training input.
    """

    def __init__(
        self,
        plan: DistillationPlan,
        oracles: TeacherOracleCollection,
        *,
        tokenizer_fingerprints: dict[str, str],
        teacher_limits: dict[str, AsyncTeacherQueueLimits],
    ) -> None:
        self._planner = _RoutedDistillationPlanner(plan, tokenizer_fingerprints=tokenizer_fingerprints)
        self._adapter = AdmittedGroupDistillationAdapter(
            coordinator=TeacherEvidenceCoordinator(oracles),
            teacher_limits=teacher_limits,
        )
        self.domain_balancer = (
            DomainGradientBalancer(plan.domain_gradient_balance) if plan.domain_gradient_balance is not None else None
        )

    async def start(self) -> None:
        await self._adapter.start()

    async def submit_before_batch_assembly(
        self,
        trajectory_batch: TrajectoryBatch,
    ) -> AsyncRoutedTeacherScoreTicket:
        """Route and enqueue one admitted, optimization-selected rollout group."""
        work = self._planner.build_work(trajectory_batch)
        return await self._adapter.submit_routed_before_batch_assembly(work)

    def attach_to_training_input(
        self,
        training_input: TrainingInputBatch,
        scored_groups: Sequence[RoutedScoredDistillationBatch],
    ) -> dict[str, float]:
        """Attach scored groups and return per-domain balance metrics, if enabled."""
        if not scored_groups:
            raise ValueError("distillation requires at least one scored group")
        response_shape = training_input["response_mask"].shape
        row_count = sum(len(scored.trajectory_ids) for scored in scored_groups)
        if "pad_size" not in training_input.metadata:
            raise ValueError("learner batch is missing required pad_size metadata")
        pad_size = int(training_input.metadata["pad_size"])
        if row_count + pad_size != response_shape[0]:
            raise ValueError(
                "scored distillation rows must match the unpadded learner batch: "
                f"scored={row_count}, padding={pad_size}, learner={response_shape[0]}"
            )
        indexed_inputs = []
        offset = 0
        for scored in scored_groups:
            next_offset = offset + len(scored.trajectory_ids)
            indexed_inputs.append((tuple(range(offset, next_offset)), scored.distillation))
            offset = next_offset
        assembled = assemble_distillation_inputs(indexed_inputs, tuple(response_shape))
        balance_metrics: dict[str, float] = {}
        if self.domain_balancer is not None:
            route_ids = tuple(route.route_id for scored in scored_groups for route in scored.routes)
            assembled, balance_metrics = self.domain_balancer.apply(assembled, route_ids)
        training_input.update(assembled.training_tensors())
        training_input.metadata.update(_distillation_provenance(scored_groups))
        return balance_metrics

    async def close(self) -> None:
        await self._adapter.close()


def _distillation_provenance(
    scored_groups: Sequence[RoutedScoredDistillationBatch],
) -> dict[str, tuple[str, ...]]:
    """Return row-aligned route, teacher, and plan revisions for learner diagnostics."""
    return {
        "distillation_plan_versions": tuple(
            scored.plan_version for scored in scored_groups for _ in scored.trajectory_ids
        ),
        "distillation_teacher_revisions": tuple(
            revision for scored in scored_groups for revision in scored.teacher_revisions
        ),
        "distillation_route_ids": tuple(route.route_id for scored in scored_groups for route in scored.routes),
    }
