# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Derive a distributed routed-expert reverse plan from generic forward semantics."""

from dataclasses import dataclass
from enum import StrEnum

from tile_lifetime.autodiff import scalar_expression_vjp
from tile_lifetime.expert_parallel_plan import ExpertParallelPlan, TransportSemantics
from tile_lifetime.tensor_program import ScalarExpression


class ExpertParallelTrainingStageKind(StrEnum):
    """Generic algebra and placement stages in a routed training boundary."""

    ROUTER_CONTRACT = "router_contract"
    TOP_K_SELECTION = "top_k_selection"
    NORMALIZED_ROUTE_WEIGHT_FOLD = "normalized_route_weight_fold"
    FORWARD_EXPERT_PROGRAM = "forward_expert_program"
    OUTPUT_ADJOINT_EDGE_MAP = "output_adjoint_edge_map"
    OUTPUT_ADJOINT_TRANSPORT = "output_adjoint_transport"
    DOWN_INPUT_ADJOINT = "down_input_adjoint_segmented_contract"
    DOWN_WEIGHT_ADJOINT = "down_weight_adjoint_segmented_contract"
    PAIR_MAP_ADJOINT = "pair_map_adjoint"
    GATE_UP_INPUT_ADJOINT = "gate_up_input_adjoint_segmented_contract"
    GATE_UP_WEIGHT_ADJOINT = "gate_up_weight_adjoint_segmented_contract"
    ROUTE_WEIGHT_ADJOINT = "route_weight_adjoint_fold"
    INPUT_ADJOINT_RETURN_TRANSPORT = "input_adjoint_return_transport"
    ROUTE_WEIGHT_RETURN_TRANSPORT = "route_weight_return_transport"
    SOURCE_INPUT_ADJOINT_FOLD = "source_input_adjoint_fold"
    ROUTER_VJP = "normalized_route_weight_and_router_contract_vjp"
    SHARED_EXPERT_REVERSE = "shared_expert_reverse_contract_map_fold"
    SOURCE_INPUT_ADJOINT_MAP = "source_input_adjoint_map"


@dataclass(frozen=True)
class ExpertParallelTrainingStage:
    """One inspectable stage with no physical workload-kernel identity."""

    kind: ExpertParallelTrainingStageKind
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    primitive: str


@dataclass(frozen=True)
class ExpertParallelTrainingPlan:
    """Matched natural-program boundary for distributed forward and reverse."""

    forward: ExpertParallelPlan
    stages: tuple[ExpertParallelTrainingStage, ...]
    pair_map_left_vjp: ScalarExpression
    pair_map_right_vjp: ScalarExpression
    saved_forward_values: tuple[str, ...]
    payload_transports: tuple[str, ...]
    external_implementation_boundaries: tuple[str, ...]

    def stage(self, kind: ExpertParallelTrainingStageKind) -> ExpertParallelTrainingStage:
        """Return the unique stage of one kind."""
        matches = tuple(stage for stage in self.stages if stage.kind is kind)
        if len(matches) != 1:
            raise KeyError(f"expected one {kind.value} stage, found {len(matches)}")
        return matches[0]


def derive_expert_parallel_training_plan(forward: ExpertParallelPlan) -> ExpertParallelTrainingPlan:
    """Derive reverse algebra and placement without selecting a fused MoE kernel."""
    if forward.schedule.forward_transport.semantics is not TransportSemantics.PAYLOAD_PERMUTATION:
        raise ValueError("forward transport must not contain semantic accumulation")
    if forward.schedule.reverse_transport.semantics is not TransportSemantics.PAYLOAD_PERMUTATION:
        raise ValueError("reverse transport must not contain semantic accumulation")

    pair_map = forward.map_fold_semantics.pair_map
    left_vjp = scalar_expression_vjp(pair_map, input_name="left", cotangent_name="cotangent")
    right_vjp = scalar_expression_vjp(pair_map, input_name="right", cotangent_name="cotangent")
    route_name = forward.route_relation.name
    stages = (
        _stage(
            ExpertParallelTrainingStageKind.ROUTER_CONTRACT, ("tokens", "router.weight"), ("router.logits",), "Contract"
        ),
        _stage(
            ExpertParallelTrainingStageKind.TOP_K_SELECTION, ("router.logits",), (route_name,), "Selection -> Relation"
        ),
        _stage(
            ExpertParallelTrainingStageKind.NORMALIZED_ROUTE_WEIGHT_FOLD,
            (route_name, "router.logits"),
            ("route.weights",),
            "Map + Fold(max,sum_exp) + Map",
        ),
        _stage(
            ExpertParallelTrainingStageKind.FORWARD_EXPERT_PROGRAM,
            ("tokens", route_name, "route.weights", "expert.weights"),
            ("output", "saved.edge_output", "saved.pair_input", "saved.hidden"),
            "RelationPlan + SegmentedContract + Map + SegmentedContract + Fold",
        ),
        _stage(
            ExpertParallelTrainingStageKind.OUTPUT_ADJOINT_EDGE_MAP,
            ("output.cotangent", "route.weights"),
            ("edge_output.cotangent",),
            "Map",
        ),
        _stage(
            ExpertParallelTrainingStageKind.OUTPUT_ADJOINT_TRANSPORT,
            ("edge_output.cotangent", route_name),
            ("expert.edge_output.cotangent",),
            "Transport(payload_permutation)",
        ),
        _stage(
            ExpertParallelTrainingStageKind.DOWN_INPUT_ADJOINT,
            ("expert.edge_output.cotangent", "expert.down_weight"),
            ("hidden.cotangent",),
            "SegmentedContract",
        ),
        _stage(
            ExpertParallelTrainingStageKind.DOWN_WEIGHT_ADJOINT,
            ("saved.hidden", "expert.edge_output.cotangent"),
            ("expert.down_weight.cotangent",),
            "SegmentedContract(reduction_domain=edge_rows)",
        ),
        _stage(
            ExpertParallelTrainingStageKind.PAIR_MAP_ADJOINT,
            ("saved.pair_input", "hidden.cotangent"),
            ("gate.cotangent", "up.cotangent"),
            "generated Map VJP",
        ),
        _stage(
            ExpertParallelTrainingStageKind.GATE_UP_INPUT_ADJOINT,
            ("gate.cotangent", "up.cotangent", "expert.gate_up_weight"),
            ("expert.edge_input.cotangent",),
            "SegmentedContract",
        ),
        _stage(
            ExpertParallelTrainingStageKind.GATE_UP_WEIGHT_ADJOINT,
            ("expert.edge_input", "gate.cotangent", "up.cotangent"),
            ("expert.gate_up_weight.cotangent",),
            "SegmentedContract(reduction_domain=edge_rows)",
        ),
        _stage(
            ExpertParallelTrainingStageKind.ROUTE_WEIGHT_ADJOINT,
            ("saved.edge_output", "output.cotangent"),
            ("route.weights.cotangent",),
            "Map(multiply) + Fold(sum over feature)",
        ),
        _stage(
            ExpertParallelTrainingStageKind.INPUT_ADJOINT_RETURN_TRANSPORT,
            ("expert.edge_input.cotangent", route_name),
            ("source.edge_input.cotangent",),
            "Transport(payload_permutation)",
        ),
        _stage(
            ExpertParallelTrainingStageKind.ROUTE_WEIGHT_RETURN_TRANSPORT,
            ("route.weights.cotangent", route_name),
            ("source.route_weight.cotangent",),
            "Transport(payload_permutation)",
        ),
        _stage(
            ExpertParallelTrainingStageKind.SOURCE_INPUT_ADJOINT_FOLD,
            ("source.edge_input.cotangent", route_name),
            ("routed_input.cotangent",),
            "deterministic source-slot Fold",
        ),
        _stage(
            ExpertParallelTrainingStageKind.ROUTER_VJP,
            ("source.route_weight.cotangent", "router.logits", "tokens", "router.weight"),
            ("router_input.cotangent", "router.weight.cotangent"),
            "JAX-owned Map/Fold/Contract reverse",
        ),
        _stage(
            ExpertParallelTrainingStageKind.SHARED_EXPERT_REVERSE,
            ("output.cotangent", "tokens", "shared.weights", "saved.shared"),
            ("shared_input.cotangent", "shared.weight.cotangents"),
            "Contract + generated Map VJP + Contract + weight Contracts",
        ),
        _stage(
            ExpertParallelTrainingStageKind.SOURCE_INPUT_ADJOINT_MAP,
            ("routed_input.cotangent", "router_input.cotangent", "shared_input.cotangent"),
            ("tokens.cotangent",),
            "Map(add)",
        ),
    )
    return ExpertParallelTrainingPlan(
        forward=forward,
        stages=stages,
        pair_map_left_vjp=left_vjp,
        pair_map_right_vjp=right_vjp,
        saved_forward_values=(
            "expert.edge_input",
            "saved.pair_input",
            "saved.hidden",
            "saved.edge_output",
            "saved.shared",
        ),
        payload_transports=(
            forward.schedule.forward_transport.implementation,
            forward.schedule.reverse_transport.implementation,
            "output_adjoint_to_expert_owner",
            "input_adjoint_to_source_owner",
            "route_weight_adjoint_to_source_owner",
        ),
        external_implementation_boundaries=(
            "generic grouped/ragged Contract mainloop",
            "generic payload transport",
            "JAX-owned automatic differentiation and router VJP",
            "expert implementation used only as forward/backward oracle",
        ),
    )


def _stage(
    kind: ExpertParallelTrainingStageKind,
    inputs: tuple[str, ...],
    outputs: tuple[str, ...],
    primitive: str,
) -> ExpertParallelTrainingStage:
    return ExpertParallelTrainingStage(kind=kind, inputs=inputs, outputs=outputs, primitive=primitive)
