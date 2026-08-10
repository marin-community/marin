# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Reference execution and ABI checks for distributed segmented reverse plans."""

from dataclasses import dataclass

import numpy as np

from tile_lifetime.expert_parallel_training import ExpertParallelTrainingPlan
from tile_lifetime.ir import DType
from tile_lifetime.relation import RelationPlan
from tile_lifetime.tensor_program import ScalarExpression, ScalarExpressionKind


@dataclass(frozen=True)
class BackwardBufferContract:
    """One bounded buffer in a rank-local segmented reverse schedule."""

    name: str
    shape: tuple[int, ...]
    dtype: DType
    producer: str
    consumer: str
    zero_initialized: bool = False

    @property
    def bytes(self) -> int:
        """Return the statically required capacity."""
        item_size = {
            DType.BOOL: 1,
            DType.BF16: 2,
            DType.FP32: 4,
            DType.FP64: 8,
            DType.INT32: 4,
        }[self.dtype]
        return int(np.prod(self.shape, dtype=np.int64)) * item_size


@dataclass(frozen=True)
class ExpertBackwardRankABI:
    """Rank-local row domains and buffers derived from a global RelationPlan."""

    rank: int
    local_expert_count: int
    valid_destination_rows: np.ndarray
    padded_destination_rows: np.ndarray
    buffers: tuple[BackwardBufferContract, ...]

    @property
    def total_buffer_bytes(self) -> int:
        """Return total logical capacity, including explicit output buffers."""
        return sum(buffer.bytes for buffer in self.buffers)


@dataclass(frozen=True)
class DistributedExpertBackwardABI:
    """Four-rank compatible payload and buffer contract for one reverse plan."""

    ranks: tuple[ExpertBackwardRankABI, ...]
    source_item_count: int
    route_slots: int
    hidden: int
    intermediate: int
    global_expert_count: int
    transport_semantics: str = "payload_permutation_only"
    source_fold_order: str = "source item ascending, then route slot ascending"


@dataclass(frozen=True)
class DistributedExpertBackwardResult:
    """Reference outputs from generic relation, Contract, Map, and Fold stages."""

    output: np.ndarray
    input_cotangent: np.ndarray
    route_weight_cotangent: np.ndarray
    gate_up_weight_cotangent: np.ndarray
    down_weight_cotangent: np.ndarray


def derive_distributed_expert_backward_abi(
    relation: RelationPlan,
    *,
    hidden: int,
    intermediate: int,
) -> DistributedExpertBackwardABI:
    """Derive all rank-local buffer shapes without allocating payload storage."""
    if hidden <= 0 or intermediate <= 0:
        raise ValueError("hidden and intermediate dimensions must be positive")
    local_expert_count, remainder = divmod(relation.destination_count, relation.destination_rank_count)
    if remainder:
        raise ValueError("destination groups must be evenly partitioned across ranks")

    ranks = tuple(
        _derive_rank_abi(
            relation,
            rank=rank,
            local_expert_count=local_expert_count,
            hidden=hidden,
            intermediate=intermediate,
        )
        for rank in range(relation.destination_rank_count)
    )
    abi = DistributedExpertBackwardABI(
        ranks=ranks,
        source_item_count=relation.source_item_count,
        route_slots=relation.route_slots,
        hidden=hidden,
        intermediate=intermediate,
        global_expert_count=relation.destination_count,
    )
    verify_distributed_expert_backward_abi(relation, abi)
    return abi


def verify_distributed_expert_backward_abi(
    relation: RelationPlan,
    abi: DistributedExpertBackwardABI,
) -> None:
    """Reject missing rows, semantic transport, and inconsistent buffer domains."""
    if abi.transport_semantics != "payload_permutation_only":
        raise ValueError("distributed reverse transport must not perform a semantic combine")
    if len(abi.ranks) != relation.destination_rank_count:
        raise ValueError("ABI rank count disagrees with RelationPlan placement")
    valid_rows = np.flatnonzero(relation.row_valid)
    observed = np.concatenate(tuple(rank.valid_destination_rows for rank in abi.ranks))
    if not np.array_equal(np.sort(observed), valid_rows):
        raise ValueError("rank-local reverse rows do not cover each valid relation edge exactly once")
    for rank in abi.ranks:
        if not np.all(relation.row_destination_rank[rank.valid_destination_rows] == rank.rank):
            raise ValueError(f"rank {rank.rank} contains a destination row owned by another rank")
        names = tuple(buffer.name for buffer in rank.buffers)
        if len(set(names)) != len(names):
            raise ValueError(f"rank {rank.rank} has duplicate backward buffer names")
        if any(min(buffer.shape, default=1) < 0 for buffer in rank.buffers):
            raise ValueError(f"rank {rank.rank} has a negative buffer extent")


def execute_distributed_expert_backward_reference(
    relation: RelationPlan,
    source: np.ndarray,
    gate_up_weight: np.ndarray,
    down_weight: np.ndarray,
    output_cotangent: np.ndarray,
    training_plan: ExpertParallelTrainingPlan,
) -> DistributedExpertBackwardResult:
    """Execute a deterministic multi-rank reverse schedule on CPU.

    Transport is modeled only as the RelationPlan's gather/inverse-gather. All
    semantic multiplication and reduction remains in generated Map/Fold stages.
    """
    source = np.asarray(source, dtype=np.float32)
    gate_up_weight = np.asarray(gate_up_weight, dtype=np.float32)
    down_weight = np.asarray(down_weight, dtype=np.float32)
    output_cotangent = np.asarray(output_cotangent, dtype=np.float32)
    _validate_execution_shapes(relation, source, gate_up_weight, down_weight, output_cotangent)

    padded_source = relation.dispatch(source)
    pair_input = np.zeros((relation.destination_row_count, gate_up_weight.shape[1]), dtype=np.float32)
    hidden = np.zeros((relation.destination_row_count, down_weight.shape[2]), dtype=np.float32)
    edge_output = np.zeros((relation.destination_row_count, source.shape[1]), dtype=np.float32)

    for group, row_slice in _group_slices(relation):
        expert = int(relation.group_destination_item[group])
        pair_input[row_slice] = padded_source[row_slice] @ gate_up_weight[expert].T
        intermediate = hidden.shape[1]
        hidden[row_slice] = _evaluate_array_expression(
            training_plan.forward.map_fold_semantics.pair_map,
            {
                "left": pair_input[row_slice, :intermediate],
                "right": pair_input[row_slice, intermediate:],
            },
        )
        edge_output[row_slice] = hidden[row_slice] @ down_weight[expert].T

    output = relation.weighted_merge(edge_output)
    unweighted_output_cotangent = relation.dispatch(output_cotangent)
    weighted_output_cotangent = unweighted_output_cotangent * relation.row_weight[:, None]
    pair_cotangent = np.zeros_like(pair_input)
    edge_input_cotangent = np.zeros_like(padded_source)
    gate_up_weight_cotangent = np.zeros_like(gate_up_weight)
    down_weight_cotangent = np.zeros_like(down_weight)

    for group, row_slice in _group_slices(relation):
        expert = int(relation.group_destination_item[group])
        hidden_cotangent = weighted_output_cotangent[row_slice] @ down_weight[expert]
        intermediate = hidden.shape[1]
        pair_cotangent[row_slice, :intermediate] = _evaluate_array_expression(
            training_plan.pair_map_left_vjp,
            {
                "left": pair_input[row_slice, :intermediate],
                "right": pair_input[row_slice, intermediate:],
                "cotangent": hidden_cotangent,
            },
        )
        pair_cotangent[row_slice, intermediate:] = _evaluate_array_expression(
            training_plan.pair_map_right_vjp,
            {
                "left": pair_input[row_slice, :intermediate],
                "right": pair_input[row_slice, intermediate:],
                "cotangent": hidden_cotangent,
            },
        )
        edge_input_cotangent[row_slice] = pair_cotangent[row_slice] @ gate_up_weight[expert]
        down_weight_cotangent[expert] += weighted_output_cotangent[row_slice].T @ hidden[row_slice]
        gate_up_weight_cotangent[expert] += pair_cotangent[row_slice].T @ padded_source[row_slice]

    restored_input_cotangent = relation.inverse_dispatch(edge_input_cotangent)
    input_cotangent = np.zeros_like(source)
    for source_item in range(relation.source_item_count):
        for route_slot in range(relation.route_slots):
            if relation.edge_valid[source_item, route_slot]:
                input_cotangent[source_item] += restored_input_cotangent[source_item, route_slot]

    route_weight_cotangent_rows = np.asarray(
        np.sum(
            edge_output * unweighted_output_cotangent,
            axis=1,
            dtype=np.float32,
        ),
        dtype=np.float32,
    ).reshape(-1, 1)
    route_weight_cotangent = relation.inverse_dispatch(route_weight_cotangent_rows)[:, :, 0]
    return DistributedExpertBackwardResult(
        output=np.asarray(output, dtype=np.float32),
        input_cotangent=input_cotangent,
        route_weight_cotangent=route_weight_cotangent,
        gate_up_weight_cotangent=gate_up_weight_cotangent,
        down_weight_cotangent=down_weight_cotangent,
    )


def _derive_rank_abi(
    relation: RelationPlan,
    *,
    rank: int,
    local_expert_count: int,
    hidden: int,
    intermediate: int,
) -> ExpertBackwardRankABI:
    rank_rows = np.flatnonzero(relation.row_destination_rank == rank)
    valid_rows = rank_rows[relation.row_valid[rank_rows]]
    padded_rows = rank_rows[~relation.row_valid[rank_rows]]
    row_count = rank_rows.shape[0]
    valid_count = valid_rows.shape[0]
    buffers = (
        BackwardBufferContract("received_output_cotangent", (valid_count, hidden), DType.BF16, "Transport", "edge Map"),
        BackwardBufferContract(
            "padded_output_cotangent",
            (row_count, hidden),
            DType.BF16,
            "edge Map",
            "down input/weight Contracts",
            True,
        ),
        BackwardBufferContract(
            "hidden_cotangent", (row_count, intermediate), DType.BF16, "down Contract", "pair Map VJP"
        ),
        BackwardBufferContract(
            "pair_cotangent", (row_count, 2 * intermediate), DType.BF16, "pair Map VJP", "gate/up Contracts"
        ),
        BackwardBufferContract(
            "padded_input_cotangent", (row_count, hidden), DType.BF16, "gate/up Contract", "return Transport"
        ),
        BackwardBufferContract("route_weight_cotangent", (valid_count,), DType.FP32, "feature Fold", "return Transport"),
        BackwardBufferContract(
            "gate_up_weight_cotangent",
            (local_expert_count, 2 * intermediate, hidden),
            DType.BF16,
            "gate/up weight Contract",
            "external collective/JAX",
            True,
        ),
        BackwardBufferContract(
            "down_weight_cotangent",
            (local_expert_count, hidden, intermediate),
            DType.BF16,
            "down weight Contract",
            "external collective/JAX",
            True,
        ),
    )
    return ExpertBackwardRankABI(
        rank=rank,
        local_expert_count=local_expert_count,
        valid_destination_rows=valid_rows,
        padded_destination_rows=padded_rows,
        buffers=buffers,
    )


def _group_slices(relation: RelationPlan):
    for group, (offset, count) in enumerate(zip(relation.group_offset, relation.group_count, strict=True)):
        yield group, slice(int(offset), int(offset + count))


def _validate_execution_shapes(
    relation: RelationPlan,
    source: np.ndarray,
    gate_up_weight: np.ndarray,
    down_weight: np.ndarray,
    output_cotangent: np.ndarray,
) -> None:
    if source.ndim != 2 or output_cotangent.shape != source.shape:
        raise ValueError("source and output cotangent must have identical [source, hidden] shapes")
    if gate_up_weight.ndim != 3 or down_weight.ndim != 3:
        raise ValueError("segmented Contract weights must have [group, output, input] rank")
    if gate_up_weight.shape[0] != relation.destination_count or down_weight.shape[0] != relation.destination_count:
        raise ValueError("weight group count must match RelationPlan destinations")
    if gate_up_weight.shape[2] != source.shape[1] or down_weight.shape[1] != source.shape[1]:
        raise ValueError("Contract hidden dimensions disagree")
    if gate_up_weight.shape[1] != 2 * down_weight.shape[2]:
        raise ValueError("pair Contract output must contain two equal intermediate halves")


def _evaluate_array_expression(
    expression: ScalarExpression,
    inputs: dict[str, np.ndarray],
) -> np.ndarray:
    kind = expression.kind
    if kind is ScalarExpressionKind.INPUT:
        assert expression.input_name is not None
        return inputs[expression.input_name]
    if kind is ScalarExpressionKind.CONSTANT:
        assert expression.constant is not None
        return np.asarray(expression.constant, dtype=np.float32)
    operands = tuple(_evaluate_array_expression(operand, inputs) for operand in expression.operands)
    if kind is ScalarExpressionKind.ADD:
        return operands[0] + operands[1]
    if kind is ScalarExpressionKind.SUBTRACT:
        return operands[0] - operands[1]
    if kind is ScalarExpressionKind.MULTIPLY:
        return operands[0] * operands[1]
    if kind is ScalarExpressionKind.DIVIDE:
        return operands[0] / operands[1]
    if kind is ScalarExpressionKind.EXP:
        return np.exp(operands[0])
    if kind is ScalarExpressionKind.LOG:
        return np.log(operands[0])
    if kind is ScalarExpressionKind.RSQRT:
        return np.reciprocal(np.sqrt(operands[0]))
    if kind is ScalarExpressionKind.TANH:
        return np.tanh(operands[0])
    if kind is ScalarExpressionKind.LESS_EQUAL:
        return operands[0] <= operands[1]
    if kind is ScalarExpressionKind.SELECT:
        return np.where(operands[0], operands[1], operands[2])
    raise AssertionError(f"unhandled scalar expression kind {kind}")
