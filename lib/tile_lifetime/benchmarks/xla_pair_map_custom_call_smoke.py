#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "jax==0.11.0",
#   "jaxlib==0.11.0",
# ]
# ///

"""Disposable proof that a recovered XLA region can become a Shuttle call.

This smoke intentionally uses two mechanisms that are not suitable for the
production bridge: an HLO text round trip and the legacy CPU custom-call ABI.
The useful result is narrower: a PRE_SCHEDULER callback can structurally
recover an ordinary two-Contract/Map/Contract program, generate its fixed-shape
physical body, replace the region, and execute the replacement.
"""

from __future__ import annotations

import argparse
import ctypes
import gzip
import hashlib
import importlib
import json
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import jaxlib
import numpy as np

from tile_lifetime.xla_hlo_recovery import (
    HloModuleGraph,
    InlinedHloGraph,
    RecoveredEntryRegionBoundary,
    form_pair_map_entry_region,
    inline_elementwise_fusions,
    parse_hlo_module_text,
    recover_multi_output_contract_map_regions,
    recover_pair_map_regions,
)

_PASS_NAME = "shuttle_pair_map_custom_call_smoke_v1"
_TARGET_NAME = "shuttle.pair_map_contract_smoke_v1"
_PARAMETER_NUMBER = re.compile(r"parameter\((\d+)\)")
_CONSTANT_VALUE = re.compile(r"constant\(([^)]+)\)")
_SHAPE = re.compile(r"(?P<dtype>[a-z0-9]+)\[(?P<dims>[0-9,]*)\](?:\{[^}]*\})?")
_CONTRACTING_DIMENSIONS = re.compile(
    r"lhs_contracting_dims=\{(?P<lhs>[0-9]+)\}, rhs_contracting_dims=\{(?P<rhs>[0-9]+)\}"
)


def write_gzip_text(path: Path, value: str) -> None:
    """Write a reproducible compressed text artifact."""
    path.write_bytes(gzip.compress(value.encode(), compresslevel=9, mtime=0))


@dataclass(frozen=True)
class FixedShapeProgram:
    """A structurally recovered fixed-shape pair-Map program."""

    rows: int
    reduction: int
    features: int
    outputs: int
    activation_parameter: int
    left_weight_parameter: int
    right_weight_parameter: int
    down_weight_parameter: int
    scalar_expression: str


@dataclass(frozen=True)
class RegionLocalRewrite:
    """A checked entry-instruction boundary for one recovered region."""

    program: FixedShapeProgram
    target_instruction: str
    target_shape: str
    operand_instructions: tuple[str, str, str, str]
    operand_shapes: tuple[str, str, str, str]
    preserved_map_casts: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class MultiOutputFixedShapeProgram:
    """A pair-Contract plus scalar reverse Map with several live outputs."""

    rows: int
    reduction: int
    features: int
    scalar_expressions: tuple[str, ...]


@dataclass(frozen=True)
class MultiOutputRegionRewrite:
    """One generic connected region lowered through a tuple result."""

    program: MultiOutputFixedShapeProgram
    boundary: RecoveredEntryRegionBoundary


@dataclass(frozen=True)
class ContractMapFixedShapeProgram:
    """One generic Contract followed by a generated multi-output scalar Map."""

    rows: int
    reduction: int
    features: int
    input_dtypes: tuple[str, ...]
    output_dtype: str
    contract_lhs_input: int
    contract_rhs_input: int
    rhs_contracting_dimension: int
    scalar_expressions: tuple[str, ...]


@dataclass(frozen=True)
class ContractMapRegionRewrite:
    """A checked one-Contract/multi-output-Map entry boundary."""

    program: ContractMapFixedShapeProgram
    boundary: RecoveredEntryRegionBoundary


def natural_program(
    activation: jax.Array,
    left_weight: jax.Array,
    right_weight: jax.Array,
    down_weight: jax.Array,
) -> jax.Array:
    """Ordinary tensor algebra; no Shuttle or workload operation appears."""
    left = activation @ left_weight
    right = activation @ right_weight
    return (jnp.tanh(left) * right) @ down_weight


def _shape(shape: str) -> tuple[str, tuple[int, ...]]:
    match = _SHAPE.fullmatch(shape)
    if match is None:
        raise ValueError(f"unsupported fixed-shape HLO value {shape!r}")
    dimensions = tuple(int(value) for value in match.group("dims").split(",") if value)
    return match.group("dtype"), dimensions


def _entry_parameter_numbers(module: HloModuleGraph, graph: InlinedHloGraph) -> dict[str, int]:
    entry = module.computation(module.entry)
    instructions = {instruction.name: instruction for instruction in entry.instructions}
    result: dict[str, int] = {}
    for node in graph.nodes:
        if node.source_computation != module.entry or node.opcode != "parameter":
            continue
        instruction = instructions[node.source_instruction]
        match = _PARAMETER_NUMBER.search(instruction.attributes)
        if match is None:
            raise ValueError(f"entry parameter {instruction.name!r} has no ordinal")
        result[node.id] = int(match.group(1))
    return result


def _pointwise_c_expression(
    graph: InlinedHloGraph,
    node_id: str,
    leaf_expressions: dict[str, str],
    *,
    round_bf16_operations: bool = False,
) -> str:
    if node_id in leaf_expressions:
        return leaf_expressions[node_id]
    node = graph.node(node_id)
    if node.opcode == "convert" and len(node.operands) == 1:
        operand = _pointwise_c_expression(
            graph,
            node.operands[0],
            leaf_expressions,
            round_bf16_operations=round_bf16_operations,
        )
        source_dtype, _ = _shape(graph.node(node.operands[0]).shape)
        result_dtype, _ = _shape(node.shape)
        if source_dtype == "f32" and result_dtype == "bf16":
            return f"shuttle_round_bf16({operand})"
        if source_dtype == result_dtype or {source_dtype, result_dtype} == {"f32", "bf16"}:
            return operand
        raise ValueError(f"unsupported generated scalar conversion {source_dtype}->{result_dtype}")
    if node.opcode in {"copy", "reshape", "bitcast", "broadcast"} and len(node.operands) == 1:
        return _pointwise_c_expression(
            graph,
            node.operands[0],
            leaf_expressions,
            round_bf16_operations=round_bf16_operations,
        )
    if node.opcode == "constant" and not node.operands:
        match = _CONSTANT_VALUE.search(node.attributes)
        if match is None:
            raise ValueError(f"scalar constant has no value: {node.attributes!r}")
        return match.group(1)
    if node.opcode in {"tanh", "exponential", "negate"} and len(node.operands) == 1:
        operand = _pointwise_c_expression(
            graph,
            node.operands[0],
            leaf_expressions,
            round_bf16_operations=round_bf16_operations,
        )
        if node.opcode == "tanh":
            expression = f"std::tanh({operand})"
        elif node.opcode == "exponential":
            expression = f"std::exp({operand})"
        else:
            expression = f"(-({operand}))"
        if round_bf16_operations and node.dtype == "bf16":
            return f"shuttle_round_bf16({expression})"
        return expression
    binary = {
        "add": "+",
        "subtract": "-",
        "multiply": "*",
        "divide": "/",
    }
    if node.opcode in binary and len(node.operands) == 2:
        left = _pointwise_c_expression(
            graph,
            node.operands[0],
            leaf_expressions,
            round_bf16_operations=round_bf16_operations,
        )
        right = _pointwise_c_expression(
            graph,
            node.operands[1],
            leaf_expressions,
            round_bf16_operations=round_bf16_operations,
        )
        expression = f"(({left}) {binary[node.opcode]} ({right}))"
        if round_bf16_operations and node.dtype == "bf16":
            return f"shuttle_round_bf16({expression})"
        return expression
    raise ValueError(f"unsupported generated scalar node {node.opcode!r}")


def recover_fixed_shape_program(hlo_text: str) -> FixedShapeProgram:
    """Recover dimensions, buffers, and scalar AST without source-name checks."""
    report = recover_pair_map_regions(hlo_text)
    if len(report.regions) != 1:
        raise ValueError(f"expected one pair-Map region, found {len(report.regions)}")
    region = report.regions[0]
    if len(region.consumer_contracts) != 1:
        raise ValueError("pair-Map region must feed exactly one consumer Contract")
    module = parse_hlo_module_text(hlo_text)
    graph = inline_elementwise_fusions(module)
    parameters = _entry_parameter_numbers(module, graph)

    contract_nodes = (region.left_contract.node, region.right_contract.node)
    contracts = tuple(graph.node(node_id) for node_id in contract_nodes)
    shared = region.shared_input
    if shared not in parameters:
        raise ValueError("shared Contract input is not an entry parameter")
    weight_nodes: list[str] = []
    for contract in contracts:
        if len(contract.operands) != 2 or contract.operands[0] != shared:
            raise ValueError("smoke supports row-major shared lhs Contracts only")
        weight_nodes.append(contract.operands[1])
    if any(weight not in parameters for weight in weight_nodes):
        raise ValueError("projection weight is not an entry parameter")

    consumer = graph.node(region.consumer_contracts[0])
    if len(consumer.operands) != 2:
        raise ValueError("consumer Contract must be binary")
    mapped_operand = graph.strip_wrappers(consumer.operands[0])
    if mapped_operand.base != region.map_root:
        raise ValueError("smoke supports pair-Map as consumer Contract lhs")
    down_weight = graph.strip_wrappers(consumer.operands[1]).base
    if down_weight not in parameters:
        raise ValueError("consumer weight is not an entry parameter")

    activation_dtype, activation_dims = _shape(graph.node(shared).shape)
    left_dtype, left_dims = _shape(graph.node(weight_nodes[0]).shape)
    right_dtype, right_dims = _shape(graph.node(weight_nodes[1]).shape)
    down_dtype, down_dims = _shape(graph.node(down_weight).shape)
    output_dtype, output_dims = _shape(consumer.shape)
    if {activation_dtype, left_dtype, right_dtype, down_dtype, output_dtype} != {"f32"}:
        raise ValueError("legacy CPU smoke supports f32 only")
    rows, reduction = activation_dims
    if left_dims != right_dims or left_dims[0] != reduction:
        raise ValueError("projection shapes are incompatible")
    features = left_dims[1]
    if down_dims[0] != features or output_dims[0] != rows:
        raise ValueError("consumer Contract shape is incompatible")
    outputs = down_dims[1]
    if output_dims != (rows, outputs):
        raise ValueError("consumer output shape is incompatible")

    contract_names = {
        contract_nodes[0]: "projection0[row * kFeatures + feature]",
        contract_nodes[1]: "projection1[row * kFeatures + feature]",
    }
    expression = _pointwise_c_expression(graph, region.map_root, contract_names)
    return FixedShapeProgram(
        rows=rows,
        reduction=reduction,
        features=features,
        outputs=outputs,
        activation_parameter=parameters[shared],
        left_weight_parameter=parameters[weight_nodes[0]],
        right_weight_parameter=parameters[weight_nodes[1]],
        down_weight_parameter=parameters[down_weight],
        scalar_expression=expression,
    )


def recover_region_local_rewrite(hlo_text: str, region_index: int) -> RegionLocalRewrite:
    """Recover one entry-local call boundary around a generic pair-Map region."""
    report = recover_pair_map_regions(hlo_text)
    if not 0 <= region_index < len(report.regions):
        raise ValueError(f"region index {region_index} is outside {len(report.regions)} recovered regions")
    region = report.regions[region_index]
    if len(region.consumer_contracts) != 1:
        raise ValueError("region-local smoke requires exactly one consumer Contract")
    module = parse_hlo_module_text(hlo_text)
    graph = inline_elementwise_fusions(module)
    entry = module.computation(module.entry)
    instructions = {instruction.name: instruction for instruction in entry.instructions}

    def entry_instruction_for(node_id: str) -> str:
        prefix = f"{module.entry}/"
        if not node_id.startswith(prefix):
            raise ValueError(f"node {node_id!r} is outside the entry computation")
        name = node_id.removeprefix(prefix).split("/", 1)[0]
        if name not in instructions:
            raise ValueError(f"node {node_id!r} has no entry instruction boundary")
        return name

    projection_names = (
        entry_instruction_for(region.left_contract.node),
        entry_instruction_for(region.right_contract.node),
    )
    projection_instructions = tuple(instructions[name] for name in projection_names)
    if any(value.opcode != "fusion" or len(value.operands) != 2 for value in projection_instructions):
        raise ValueError("projection Contracts are not isolated two-input entry fusions")
    shared_names = {value.operands[0] for value in projection_instructions}
    if len(shared_names) != 1:
        raise ValueError("projection Contracts do not share one physical lhs boundary")
    shared_name = next(iter(shared_names))
    weight_names = tuple(value.operands[1] for value in projection_instructions)

    map_name = entry_instruction_for(region.map_root)
    map_instruction = instructions[map_name]
    if map_instruction.opcode != "fusion" or set(map_instruction.operands) != set(projection_names):
        raise ValueError("pair Map is not an isolated entry fusion over the recovered Contracts")
    consumer_name = entry_instruction_for(region.consumer_contracts[0])
    consumer_instruction = instructions[consumer_name]
    if consumer_instruction.opcode not in {"dot", "fusion"} or len(consumer_instruction.operands) != 2:
        raise ValueError("consumer Contract is not one binary entry instruction")
    if consumer_instruction.operands[0] != map_name:
        raise ValueError("pair Map is not the physical lhs of the consumer Contract")
    down_weight_name = consumer_instruction.operands[1]
    operand_names = (shared_name, weight_names[0], weight_names[1], down_weight_name)
    if any(name not in instructions for name in operand_names):
        raise ValueError("region-local operand is not defined in the entry computation")

    activation_dtype, activation_dims = _shape(instructions[shared_name].shape)
    left_dtype, left_dims = _shape(instructions[weight_names[0]].shape)
    right_dtype, right_dims = _shape(instructions[weight_names[1]].shape)
    down_dtype, down_dims = _shape(instructions[down_weight_name].shape)
    output_dtype, output_dims = _shape(consumer_instruction.shape)
    if {activation_dtype, left_dtype, right_dtype, down_dtype, output_dtype} != {"f32"}:
        raise ValueError("legacy region-local CPU smoke supports f32 physical boundaries only")
    rows, reduction = activation_dims
    if left_dims != right_dims or left_dims[0] != reduction:
        raise ValueError("projection shapes are incompatible")
    features = left_dims[1]
    outputs = down_dims[1]
    if down_dims[0] != features or output_dims != (rows, outputs):
        raise ValueError("consumer Contract shapes are incompatible")
    contract_names = {
        region.left_contract.node: "projection0[row * kFeatures + feature]",
        region.right_contract.node: "projection1[row * kFeatures + feature]",
    }
    expression = _pointwise_c_expression(graph, region.map_root, contract_names)
    program = FixedShapeProgram(
        rows=rows,
        reduction=reduction,
        features=features,
        outputs=outputs,
        activation_parameter=0,
        left_weight_parameter=1,
        right_weight_parameter=2,
        down_weight_parameter=3,
        scalar_expression=expression,
    )
    cast_boundaries = tuple((boundary.source_shape, boundary.result_shape) for boundary in region.map_cast_boundaries)
    return RegionLocalRewrite(
        program=program,
        target_instruction=consumer_name,
        target_shape=consumer_instruction.shape,
        operand_instructions=operand_names,
        operand_shapes=tuple(instructions[name].shape for name in operand_names),
        preserved_map_casts=cast_boundaries,
    )


def recover_multi_output_region_rewrite(hlo_text: str, region_index: int) -> MultiOutputRegionRewrite:
    """Recover a connected pair-Contract reverse Map with explicit side inputs."""
    report = recover_pair_map_regions(hlo_text)
    if not 0 <= region_index < len(report.regions):
        raise ValueError(f"region index {region_index} is outside {len(report.regions)} recovered regions")
    region = report.regions[region_index]
    boundary = form_pair_map_entry_region(hlo_text, region)
    if boundary.has_explicit_sharding or boundary.has_side_effect:
        raise ValueError("text smoke refuses sharded or effectful connected regions")
    if len(boundary.inputs) != 4 or len(boundary.outputs) < 2:
        raise ValueError("multi-output smoke expects four inputs and at least two outputs")
    activation_dtype, activation_dims = _shape(boundary.inputs[0].shape)
    left_dtype, left_dims = _shape(boundary.inputs[1].shape)
    right_dtype, right_dims = _shape(boundary.inputs[2].shape)
    side_dtype, side_dims = _shape(boundary.inputs[3].shape)
    if (activation_dtype, left_dtype, right_dtype, side_dtype) != ("f32", "f32", "f32", "bf16"):
        raise ValueError("unsupported connected-region physical input types")
    rows, reduction = activation_dims
    if left_dims != right_dims or left_dims[0] != reduction:
        raise ValueError("connected-region projection shapes are incompatible")
    features = left_dims[1]
    if side_dims != (rows, features):
        raise ValueError("connected-region side input shape is incompatible")
    if any(_shape(output.shape) != ("f32", (rows, features)) for output in boundary.outputs):
        raise ValueError("connected-region outputs must be row-major f32 pair-Map values")

    module = parse_hlo_module_text(hlo_text)
    graph = inline_elementwise_fusions(module)
    prefix = f"{module.entry}/"

    def projection_node(instruction: str) -> str:
        candidates = (
            node.id for node in graph.nodes if node.id.startswith(f"{prefix}{instruction}/") and node.opcode == "dot"
        )
        try:
            return next(candidates)
        except StopIteration as error:
            raise ValueError(f"entry instruction {instruction!r} has no inlined Contract") from error

    projection_instructions = (
        region.left_contract.node.removeprefix(prefix).split("/", 1)[0],
        region.right_contract.node.removeprefix(prefix).split("/", 1)[0],
    )
    leaves = {
        projection_node(projection_instructions[0]): "projection0[row * kFeatures + feature]",
        projection_node(projection_instructions[1]): "projection1[row * kFeatures + feature]",
        graph.entry_value(boundary.inputs[3].instruction): "shuttle_bf16_to_f32(cotangent[row * kFeatures + feature])",
    }
    expressions = tuple(
        _pointwise_c_expression(graph, graph.entry_value(output.instruction), leaves) for output in boundary.outputs
    )
    return MultiOutputRegionRewrite(
        program=MultiOutputFixedShapeProgram(
            rows=rows,
            reduction=reduction,
            features=features,
            scalar_expressions=expressions,
        ),
        boundary=boundary,
    )


def recover_contract_map_region_rewrite(hlo_text: str, region_index: int) -> ContractMapRegionRewrite:
    """Lower one structurally recovered Contract/multi-output-Map boundary.

    Selection and scalar generation use only HLO opcodes, shapes, and
    dependencies. The current execution proof accepts a rank-two lhs Contract
    and either physical rhs orientation; surrounding scalar inputs and outputs
    may be f32 or bf16.
    """
    report = recover_multi_output_contract_map_regions(hlo_text)
    if not 0 <= region_index < len(report.regions):
        raise ValueError(f"region index {region_index} is outside {len(report.regions)} recovered regions")
    region = report.regions[region_index]
    boundary = region.boundary
    if boundary.has_explicit_sharding or boundary.has_side_effect:
        raise ValueError("text smoke refuses sharded or effectful connected regions")
    if len(boundary.outputs) < 2:
        raise ValueError("Contract/Map smoke requires at least two outputs")

    module = parse_hlo_module_text(hlo_text)
    graph = inline_elementwise_fusions(module)
    entry = module.computation(module.entry)
    instructions = {instruction.name: instruction for instruction in entry.instructions}
    prefix = f"{module.entry}/"
    if not region.contract.node.startswith(prefix):
        raise ValueError("recovered Contract is not physically in the entry computation")
    contract_name = region.contract.node.removeprefix(prefix).split("/", 1)[0]
    contract = instructions[contract_name]
    if contract.opcode != "dot" or len(contract.operands) != 2:
        raise ValueError("execution proof requires one physical binary entry Contract")
    try:
        lhs_input = tuple(value.instruction for value in boundary.inputs).index(contract.operands[0])
        rhs_input = tuple(value.instruction for value in boundary.inputs).index(contract.operands[1])
    except ValueError as error:
        raise ValueError("physical Contract operands are not explicit region inputs") from error

    dimension_match = _CONTRACTING_DIMENSIONS.search(contract.attributes)
    if dimension_match is None:
        raise ValueError("Contract has no one-axis contracting-dimension description")
    lhs_contracting_dimension = int(dimension_match.group("lhs"))
    rhs_contracting_dimension = int(dimension_match.group("rhs"))
    if lhs_contracting_dimension != 1 or rhs_contracting_dimension not in {0, 1}:
        raise ValueError("execution proof requires lhs dimension 1 and rank-two rhs dimension 0 or 1")

    input_types_and_dims = tuple(_shape(value.shape) for value in boundary.inputs)
    lhs_dtype, lhs_dims = input_types_and_dims[lhs_input]
    rhs_dtype, rhs_dims = input_types_and_dims[rhs_input]
    contract_dtype, contract_dims = _shape(contract.shape)
    if lhs_dtype != rhs_dtype or contract_dtype != lhs_dtype or lhs_dtype not in {"f32", "bf16"}:
        raise ValueError("Contract execution proof requires one homogeneous f32 or bf16 dtype")
    if len(lhs_dims) != 2 or len(rhs_dims) != 2 or len(contract_dims) != 2:
        raise ValueError("Contract execution proof requires rank-two physical values")
    rows, reduction = lhs_dims
    if rhs_contracting_dimension == 0:
        rhs_reduction, features = rhs_dims
    else:
        features, rhs_reduction = rhs_dims
    if rhs_reduction != reduction or contract_dims != (rows, features):
        raise ValueError("Contract dimensions are incompatible")

    output_types_and_dims = tuple(_shape(value.shape) for value in boundary.outputs)
    if any(value != (contract_dtype, contract_dims) for value in output_types_and_dims):
        raise ValueError("scalar Map outputs must preserve the Contract shape and dtype")
    for index, (dtype, dimensions) in enumerate(input_types_and_dims):
        if dtype not in {"f32", "bf16"} or len(dimensions) != 2:
            raise ValueError(f"unsupported region input {index}: {boundary.inputs[index].shape}")
        if index not in {lhs_input, rhs_input} and dimensions != contract_dims:
            raise ValueError("scalar side inputs must have the Contract output shape")

    leaves = {region.contract.node: "projection_value"}
    for index, value in enumerate(boundary.inputs):
        if index in {lhs_input, rhs_input}:
            continue
        node = graph.entry_value(value.instruction)
        if input_types_and_dims[index][0] == "bf16":
            leaves[node] = f"shuttle_bf16_to_f32(input{index}[index])"
        else:
            leaves[node] = f"input{index}[index]"
    expressions = tuple(
        _pointwise_c_expression(
            graph,
            graph.entry_value(output.instruction),
            leaves,
            round_bf16_operations=True,
        )
        for output in boundary.outputs
    )
    return ContractMapRegionRewrite(
        program=ContractMapFixedShapeProgram(
            rows=rows,
            reduction=reduction,
            features=features,
            input_dtypes=tuple(dtype for dtype, _ in input_types_and_dims),
            output_dtype=contract_dtype,
            contract_lhs_input=lhs_input,
            contract_rhs_input=rhs_input,
            rhs_contracting_dimension=rhs_contracting_dimension,
            scalar_expressions=expressions,
        ),
        boundary=boundary,
    )


def pair_map_recovery_diagnostic(hlo_text: str) -> dict[str, Any]:
    """Describe every structural pair-Map candidate before selecting a region."""
    report = recover_pair_map_regions(hlo_text)
    candidates: list[dict[str, Any]] = []
    for index, region in enumerate(report.regions):
        try:
            boundary = form_pair_map_entry_region(hlo_text, region)
        except ValueError as error:
            candidates.append(
                {
                    "index": index,
                    "map_opcodes": region.map_opcodes,
                    "boundary_error": str(error),
                }
            )
            continue
        candidates.append(
            {
                "index": index,
                "map_opcodes": region.map_opcodes,
                "input_shapes": tuple(value.shape for value in boundary.inputs),
                "output_shapes": tuple(value.shape for value in boundary.outputs),
                "external_user_counts": tuple(len(users) for _, users in boundary.external_users),
                "internal_instruction_count": len(boundary.internal_instructions),
                "has_explicit_sharding": boundary.has_explicit_sharding,
                "has_side_effect": boundary.has_side_effect,
            }
        )
    return {
        "contract_count": report.contract_count,
        "pair_map_region_count": len(report.regions),
        "candidates": candidates,
    }


def contract_map_recovery_diagnostic(hlo_text: str) -> dict[str, Any]:
    """Describe generic one-Contract/multi-output-Map candidates and lowering."""
    report = recover_multi_output_contract_map_regions(hlo_text)
    candidates: list[dict[str, Any]] = []
    for index, region in enumerate(report.regions):
        candidate: dict[str, Any] = {
            "index": index,
            "contract_shape": region.contract.output_shape,
            "map_opcodes": region.map_opcodes,
            "input_shapes": tuple(value.shape for value in region.boundary.inputs),
            "output_shapes": tuple(value.shape for value in region.boundary.outputs),
            "consumer_contract_count": len(region.consumer_contracts),
            "has_explicit_sharding": region.boundary.has_explicit_sharding,
            "has_side_effect": region.boundary.has_side_effect,
        }
        try:
            rewrite = recover_contract_map_region_rewrite(hlo_text, index)
        except ValueError as error:
            candidate["lowering_error"] = str(error)
        else:
            candidate["contract_lhs_input"] = rewrite.program.contract_lhs_input
            candidate["contract_rhs_input"] = rewrite.program.contract_rhs_input
            candidate["rhs_contracting_dimension"] = rewrite.program.rhs_contracting_dimension
            candidate["scalar_expressions"] = rewrite.program.scalar_expressions
        candidates.append(candidate)
    return {
        "contract_count": report.contract_count,
        "contract_map_region_count": len(report.regions),
        "candidates": candidates,
    }


def generate_cpu_handler(program: FixedShapeProgram) -> str:
    """Generate a fixed-shape legacy CPU handler from the recovered program."""
    return f"""// Generated by xla_pair_map_custom_call_smoke.py; do not edit.
#include <atomic>
#include <cstdint>
#include <cstring>
#include <cmath>

namespace {{
constexpr int kRows = {program.rows};
constexpr int kReduction = {program.reduction};
constexpr int kFeatures = {program.features};
constexpr int kOutputs = {program.outputs};
std::atomic<int> call_count{{0}};

float shuttle_round_bf16(float value) {{
  std::uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  const std::uint32_t least_significant = (bits >> 16) & 1;
  bits += 0x7fff + least_significant;
  bits &= 0xffff0000;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}}
}}

extern "C" void {_TARGET_NAME.replace('.', '_')}(void* output, const void** inputs) {{
  auto* destination = static_cast<float*>(output);
  const auto* activation = static_cast<const float*>(inputs[{program.activation_parameter}]);
  const auto* left_weight = static_cast<const float*>(inputs[{program.left_weight_parameter}]);
  const auto* right_weight = static_cast<const float*>(inputs[{program.right_weight_parameter}]);
  const auto* down_weight = static_cast<const float*>(inputs[{program.down_weight_parameter}]);
  float projection0[kRows * kFeatures];
  float projection1[kRows * kFeatures];
  float mapped[kRows * kFeatures];
  for (int row = 0; row < kRows; ++row) {{
    for (int feature = 0; feature < kFeatures; ++feature) {{
      float left = 0.0f;
      float right = 0.0f;
      for (int reduction = 0; reduction < kReduction; ++reduction) {{
        const float value = activation[row * kReduction + reduction];
        left += value * left_weight[reduction * kFeatures + feature];
        right += value * right_weight[reduction * kFeatures + feature];
      }}
      projection0[row * kFeatures + feature] = left;
      projection1[row * kFeatures + feature] = right;
      mapped[row * kFeatures + feature] = {program.scalar_expression};
    }}
  }}
  for (int row = 0; row < kRows; ++row) {{
    for (int column = 0; column < kOutputs; ++column) {{
      float accumulator = 0.0f;
      for (int feature = 0; feature < kFeatures; ++feature) {{
        accumulator += mapped[row * kFeatures + feature] * down_weight[feature * kOutputs + column];
      }}
      destination[row * kOutputs + column] = accumulator;
    }}
  }}
  call_count.fetch_add(1, std::memory_order_relaxed);
}}

extern "C" int shuttle_pair_map_smoke_call_count() {{
  return call_count.load(std::memory_order_relaxed);
}}
"""


def generate_cpu_multi_output_handler(program: MultiOutputFixedShapeProgram) -> str:
    """Generate a legacy tuple-output handler for one connected reverse Map."""
    stores = "\n".join(
        f"      output{index}[row * kFeatures + feature] = {expression};"
        for index, expression in enumerate(program.scalar_expressions)
    )
    output_bindings = "\n".join(
        f"  auto* output{index} = static_cast<float*>(output_tuple[{index}]);"
        for index in range(len(program.scalar_expressions))
    )
    return f"""// Generated by xla_pair_map_custom_call_smoke.py; do not edit.
#include <atomic>
#include <cstdint>
#include <cstring>
#include <cmath>

namespace {{
constexpr int kRows = {program.rows};
constexpr int kReduction = {program.reduction};
constexpr int kFeatures = {program.features};
std::atomic<int> call_count{{0}};

float shuttle_round_bf16(float value) {{
  std::uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  const std::uint32_t least_significant = (bits >> 16) & 1;
  bits += 0x7fff + least_significant;
  bits &= 0xffff0000;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}}

float shuttle_bf16_to_f32(std::uint16_t value) {{
  std::uint32_t bits = static_cast<std::uint32_t>(value) << 16;
  float result;
  std::memcpy(&result, &bits, sizeof(result));
  return result;
}}
}}

extern "C" void {_TARGET_NAME.replace('.', '_')}(void* output, const void** inputs) {{
  auto** output_tuple = static_cast<void**>(output);
{output_bindings}
  const auto* activation = static_cast<const float*>(inputs[0]);
  const auto* left_weight = static_cast<const float*>(inputs[1]);
  const auto* right_weight = static_cast<const float*>(inputs[2]);
  const auto* cotangent = static_cast<const std::uint16_t*>(inputs[3]);
  float projection0[kRows * kFeatures];
  float projection1[kRows * kFeatures];
  for (int row = 0; row < kRows; ++row) {{
    for (int feature = 0; feature < kFeatures; ++feature) {{
      float left = 0.0f;
      float right = 0.0f;
      for (int reduction = 0; reduction < kReduction; ++reduction) {{
        const float value = activation[row * kReduction + reduction];
        left += value * left_weight[reduction * kFeatures + feature];
        right += value * right_weight[reduction * kFeatures + feature];
      }}
      projection0[row * kFeatures + feature] = left;
      projection1[row * kFeatures + feature] = right;
{stores}
    }}
  }}
  call_count.fetch_add(1, std::memory_order_relaxed);
}}

extern "C" int shuttle_pair_map_smoke_call_count() {{
  return call_count.load(std::memory_order_relaxed);
}}
"""


def generate_cpu_multi_output_ffi_handler(program: MultiOutputFixedShapeProgram) -> str:
    """Generate a supported typed-FFI handler for a connected reverse Map."""
    output_arguments = ",\n    ".join(
        f"ffi::Result<ffi::Buffer<ffi::F32, 2>> output{index}" for index in range(len(program.scalar_expressions))
    )
    output_bindings = "\n".join(
        f"  auto* output{index}_data = output{index}->typed_data();" for index in range(len(program.scalar_expressions))
    )
    stores = "\n".join(
        f"      output{index}_data[row * kFeatures + feature] = {expression};"
        for index, expression in enumerate(program.scalar_expressions)
    )
    return_bindings = "\n".join("      .Ret<ffi::Buffer<ffi::F32, 2>>()" for _ in program.scalar_expressions)
    target_symbol = _TARGET_NAME.replace(".", "_")
    return f"""// Generated by xla_pair_map_custom_call_smoke.py; do not edit.
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstring>

#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;

namespace {{
constexpr int kRows = {program.rows};
constexpr int kReduction = {program.reduction};
constexpr int kFeatures = {program.features};
std::atomic<int> call_count{{0}};

float shuttle_round_bf16(float value) {{
  std::uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  const std::uint32_t least_significant = (bits >> 16) & 1;
  bits += 0x7fff + least_significant;
  bits &= 0xffff0000;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}}

float shuttle_bf16_to_f32(std::uint16_t value) {{
  std::uint32_t bits = static_cast<std::uint32_t>(value) << 16;
  float result;
  std::memcpy(&result, &bits, sizeof(result));
  return result;
}}

ffi::Error ShuttlePairMapRegion(
    ffi::Buffer<ffi::F32, 2> activation,
    ffi::Buffer<ffi::F32, 2> left_weight,
    ffi::Buffer<ffi::F32, 2> right_weight,
    ffi::Buffer<ffi::BF16, 2> cotangent_buffer,
    {output_arguments}) {{
  const auto* activation_data = activation.typed_data();
  const auto* left_weight_data = left_weight.typed_data();
  const auto* right_weight_data = right_weight.typed_data();
  const auto* cotangent = cotangent_buffer.typed_data();
{output_bindings}
  float projection0[kRows * kFeatures];
  float projection1[kRows * kFeatures];
  for (int row = 0; row < kRows; ++row) {{
    for (int feature = 0; feature < kFeatures; ++feature) {{
      float left = 0.0f;
      float right = 0.0f;
      for (int reduction = 0; reduction < kReduction; ++reduction) {{
        const float value = activation_data[row * kReduction + reduction];
        left += value * left_weight_data[reduction * kFeatures + feature];
        right += value * right_weight_data[reduction * kFeatures + feature];
      }}
      projection0[row * kFeatures + feature] = left;
      projection1[row * kFeatures + feature] = right;
{stores}
    }}
  }}
  call_count.fetch_add(1, std::memory_order_relaxed);
  return ffi::Error::Success();
}}

auto ShuttlePairMapRegionBinding() {{
  return ffi::Ffi::Bind()
      .Arg<ffi::Buffer<ffi::F32, 2>>()
      .Arg<ffi::Buffer<ffi::F32, 2>>()
      .Arg<ffi::Buffer<ffi::F32, 2>>()
      .Arg<ffi::Buffer<ffi::BF16, 2>>()
{return_bindings};
}}
}}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    {target_symbol},
    ShuttlePairMapRegion,
    ShuttlePairMapRegionBinding());

extern "C" int shuttle_pair_map_smoke_call_count() {{
  return call_count.load(std::memory_order_relaxed);
}}
"""


def generate_cuda_multi_output_ffi_handler(program: MultiOutputFixedShapeProgram) -> str:
    """Generate a CUDA typed-FFI handler from the same recovered scalar body."""
    output_arguments = ",\n    ".join(
        f"ffi::Result<ffi::Buffer<ffi::F32, 2>> output{index}" for index in range(len(program.scalar_expressions))
    )
    output_bindings = "\n".join(
        f"  auto* output{index}_data = output{index}->typed_data();" for index in range(len(program.scalar_expressions))
    )
    kernel_output_arguments = ", ".join(f"float* output{index}" for index in range(len(program.scalar_expressions)))
    launch_outputs = ", ".join(f"output{index}_data" for index in range(len(program.scalar_expressions)))
    stores = "\n".join(
        f"  output{index}[index] = {_cuda_scalar_expression(expression)};"
        for index, expression in enumerate(program.scalar_expressions)
    )
    return_bindings = "\n".join("      .Ret<ffi::Buffer<ffi::F32, 2>>()" for _ in program.scalar_expressions)
    target_symbol = _TARGET_NAME.replace(".", "_")
    return f"""// Generated by xla_pair_map_custom_call_smoke.py; do not edit.
#include <atomic>
#include <cstdint>
#include <string>

#include <cublas_v2.h>
#include <cuda_runtime_api.h>

#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;

namespace {{
constexpr int kRows = {program.rows};
constexpr int kReduction = {program.reduction};
constexpr int kFeatures = {program.features};
std::atomic<int> call_count{{0}};
thread_local cublasHandle_t contract_handle = nullptr;

__device__ float shuttle_round_bf16(float value) {{
  std::uint32_t bits = __float_as_uint(value);
  const std::uint32_t least_significant = (bits >> 16) & 1;
  bits += 0x7fff + least_significant;
  return __uint_as_float(bits & 0xffff0000);
}}

__device__ float shuttle_bf16_to_f32(std::uint16_t value) {{
  return __uint_as_float(static_cast<std::uint32_t>(value) << 16);
}}

__global__ void ShuttlePairMapKernel(
    const float* projection0,
    const float* projection1,
    const std::uint16_t* cotangent,
    {kernel_output_arguments}) {{
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= kRows * kFeatures) {{
    return;
  }}
  const int row = index / kFeatures;
  const int feature = index - row * kFeatures;
  const float left = projection0[index];
  const float right = projection1[index];
{stores}
}}

ffi::Error Contract(
    cudaStream_t stream,
    const float* activation,
    const float* weight,
    float* output) {{
  if (contract_handle == nullptr) {{
    const cublasStatus_t create_status = cublasCreate(&contract_handle);
    if (create_status != CUBLAS_STATUS_SUCCESS) {{
      return ffi::Error::Internal(
          "cublasCreate failed with status " + std::to_string(static_cast<int>(create_status)));
    }}
  }}
  cublasStatus_t status = cublasSetStream(contract_handle, stream);
  if (status != CUBLAS_STATUS_SUCCESS) {{
    return ffi::Error::Internal(
        "cublasSetStream failed with status " + std::to_string(static_cast<int>(status)));
  }}
  const float alpha = 1.0f;
  const float beta = 0.0f;
  status = cublasGemmEx(
      contract_handle,
      CUBLAS_OP_N,
      CUBLAS_OP_N,
      kFeatures,
      kRows,
      kReduction,
      &alpha,
      weight,
      CUDA_R_32F,
      kFeatures,
      activation,
      CUDA_R_32F,
      kReduction,
      &beta,
      output,
      CUDA_R_32F,
      kFeatures,
      CUBLAS_COMPUTE_32F_PEDANTIC,
      CUBLAS_GEMM_DEFAULT);
  if (status != CUBLAS_STATUS_SUCCESS) {{
    return ffi::Error::Internal(
        "cublasGemmEx failed with status " + std::to_string(static_cast<int>(status)));
  }}
  return ffi::Error::Success();
}}

ffi::Error ShuttlePairMapRegion(
    cudaStream_t stream,
    ffi::Buffer<ffi::F32, 2> activation,
    ffi::Buffer<ffi::F32, 2> left_weight,
    ffi::Buffer<ffi::F32, 2> right_weight,
    ffi::Buffer<ffi::BF16, 2> cotangent_buffer,
    {output_arguments}) {{
{output_bindings}
  ffi::Error contract0 = Contract(
      stream, activation.typed_data(), left_weight.typed_data(), output0_data);
  if (contract0.failure()) {{
    return contract0;
  }}
  ffi::Error contract1 = Contract(
      stream, activation.typed_data(), right_weight.typed_data(), output1_data);
  if (contract1.failure()) {{
    return contract1;
  }}
  constexpr int kThreads = 256;
  constexpr int kBlocks = (kRows * kFeatures + kThreads - 1) / kThreads;
  ShuttlePairMapKernel<<<kBlocks, kThreads, 0, stream>>>(
      output0_data,
      output1_data,
      reinterpret_cast<const std::uint16_t*>(cotangent_buffer.typed_data()),
      {launch_outputs});
  const cudaError_t status = cudaGetLastError();
  if (status != cudaSuccess) {{
    return ffi::Error::Internal(std::string("ShuttlePairMapKernel: ") + cudaGetErrorString(status));
  }}
  call_count.fetch_add(1, std::memory_order_relaxed);
  return ffi::Error::Success();
}}

auto ShuttlePairMapRegionBinding() {{
  return ffi::Ffi::Bind()
      .Ctx<ffi::PlatformStream<cudaStream_t>>()
      .Arg<ffi::Buffer<ffi::F32, 2>>()
      .Arg<ffi::Buffer<ffi::F32, 2>>()
      .Arg<ffi::Buffer<ffi::F32, 2>>()
      .Arg<ffi::Buffer<ffi::BF16, 2>>()
{return_bindings};
}}
}}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    {target_symbol},
    ShuttlePairMapRegion,
    ShuttlePairMapRegionBinding());

extern "C" int shuttle_pair_map_smoke_call_count() {{
  return call_count.load(std::memory_order_relaxed);
}}
"""


def generate_cuda_contract_map_ffi_handler(program: ContractMapFixedShapeProgram) -> str:
    """Generate a typed CUDA FFI body for one Contract and scalar Map AST."""
    if program.output_dtype not in {"f32", "bf16"}:
        raise ValueError(f"unsupported output dtype {program.output_dtype!r}")
    if any(dtype not in {"f32", "bf16"} for dtype in program.input_dtypes):
        raise ValueError(f"unsupported input dtypes {program.input_dtypes!r}")
    contract_dtype = program.input_dtypes[program.contract_lhs_input]
    if program.input_dtypes[program.contract_rhs_input] != contract_dtype or program.output_dtype != contract_dtype:
        raise ValueError("generated Contract requires homogeneous lhs, rhs, and output dtypes")

    ffi_types = {"f32": "ffi::F32", "bf16": "ffi::BF16"}
    cpp_types = {"f32": "float", "bf16": "std::uint16_t"}
    cuda_types = {"f32": "CUDA_R_32F", "bf16": "CUDA_R_16BF"}
    scalar_inputs = tuple(
        index
        for index in range(len(program.input_dtypes))
        if index not in {program.contract_lhs_input, program.contract_rhs_input}
    )
    ffi_input_arguments = ",\n    ".join(
        f"ffi::Buffer<{ffi_types[dtype]}, 2> input{index}_buffer" for index, dtype in enumerate(program.input_dtypes)
    )
    input_bindings = "\n".join(
        (
            f"  const auto* input{index} = input{index}_buffer.typed_data();"
            if dtype == "f32"
            else (
                f"  const auto* input{index} = reinterpret_cast<const std::uint16_t*>("
                f"input{index}_buffer.typed_data());"
            )
        )
        for index, dtype in enumerate(program.input_dtypes)
    )
    kernel_input_arguments = ",\n    ".join(
        f"const {cpp_types[program.input_dtypes[index]]}* input{index}" for index in scalar_inputs
    )
    launch_inputs = ",\n      ".join(f"input{index}" for index in scalar_inputs)
    output_arguments = ",\n    ".join(
        f"ffi::Result<ffi::Buffer<{ffi_types[program.output_dtype]}, 2>> output{index}"
        for index in range(len(program.scalar_expressions))
    )
    output_bindings = "\n".join(
        (
            f"  auto* output{index}_data = output{index}->typed_data();"
            if program.output_dtype == "f32"
            else (f"  auto* output{index}_data = reinterpret_cast<std::uint16_t*>(" f"output{index}->typed_data());")
        )
        for index in range(len(program.scalar_expressions))
    )
    kernel_output_arguments = ",\n    ".join(
        f"{cpp_types[program.output_dtype]}* output{index}" for index in range(len(program.scalar_expressions))
    )
    launch_outputs = ",\n      ".join(f"output{index}_data" for index in range(len(program.scalar_expressions)))
    stores = "\n".join(
        (
            f"  output{index}[index] = {_cuda_scalar_expression(expression)};"
            if program.output_dtype == "f32"
            else f"  output{index}[index] = shuttle_f32_to_bf16({_cuda_scalar_expression(expression)});"
        )
        for index, expression in enumerate(program.scalar_expressions)
    )
    input_bindings_ffi = "\n".join(f"      .Arg<ffi::Buffer<{ffi_types[dtype]}, 2>>()" for dtype in program.input_dtypes)
    output_bindings_ffi = "\n".join(
        f"      .Ret<ffi::Buffer<{ffi_types[program.output_dtype]}, 2>>()" for _ in program.scalar_expressions
    )
    rhs_operation = "CUBLAS_OP_N" if program.rhs_contracting_dimension == 0 else "CUBLAS_OP_T"
    rhs_leading_dimension = program.features if program.rhs_contracting_dimension == 0 else program.reduction
    target_symbol = _TARGET_NAME.replace(".", "_")
    projection_load = "projection[index]" if program.output_dtype == "f32" else "shuttle_bf16_to_f32(projection[index])"
    contract_cpp_type = cpp_types[contract_dtype]
    contract_cuda_type = cuda_types[contract_dtype]
    return f"""// Generated by xla_pair_map_custom_call_smoke.py; do not edit.
#include <atomic>
#include <cstdint>
#include <string>

#include <cublas_v2.h>
#include <cuda_runtime_api.h>

#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;

namespace {{
constexpr int kRows = {program.rows};
constexpr int kReduction = {program.reduction};
constexpr int kFeatures = {program.features};
std::atomic<int> call_count{{0}};
thread_local cublasHandle_t contract_handle = nullptr;

__device__ float shuttle_round_bf16(float value) {{
  std::uint32_t bits = __float_as_uint(value);
  const std::uint32_t least_significant = (bits >> 16) & 1;
  bits += 0x7fff + least_significant;
  return __uint_as_float(bits & 0xffff0000);
}}

__device__ float shuttle_bf16_to_f32(std::uint16_t value) {{
  return __uint_as_float(static_cast<std::uint32_t>(value) << 16);
}}

__device__ std::uint16_t shuttle_f32_to_bf16(float value) {{
  return static_cast<std::uint16_t>(__float_as_uint(shuttle_round_bf16(value)) >> 16);
}}

__global__ void ShuttleContractMapKernel(
    const {contract_cpp_type}* projection,
    {kernel_input_arguments},
    {kernel_output_arguments}) {{
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= kRows * kFeatures) {{
    return;
  }}
  const float projection_value = {projection_load};
{stores}
}}

ffi::Error Contract(
    cudaStream_t stream,
    const {contract_cpp_type}* lhs,
    const {contract_cpp_type}* rhs,
    {contract_cpp_type}* output) {{
  if (contract_handle == nullptr) {{
    const cublasStatus_t create_status = cublasCreate(&contract_handle);
    if (create_status != CUBLAS_STATUS_SUCCESS) {{
      return ffi::Error::Internal(
          "cublasCreate failed with status " + std::to_string(static_cast<int>(create_status)));
    }}
  }}
  cublasStatus_t status = cublasSetStream(contract_handle, stream);
  if (status != CUBLAS_STATUS_SUCCESS) {{
    return ffi::Error::Internal(
        "cublasSetStream failed with status " + std::to_string(static_cast<int>(status)));
  }}
  const float alpha = 1.0f;
  const float beta = 0.0f;
  status = cublasGemmEx(
      contract_handle,
      {rhs_operation},
      CUBLAS_OP_N,
      kFeatures,
      kRows,
      kReduction,
      &alpha,
      rhs,
      {contract_cuda_type},
      {rhs_leading_dimension},
      lhs,
      {contract_cuda_type},
      kReduction,
      &beta,
      output,
      {contract_cuda_type},
      kFeatures,
      CUBLAS_COMPUTE_32F_PEDANTIC,
      CUBLAS_GEMM_DEFAULT);
  if (status != CUBLAS_STATUS_SUCCESS) {{
    return ffi::Error::Internal(
        "cublasGemmEx failed with status " + std::to_string(static_cast<int>(status)));
  }}
  return ffi::Error::Success();
}}

ffi::Error ShuttleContractMapRegion(
    cudaStream_t stream,
    {ffi_input_arguments},
    {output_arguments}) {{
{input_bindings}
{output_bindings}
  ffi::Error contract = Contract(
      stream,
      input{program.contract_lhs_input},
      input{program.contract_rhs_input},
      output0_data);
  if (contract.failure()) {{
    return contract;
  }}
  constexpr int kThreads = 256;
  constexpr int kBlocks = (kRows * kFeatures + kThreads - 1) / kThreads;
  ShuttleContractMapKernel<<<kBlocks, kThreads, 0, stream>>>(
      output0_data,
      {launch_inputs},
      {launch_outputs});
  const cudaError_t status = cudaGetLastError();
  if (status != cudaSuccess) {{
    return ffi::Error::Internal(std::string("ShuttleContractMapKernel: ") + cudaGetErrorString(status));
  }}
  call_count.fetch_add(1, std::memory_order_relaxed);
  return ffi::Error::Success();
}}

auto ShuttleContractMapRegionBinding() {{
  return ffi::Ffi::Bind()
      .Ctx<ffi::PlatformStream<cudaStream_t>>()
{input_bindings_ffi}
{output_bindings_ffi};
}}
}}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    {target_symbol},
    ShuttleContractMapRegion,
    ShuttleContractMapRegionBinding());

extern "C" int shuttle_pair_map_smoke_call_count() {{
  return call_count.load(std::memory_order_relaxed);
}}
"""


def _cuda_scalar_expression(expression: str) -> str:
    """Bind one recovered element expression to thread-local Contract results."""
    return (
        expression.replace("std::", "::")
        .replace("projection0[row * kFeatures + feature]", "left")
        .replace("projection1[row * kFeatures + feature]", "right")
        .replace("cotangent[row * kFeatures + feature]", "cotangent[index]")
    )


def _replace_entry_with_custom_call(hlo_text: str, target: str) -> str:
    module = parse_hlo_module_text(hlo_text)
    entry = module.computation(module.entry)
    parameters = sorted(
        (instruction for instruction in entry.instructions if instruction.opcode == "parameter"),
        key=lambda instruction: int(_PARAMETER_NUMBER.search(instruction.attributes).group(1)),  # type: ignore[union-attr]
    )
    if len(parameters) != 4:
        raise ValueError(f"expected four entry parameters, found {len(parameters)}")
    entry_marker = f"ENTRY %{entry.name} "
    start = hlo_text.index(entry_marker)
    body_start = hlo_text.index("{", start)
    depth = 0
    body_end = -1
    for index in range(body_start, len(hlo_text)):
        if hlo_text[index] == "{":
            depth += 1
        elif hlo_text[index] == "}":
            depth -= 1
            if depth == 0:
                body_end = index
                break
    if body_end < 0:
        raise ValueError("could not find entry computation end")
    header = hlo_text[start:body_start].rstrip()
    parameter_lines = [
        f"  %{value.name} = {value.shape} parameter({ordinal})" for ordinal, value in enumerate(parameters)
    ]
    operands = ", ".join(f"%{value.name}" for value in parameters)
    operand_layouts = ", ".join(value.shape for value in parameters)
    output_shape = entry.root.shape
    replacement = "\n".join(
        [
            f"{header} {{",
            *parameter_lines,
            (
                f"  ROOT %shuttle_generated_region = {output_shape} custom-call({operands}), "
                f'custom_call_target="{target}", operand_layout_constraints={{{operand_layouts}}}'
            ),
            "}",
        ]
    )
    return hlo_text[:start] + replacement + hlo_text[body_end + 1 :]


def replace_region_instruction_with_custom_call(
    hlo_text: str,
    rewrite: RegionLocalRewrite,
    target: str,
) -> str:
    """Replace one structurally checked entry instruction, not the module entry."""
    module = parse_hlo_module_text(hlo_text)
    entry = module.computation(module.entry)
    instruction = next(value for value in entry.instructions if value.name == rewrite.target_instruction)
    if instruction.shape != rewrite.target_shape:
        raise ValueError("recovered target shape changed before replacement")
    if "sharding=" in instruction.attributes:
        raise ValueError("text smoke refuses to replace an instruction with explicit sharding")
    original_pattern = re.compile(
        rf"^(?P<indent>\s*)(?P<root>ROOT )?%{re.escape(instruction.name)} = .*$",
        re.MULTILINE,
    )
    matches = tuple(original_pattern.finditer(hlo_text))
    if len(matches) != 1:
        raise ValueError(f"expected one textual definition for %{instruction.name}, found {len(matches)}")
    match = matches[0]
    operands = ", ".join(f"%{name}" for name in rewrite.operand_instructions)
    layouts = ", ".join(rewrite.operand_shapes)
    replacement = (
        f"{match.group('indent')}{match.group('root') or ''}%{instruction.name} = "
        f'{rewrite.target_shape} custom-call({operands}), custom_call_target="{target}", '
        f"operand_layout_constraints={{{layouts}}}"
    )
    return hlo_text[: match.start()] + replacement + hlo_text[match.end() :]


def replace_multi_output_region_with_custom_call(
    hlo_text: str,
    rewrite: MultiOutputRegionRewrite,
    target: str,
    *,
    typed_ffi: bool = False,
) -> str:
    """Insert one tuple call and rewire all connected-region live outputs."""
    output_names = tuple(value.instruction for value in rewrite.boundary.outputs)
    output_shapes = tuple(value.shape for value in rewrite.boundary.outputs)
    first_output = output_names[0]
    pattern = re.compile(rf"^(?P<indent>\s*)%{re.escape(first_output)} = .*$", re.MULTILINE)
    match = pattern.search(hlo_text)
    if match is None:
        raise ValueError(f"could not locate first connected-region output %{first_output}")
    tuple_shape = f"({', '.join(output_shapes)})"
    operand_names = tuple(value.instruction for value in rewrite.boundary.inputs)
    operand_shapes = tuple(value.shape for value in rewrite.boundary.inputs)
    operands = ", ".join(f"%{name}" for name in operand_names)
    layouts = ", ".join(operand_shapes)
    call_name = "shuttle_generated_multi_output_region"
    api = ", api_version=API_VERSION_TYPED_FFI, backend_config={}" if typed_ffi else ""
    call = (
        f"{match.group('indent')}%{call_name} = {tuple_shape} custom-call({operands}), "
        f'custom_call_target="{target}", operand_layout_constraints={{{layouts}}}{api}\n'
    )
    rewritten = hlo_text[: match.start()] + call + hlo_text[match.start() :]
    for index, (name, shape) in enumerate(zip(output_names, output_shapes, strict=True)):
        output_pattern = re.compile(rf"^(?P<indent>\s*)%{re.escape(name)} = .*$", re.MULTILINE)
        matches = tuple(output_pattern.finditer(rewritten))
        if len(matches) != 1:
            raise ValueError(f"expected one definition for connected-region output %{name}")
        output_match = matches[0]
        replacement = (
            f"{output_match.group('indent')}%{name} = {shape} " f"get-tuple-element(%{call_name}), index={index}"
        )
        rewritten = rewritten[: output_match.start()] + replacement + rewritten[output_match.end() :]
    return rewritten


def _compile_handler(source: str, directory: Path) -> ctypes.CDLL:
    compiler = shutil.which("clang++") or shutil.which("c++")
    if compiler is None:
        raise RuntimeError("a C++ compiler is required for the CPU custom-call smoke")
    source_path = directory / "generated_pair_map_handler.cc"
    library_path = directory / "generated_pair_map_handler.so"
    source_path.write_text(source)
    subprocess.run(
        [compiler, "-std=c++17", "-O3", "-shared", "-fPIC", str(source_path), "-o", str(library_path)],
        check=True,
    )
    return ctypes.CDLL(str(library_path))


def _compile_ffi_handler(source: str, directory: Path) -> ctypes.CDLL:
    compiler = shutil.which("clang++") or shutil.which("c++")
    if compiler is None:
        raise RuntimeError("a C++ compiler is required for the CPU FFI smoke")
    source_path = directory / "generated_pair_map_handler.cc"
    library_path = directory / "generated_pair_map_handler.so"
    include_directory = Path(jaxlib.__file__).resolve().parent / "include"
    source_path.write_text(source)
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-O3",
            "-shared",
            "-fPIC",
            "-I",
            str(include_directory),
            str(source_path),
            "-o",
            str(library_path),
            "-lcublas",
        ],
        check=True,
    )
    return ctypes.CDLL(str(library_path))


def _compile_cuda_ffi_handler(
    source: str,
    directory: Path,
    nvcc: Path,
    architecture: str,
) -> ctypes.CDLL:
    """Compile one generated CUDA typed-FFI handler."""
    if not nvcc.is_file():
        raise ValueError(f"CUDA compiler does not exist: {nvcc}")
    source_path = directory / "generated_pair_map_handler.cu"
    library_path = directory / "generated_pair_map_handler.so"
    include_directory = Path(jaxlib.__file__).resolve().parent / "include"
    source_path.write_text(source)
    subprocess.run(
        [
            str(nvcc),
            "-std=c++17",
            "-O3",
            f"-arch={architecture}",
            "-shared",
            "-Xcompiler",
            "-fPIC",
            "-I",
            str(include_directory),
            str(source_path),
            "-o",
            str(library_path),
            "-lcublas",
        ],
        check=True,
    )
    return ctypes.CDLL(str(library_path))


def _register_legacy_custom_call(library: ctypes.CDLL) -> None:
    xla_client = importlib.import_module("jaxlib.xla_client")
    symbol = getattr(library, _TARGET_NAME.replace(".", "_"))
    capsule_new = ctypes.pythonapi.PyCapsule_New
    capsule_new.restype = ctypes.py_object
    capsule_new.argtypes = (ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p)
    capsule = capsule_new(ctypes.cast(symbol, ctypes.c_void_p), b"xla._CUSTOM_CALL_TARGET", None)
    xla_client.register_custom_call_target(_TARGET_NAME, capsule, platform="cpu", api_version=0)


def _register_ffi_custom_call(library: ctypes.CDLL) -> None:
    handler = getattr(library, _TARGET_NAME.replace(".", "_"))
    handler.restype = ctypes.c_void_p
    jax.ffi.register_ffi_target(
        _TARGET_NAME,
        jax.ffi.pycapsule(handler),
        platform="cpu",
        api_version=1,
    )


def _register_cuda_ffi_custom_call(library: ctypes.CDLL) -> None:
    handler = getattr(library, _TARGET_NAME.replace(".", "_"))
    handler.restype = ctypes.c_void_p
    jax.ffi.register_ffi_target(
        _TARGET_NAME,
        jax.ffi.pycapsule(handler),
        platform="CUDA",
        api_version=1,
    )


def run_smoke(seed: int, artifact_directory: Path | None = None) -> dict[str, Any]:
    """Compile baseline and transformed paths, then compare their executions."""
    hlo = importlib.import_module("jaxlib._hlo")
    xla = importlib.import_module("jax.extend.xla")
    jax.config.update("jax_enable_compilation_cache", False)
    shapes = (
        jax.ShapeDtypeStruct((4, 8), jnp.float32),
        jax.ShapeDtypeStruct((8, 16), jnp.float32),
        jax.ShapeDtypeStruct((8, 16), jnp.float32),
        jax.ShapeDtypeStruct((16, 6), jnp.float32),
    )
    baseline_lowered = jax.jit(natural_program).lower(*shapes)
    baseline = baseline_lowered.compile()
    rng = np.random.default_rng(seed)
    inputs = tuple(jnp.asarray(rng.normal(size=shape.shape), dtype=jnp.float32) for shape in shapes)
    expected = np.asarray(baseline(*inputs))

    original_modules: list[str] = []
    transformed_modules: list[str] = []
    recovered_programs: list[FixedShapeProgram] = []
    temporary = None
    if artifact_directory is None:
        temporary = tempfile.TemporaryDirectory(prefix="shuttle-xla-custom-call-")
        directory = Path(temporary.name)
    else:
        artifact_directory.mkdir(parents=True, exist_ok=True)
        directory = artifact_directory
    try:
        holder: dict[str, Any] = {}

        def replace(serialized_module: bytes) -> bytes | None:
            module = hlo.HloModule.from_serialized_hlo_module_proto(serialized_module)
            if module.name != "jit_natural_program":
                return None
            original = module.to_string()
            original_modules.append(original)
            program = recover_fixed_shape_program(original)
            library = _compile_handler(generate_cpu_handler(program), directory)
            _register_legacy_custom_call(library)
            holder["library"] = library
            recovered_programs.append(program)
            transformed = _replace_entry_with_custom_call(original, _TARGET_NAME)
            transformed_module = hlo.hlo_module_from_text(transformed)
            transformed_modules.append(transformed_module.to_string())
            return transformed_module.as_serialized_hlo_module_proto()

        xla.register_hlo_module_transformation(
            replace,
            name=_PASS_NAME,
            stage=xla.PipelineStage.PRE_SCHEDULER,
            platforms="cpu",
        )
        jax.clear_caches()
        try:
            transformed = jax.jit(natural_program).lower(*shapes).compile()
        finally:
            xla.clear_hlo_module_transformation(
                _PASS_NAME,
                stage=xla.PipelineStage.PRE_SCHEDULER,
                platforms="cpu",
            )
        actual = np.asarray(transformed(*inputs))
        library = holder["library"]
        call_count_function = library.shuttle_pair_map_smoke_call_count
        call_count_function.restype = ctypes.c_int
        call_count = int(call_count_function())
        if artifact_directory is not None:
            (directory / "generated_pair_map_handler.so").unlink()
    finally:
        if temporary is not None:
            temporary.cleanup()

    if len(original_modules) != 1 or len(recovered_programs) != 1 or len(transformed_modules) != 1:
        raise RuntimeError("expected exactly one structural recovery and replacement")
    difference = np.abs(actual - expected)
    if not np.allclose(actual, expected, rtol=2e-5, atol=2e-5):
        raise RuntimeError(f"transformed output mismatch: max_abs={float(difference.max())}")
    transformed_hlo = transformed_modules[0]
    if _TARGET_NAME not in transformed_hlo or call_count < 1:
        raise RuntimeError("custom-call replacement did not execute")
    program = recovered_programs[0]
    generated_source = generate_cpu_handler(program)
    if artifact_directory is not None:
        write_gzip_text(artifact_directory / "original-pre-scheduler-hlo.txt.gz", original_modules[0])
        write_gzip_text(artifact_directory / "transformed-pre-scheduler-hlo.txt.gz", transformed_hlo)
    return {
        "kind": "xla_pre_scheduler_pair_map_custom_call_smoke",
        "jax_version": jax.__version__,
        "jaxlib_version": jaxlib.__version__,
        "platform": "cpu",
        "device_kind": jax.devices("cpu")[0].device_kind,
        "natural_frontend": "two dot_general, scalar tanh/multiply, consumer dot_general",
        "recovery": "opcode/shape/dependency-only recover_pair_map_regions",
        "recovered_dimensions": {
            "rows": program.rows,
            "reduction": program.reduction,
            "features": program.features,
            "outputs": program.outputs,
        },
        "generated_scalar_expression": program.scalar_expression,
        "generated_handler_sha256": hashlib.sha256(generated_source.encode()).hexdigest(),
        "custom_call_target": _TARGET_NAME,
        "custom_call_occurrences_in_transformed_hlo": transformed_hlo.count(_TARGET_NAME),
        "custom_call_handler_executions": call_count,
        "maximum_absolute_error": float(difference.max()),
        "mean_absolute_error": float(difference.mean()),
        "outputs_match": True,
        "explicit_warning": (
            "Disposable proof only: legacy CPU custom-call ABI and whole-module HLO text round trip; "
            "never use this rewrite mechanism on the frozen Grug module."
        ),
        "production_blockers": (
            "typed C++ HLO mutation/custom-call insertion",
            "buffer aliasing and sharding preservation",
            "generic FFI ABI and executable-plan serialization",
            "region-local replacement inside multi-output training graphs",
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--artifact-directory", type=Path)
    args = parser.parse_args()
    result = run_smoke(args.seed, args.artifact_directory)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.write_text(rendered)
    elif args.artifact_directory is not None:
        (args.artifact_directory / "summary.json").write_text(rendered)
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
