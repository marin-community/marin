# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Strict/loose metrics using the original harness IFEval and AI2 IFBench checkers."""

from collections import defaultdict
from collections.abc import Mapping
from functools import lru_cache
from importlib import import_module
from typing import Any, Protocol, cast

from verifyit.grade import InvalidTask, Reward, scored
from verifyit.spec import Constraint

from .instruction_spec import InstructionProfile, InstructionSpec


class Instruction(Protocol):
    def __init__(self, instruction_id: str): ...
    def build_description(self, **kwargs: Any) -> str: ...
    def get_instruction_args(self) -> dict | None: ...
    def check_following(self, value: str) -> bool: ...


@lru_cache
def source_registry(profile: InstructionProfile) -> Mapping[str, type[Instruction]]:
    """Load optional checker dependencies only for their explicitly selected protocol."""
    modules = {
        InstructionProfile.HARNESS: f"{__package__}.ifeval.instructions_registry",
        InstructionProfile.IFBENCH: f"{__package__}.ifbench.instructions_registry",
    }
    return cast(Mapping[str, type[Instruction]], import_module(modules[profile]).INSTRUCTION_DICT)


def validate_source_instructions(spec: InstructionSpec) -> None:
    if not spec.constraints:
        raise InvalidTask("Instruction benchmarks require nonempty constraints")
    unknown = sorted({constraint.name for constraint in spec.constraints} - source_registry(spec.profile).keys())
    if unknown:
        raise InvalidTask(f"Unknown source instruction checkers: {unknown}")


def _instruction(spec: InstructionSpec, constraint: Constraint) -> Instruction:
    instruction = source_registry(spec.profile)[constraint.name](constraint.name)
    # The harness drops all falsey values; AI2 drops only None. Preserve both.
    params = {
        key: value
        for key, value in constraint.params.items()
        if (bool(value) if spec.profile is InstructionProfile.HARNESS else value is not None)
    }
    instruction.build_description(**params)
    args = instruction.get_instruction_args()
    if args and "prompt" in args:
        instruction.build_description(prompt=spec.prompt)
    return instruction


def loose_responses(response: str) -> tuple[str, ...]:
    lines = response.split("\n")
    first = "\n".join(lines[1:]).strip()
    last = "\n".join(lines[:-1]).strip()
    both = "\n".join(lines[1:-1]).strip()
    return (
        response,
        response.replace("*", ""),
        first,
        last,
        both,
        first.replace("*", ""),
        last.replace("*", ""),
        both.replace("*", ""),
    )


def grade_source_instructions(spec: InstructionSpec, response: str) -> Reward:
    """Retain per-instruction flags, weights and category evidence for later reduction."""
    validate_source_instructions(spec)
    metrics: dict[str, float] = {}
    weights: dict[str, int] = {}
    reducer_inputs: dict[str, bool | list[bool]] = {}
    instruction_pass: dict[str, list[bool]] = {}
    for mode in ("strict", "loose"):
        candidates = (response,) if mode == "strict" else loose_responses(response)
        flags = []
        for constraint in spec.constraints:
            instruction = _instruction(spec, constraint)
            flags.append(any(candidate.strip() and instruction.check_following(candidate) for candidate in candidates))
        instruction_pass[mode] = flags
        if spec.profile is InstructionProfile.HARNESS:
            prompt_name = f"prompt_level_{mode}_acc,none"
            instruction_name = f"inst_level_{mode}_acc,none"
        else:
            prompt_name = f"{mode}_prompt_accuracy"
            instruction_name = f"{mode}_instruction_accuracy"
        metrics[prompt_name] = float(all(flags))
        metrics[instruction_name] = sum(flags) / len(flags)
        weights[instruction_name] = len(flags)
        reducer_inputs[prompt_name] = all(flags)
        reducer_inputs[instruction_name] = flags
        if spec.profile is InstructionProfile.IFBENCH:
            grouped: dict[str, list[bool]] = defaultdict(list)
            for constraint, passed in zip(spec.constraints, flags, strict=True):
                grouped[constraint.name.split(":")[0]].append(passed)
            for category, values in grouped.items():
                name = f"{mode}_per_type:{category}"
                metrics[name] = sum(values) / len(values)
                weights[name] = len(values)
    return scored(
        float(all(instruction_pass["strict"])),
        metrics=metrics,
        metric_weights=weights,
        reducer_inputs=reducer_inputs,
        instruction_pass=instruction_pass,
        instruction_counts={
            "total": len(spec.constraints),
            **{mode + "_passed": sum(flags) for mode, flags in instruction_pass.items()},
        },
    )
