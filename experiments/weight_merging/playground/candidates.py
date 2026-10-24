# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Candidate recipes and CPU tensor streaming, with separate donor/output rounding."""

import json
from contextlib import ExitStack
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from safetensors import safe_open

from experiments.weight_merging.playground.arithmetic import Method, Recipe, merge
from experiments.weight_merging.playground.checkpoints import checkpoint_map


class WeightPrecision(StrEnum):
    FP32 = "float32"
    BF16_ROUNDED = "bfloat16_rounded"


@dataclass(frozen=True)
class Candidate:
    name: str
    source: str
    donors: tuple[str, ...] = ()
    recipe: Recipe | None = None
    rope_theta: float = 1_000_000
    donor_precision: WeightPrecision = WeightPrecision.FP32
    output_precision: WeightPrecision = WeightPrecision.FP32


def read_candidates(path: Path) -> list[Candidate]:
    result = []
    for value in json.loads(path.read_text()):
        recipe = value.get("recipe")
        if recipe is not None:
            recipe = Recipe(
                **{**recipe, "method": Method(recipe["method"]), "coefficients": tuple(recipe["coefficients"])}
            )
        result.append(
            Candidate(
                name=value["name"],
                source=value["source"],
                donors=tuple(value.get("donors", [])),
                recipe=recipe,
                rope_theta=value.get("rope_theta", 1_000_000),
                donor_precision=WeightPrecision(value.get("donor_precision", "float32")),
                output_precision=WeightPrecision(value.get("output_precision", "float32")),
            )
        )
    if len({candidate.name for candidate in result}) != len(result):
        raise ValueError("Duplicate candidate names")
    return result


def merged_weights(paths: dict[str, Path], candidate: Candidate):
    """Yield merged CPU tensors to vLLM's architecture-aware weight loader."""
    sources = [candidate.source, *candidate.donors]
    maps = {name: checkpoint_map(paths[name]) for name in sources}
    expected_keys = set(maps[sources[0]])
    if any(set(mapping) != expected_keys for mapping in maps.values()):
        raise ValueError("Source checkpoint keys differ")
    for name in sorted(expected_keys):
        with ExitStack() as stack:
            tensors = []
            for source in sources:
                reader = stack.enter_context(safe_open(paths[source] / maps[source][name], framework="pt"))
                tensor = reader.get_tensor(name)
                if tensor.is_floating_point():
                    if source != sources[0] and candidate.donor_precision == WeightPrecision.BF16_ROUNDED:
                        tensor = tensor.bfloat16()
                    tensor = tensor.float()
                tensors.append(tensor)
                del tensor
            value = merge(tensors[0], tensors[1:], candidate.recipe, name) if candidate.recipe else tensors[0]
            if value.is_floating_point() and candidate.output_precision == WeightPrecision.BF16_ROUNDED:
                value = value.bfloat16().float()
            yield name, value
            del value, tensors


