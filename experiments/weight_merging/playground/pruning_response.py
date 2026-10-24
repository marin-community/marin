# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Select pruning densities by retained squared L2, then match update size."""

import argparse
import json
import math
from pathlib import Path

import torch

from experiments.weight_merging.playground.arithmetic import Method, Recipe, keep_largest, merge
from experiments.weight_merging.playground.dose_response import squared_norm, tensor_pairs

DENSITIES = tuple(10 ** (-6 + 6 * index / 240) for index in range(241))
TARGETS = (0.9, 0.7, 0.5)


def retained_energy(delta: torch.Tensor, densities: tuple[float, ...]) -> list[float]:
    """Exact per-tensor top-k energy; tied entries have equal squared magnitude."""
    values = delta.flatten().abs()
    positive = values[values != 0]
    if not positive.numel():
        return [0.0] * len(densities)
    ordered = positive.sort(descending=True).values.double().square().cumsum(0)
    return [ordered[min(ordered.numel(), max(1, math.ceil(d * delta.numel()))) - 1].item() for d in densities]


def unit_update(delta: torch.Tensor, method: Method, density: float, name: str) -> torch.Tensor:
    if method == Method.TIES:
        return keep_largest(delta.clone(), density)
    return merge(torch.zeros_like(delta), [delta], Recipe(method, (1.0,), density=density), name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchor", type=Path, required=True)
    parser.add_argument("--donor", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    energy = [0.0] * len(DENSITIES)
    for _, anchor, donor in tensor_pairs(args.anchor, args.donor):
        local = retained_energy(donor.float() - anchor.float(), DENSITIES)
        energy = [left + right for left, right in zip(energy, local, strict=True)]
    total = energy[-1]
    if total <= 0:
        raise ValueError("Donor has no update")
    selected = {target: min(range(len(energy)), key=lambda i: abs(energy[i] / total - target)) for target in TARGETS}
    unit_norms = {}
    for name, anchor, donor in tensor_pairs(args.anchor, args.donor):
        delta = donor.float() - anchor.float()
        for target, index in selected.items():
            for method in (Method.TIES, Method.DARE_LINEAR):
                key = f"{method.value}-{target}"
                value = squared_norm(unit_update(delta, method, DENSITIES[index], name))
                unit_norms[key] = unit_norms.get(key, 0.0) + value
    candidates = [{"name": "instruct", "source": "instruct"}, {"name": "aristaeus", "source": "aristaeus"}]
    stats = {}
    for target, index in selected.items():
        density = DENSITIES[index]
        for method in (Method.TIES, Method.DARE_LINEAR):
            scale = 0.5 * math.sqrt(total / unit_norms[f"{method.value}-{target}"])
            label = "magnitude_topk" if method == Method.TIES else method.value
            name = f"aristaeus-{label}-energy{target}-c0.5"
            candidates.append(
                {
                    "name": name,
                    "source": "instruct",
                    "donors": ["aristaeus"],
                    "recipe": {
                        "method": method.value,
                        "coefficients": [1.0],
                        "density": density,
                        "scale": scale,
                        "seed": 42,
                        "ties_density": 1.0,
                    },
                }
            )
            stats[name] = {
                "target_retained_energy": target,
                "density": density,
                "topk_retained_energy": energy[index] / total,
                "scale": scale,
                "unit_update_sq": unit_norms[f"{method.value}-{target}"],
                "fp32_update_sq": 0.0,
                "bf16_update_sq": 0.0,
                "cast_error_sq": 0.0,
            }
    # Measure actual arithmetic and output casting, including FP32 addition rounding.
    for tensor_name, anchor, donor in tensor_pairs(args.anchor, args.donor):
        for candidate in candidates[2:]:
            recipe = candidate["recipe"]
            update = unit_update(
                donor.float() - anchor.float(), Method(recipe["method"]), recipe["density"], tensor_name
            )
            original = anchor.flatten()
            update = update.flatten()
            metrics = stats[candidate["name"]]
            for start in range(0, update.numel(), 1_000_000):
                base = original[start : start + 1_000_000].float()
                result = (update[start : start + 1_000_000] * recipe["scale"]).add_(base)
                cast = result.bfloat16().double()
                base, result = base.double(), result.double()
                metrics["fp32_update_sq"] += (result - base).square().sum().item()
                metrics["bf16_update_sq"] += (cast - base).square().sum().item()
                metrics["cast_error_sq"] += (cast - result).square().sum().item()
    for metrics in stats.values():
        metrics["fp32_norm_relative_to_full_donor"] = math.sqrt(metrics["fp32_update_sq"] / total)
        metrics["bf16_norm_relative_to_full_donor"] = math.sqrt(metrics["bf16_update_sq"] / total)
    args.output.write_text(json.dumps(candidates, indent=2) + "\n")
    args.metrics.write_text(
        json.dumps(
            {
                "seed": 42,
                "full_update_sq": total,
                "scan": [{"density": d, "retained_energy": e / total} for d, e in zip(DENSITIES, energy, strict=True)],
                "candidates": stats,
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps(stats, indent=2), flush=True)


if __name__ == "__main__":
    main()
