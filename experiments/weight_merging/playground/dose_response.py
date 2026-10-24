# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Match single-donor pruning controls by model-wide FP32 update norm."""

import argparse
import json
import math
from contextlib import ExitStack
from pathlib import Path

import torch
from safetensors import safe_open

from experiments.weight_merging.playground.arithmetic import Method, Recipe, keep_largest, merge
from experiments.weight_merging.playground.checkpoints import checkpoint_map

DENSITY = 0.3
COEFFICIENTS = (0.25, 0.5, 0.75)


def updates(delta: torch.Tensor, method: Method, name: str) -> torch.Tensor:
    """Use reference masks; magnitude pruning consumes delta and must run last."""
    if method == Method.AVERAGE:
        return delta
    if method == Method.TIES:
        return keep_largest(delta, DENSITY)
    return merge(torch.zeros_like(delta), [delta], Recipe(method, (1.0,), density=DENSITY), name)


def tensor_pairs(anchor: Path, donor: Path):
    left, right = checkpoint_map(anchor), checkpoint_map(donor)
    if set(left) != set(right):
        raise ValueError("Model tensor keys differ")
    for name in sorted(left):
        with ExitStack() as stack:
            a = stack.enter_context(safe_open(anchor / left[name], framework="pt")).get_tensor(name)
            b = stack.enter_context(safe_open(donor / right[name], framework="pt")).get_tensor(name)
            if a.shape != b.shape:
                raise ValueError(f"Model tensor shape differs: {name}")
            if a.is_floating_point():
                yield name, a, b
            elif not torch.equal(a, b):
                raise ValueError(f"Nonfloating tensors differ: {name}")


def squared_norm(tensor: torch.Tensor) -> float:
    flat = tensor.flatten()
    return sum(
        flat[start : start + 1_000_000].double().square().sum().item() for start in range(0, flat.numel(), 1_000_000)
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchor", type=Path, required=True)
    parser.add_argument("--donor", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    methods = (Method.AVERAGE, Method.DARE_LINEAR, Method.TIES)
    norms = dict.fromkeys(methods, 0.0)
    for name, a, b in tensor_pairs(args.anchor, args.donor):
        delta = b.float() - a.float()
        for method in methods:
            norms[method] += squared_norm(updates(delta, method, name))
    scales = {
        method: {c: c * math.sqrt(norms[Method.AVERAGE] / norms[method]) for c in COEFFICIENTS} for method in methods
    }
    candidates = [{"name": "instruct", "source": "instruct"}, {"name": "aristaeus", "source": "aristaeus"}]
    stats = {}
    for method in methods:
        label = "magnitude_topk" if method == Method.TIES else method.value
        for c in COEFFICIENTS:
            name = f"aristaeus-{label}-c{c}"
            scale = scales[method][c]
            recipe = {
                "method": method.value,
                "coefficients": [c] if method == Method.AVERAGE else [1.0],
                "density": 1.0 if method == Method.AVERAGE else DENSITY,
                "scale": 1.0 if method == Method.AVERAGE else scale,
                "seed": 42,
                "ties_density": 1.0,
            }
            candidates.append({"name": name, "source": "instruct", "donors": ["aristaeus"], "recipe": recipe})
            stats[name] = {
                "coefficient_equivalent": c,
                "unit_update_sq": norms[method],
                "scale": scale,
                "fp32_update_sq": 0.0,
                "bf16_update_sq": 0.0,
                "cast_error_sq": 0.0,
                "changed_fp32": 0,
                "changed_bf16": 0,
            }
    for tensor_name, a, b in tensor_pairs(args.anchor, args.donor):
        delta = b.float() - a.float()
        for method in methods:
            update = updates(delta, method, tensor_name).flatten()
            label = "magnitude_topk" if method == Method.TIES else method.value
            base, donor = a.flatten(), b.flatten()
            for c in COEFFICIENTS:
                metrics = stats[f"aristaeus-{label}-c{c}"]
                for start in range(0, update.numel(), 1_000_000):
                    end = start + 1_000_000
                    original = base[start:end].float()
                    if method == Method.AVERAGE:
                        target = (original * (1 - c)).add_(donor[start:end].float(), alpha=c)
                    else:
                        target = (update[start:end] * scales[method][c]).add_(original)
                    cast = target.bfloat16().double()
                    target, original = target.double(), original.double()
                    metrics["fp32_update_sq"] += (target - original).square().sum().item()
                    metrics["bf16_update_sq"] += (cast - original).square().sum().item()
                    metrics["cast_error_sq"] += (cast - target).square().sum().item()
                    metrics["changed_fp32"] += (target != original).sum().item()
                    metrics["changed_bf16"] += (cast != original).sum().item()
    for metrics in stats.values():
        metrics["fp32_norm_relative_to_full_donor"] = math.sqrt(metrics["fp32_update_sq"] / norms[Method.AVERAGE])
        metrics["bf16_norm_relative_to_full_donor"] = math.sqrt(metrics["bf16_update_sq"] / norms[Method.AVERAGE])
    args.output.write_text(json.dumps(candidates, indent=2) + "\n")
    args.metrics.write_text(
        json.dumps({"density": DENSITY, "seed": 42, "unit_norms_sq": norms, "candidates": stats}, indent=2) + "\n"
    )
    print(json.dumps({"unit_norms_sq": norms, "scales": scales}, indent=2), flush=True)


if __name__ == "__main__":
    main()
