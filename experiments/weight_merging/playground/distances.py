# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Stream model-wide distance and BF16-rounding diagnostics using FP64 reductions."""

import argparse
import hashlib
import json
from contextlib import ExitStack
from pathlib import Path

import torch
from marin.merging.geometry import weight_update_gram
from safetensors import safe_open

from experiments.weight_merging.playground.checkpoints import checkpoint_map

BLOCK_SIZE = 1_000_000


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchor", type=Path, required=True)
    parser.add_argument("--donor", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    anchor_map, donor_map = checkpoint_map(args.anchor), checkpoint_map(args.donor)
    if set(anchor_map) != set(donor_map):
        raise ValueError("Checkpoint keys differ")
    sums = {
        "base_sq": 0.0,
        "donor_sq": 0.0,
        "delta_sq": 0.0,
        "dot": 0.0,
        "rounding_error_sq": 0.0,
        "rounded_delta_sq": 0.0,
        "active": 0,
        "active_rounded_to_zero": 0,
        "elements": 0,
    }
    gram = torch.zeros((3, 3), dtype=torch.float64)
    for name in sorted(anchor_map):
        with ExitStack() as stack:
            left = stack.enter_context(safe_open(args.anchor / anchor_map[name], framework="pt"))
            right = stack.enter_context(safe_open(args.donor / donor_map[name], framework="pt"))
            base, donor = left.get_tensor(name).flatten(), right.get_tensor(name).flatten()
            gram += weight_update_gram(base, [donor], chunk_elements=BLOCK_SIZE)
            for start in range(0, base.numel(), BLOCK_SIZE):
                a = base[start : start + BLOCK_SIZE].double()
                b = donor[start : start + BLOCK_SIZE].double()
                delta = b - a
                rounded = b.bfloat16().double() - a
                sums["rounding_error_sq"] += ((rounded - delta).square()).sum().item()
                sums["rounded_delta_sq"] += rounded.square().sum().item()
                sums["active"] += (delta != 0).sum().item()
                sums["active_rounded_to_zero"] += ((delta != 0) & (rounded == 0)).sum().item()
                sums["elements"] += delta.numel()
    sums.update(base_sq=gram[0, 0].item(), donor_sq=gram[1, 1].item(), delta_sq=gram[2, 2].item(), dot=gram[0, 1].item())
    result = {
        "anchor": str(args.anchor),
        "donor": str(args.donor),
        "reductions": "float64",
        **sums,
        "relative_l2": (sums["delta_sq"] / sums["base_sq"]) ** 0.5,
        "cosine": sums["dot"] / (sums["base_sq"] * sums["donor_sq"]) ** 0.5,
        "rounding_error_over_delta": (sums["rounding_error_sq"] / sums["delta_sq"]) ** 0.5,
    }
    hashes = {}
    for filename in sorted(set(donor_map.values())):
        with (args.donor / filename).open("rb") as stream:
            hashes[filename] = hashlib.file_digest(stream, "sha256").hexdigest()
    result["donor_sha256"] = hashes
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
