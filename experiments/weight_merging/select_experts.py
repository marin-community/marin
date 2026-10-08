# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Create individual-expert transfer recipes and matched random controls."""

import argparse
import copy
import hashlib
import json
import random
from pathlib import Path


def build_recipes(
    comparison: dict, template: dict, counts: list[int], seed: int, benchmark: str, recipient: str
) -> tuple[dict, dict]:
    """Rank experts independently by layer and preserve the recipient elsewhere."""
    scores = comparison["phases"]["generated"]["selection_score"]
    if len(scores) != 26 or any(len(layer) != 256 for layer in scores):
        raise ValueError("Expected 26 layers of 256 expert scores")
    ranked = [sorted(range(256), key=lambda expert: (-layer[expert], expert)) for layer in scores]
    randomized = [random.Random(seed + layer).sample(range(256), 256) for layer in range(26)]
    base = copy.deepcopy(template)
    base["parameters"]["coefficients"] = [0.0, 1.0]
    base["tensor_coefficients"] = {}
    base["row_overrides"] = {}
    output_root = template["output"].rsplit("/", 1)[0]
    recipes = {}
    selections = {"seed": seed, "randomization": "Uniform permutation of 256 IDs using seed + layer", "variants": {}}
    base_name = f"routing-base-{recipient}"
    base["output"] = f"{output_root}/{base_name}"
    recipes[base_name] = base
    for count in counts:
        if not 0 < count <= 256:
            raise ValueError("Expert counts must be between 1 and 256")
        for method, ordering in ((benchmark, ranked), ("random", randomized)):
            name = f"routing-{method}{count:03d}-{recipient}"
            selected = [sorted(layer[:count]) for layer in ordering]
            if method == benchmark and any(
                scores[layer][expert] <= 0 for layer, experts in enumerate(selected) for expert in experts
            ):
                raise ValueError("Targeted selection includes experts without positive benchmark enrichment")
            recipe = copy.deepcopy(base)
            recipe["output"] = f"{output_root}/{name}"
            for layer, experts in enumerate(selected):
                for projection in ("gate_proj", "up_proj", "down_proj"):
                    key = f"model.layers.{layer}.mlp.experts.{projection}.weight"
                    recipe["row_overrides"][key] = [{"row": expert, "coefficients": [1.0, 0.0]} for expert in experts]
            recipes[name] = recipe
            selections["variants"][name] = {"experts_per_layer": count, "selected_ids": selected}
    return recipes, selections


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--counts", type=int, nargs="+", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--expected-requests", type=int, required=True)
    parser.add_argument("--benchmark", choices=("nupa", "bfcl"), required=True)
    parser.add_argument("--recipient", choices=("step20", "step92"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw = args.comparison.read_bytes()
    comparison = json.loads(raw)
    if sum(comparison["replay_requests"]) != args.expected_requests:
        raise ValueError("Replay coverage does not match the final capture inventory")
    recipes, selections = build_recipes(
        comparison, json.loads(args.template.read_text()), args.counts, args.seed, args.benchmark, args.recipient
    )
    args.output.mkdir(parents=True, exist_ok=False)
    for name, recipe in recipes.items():
        (args.output / f"{name}.json").write_text(json.dumps(recipe, indent=2) + "\n")
    selections.update(
        comparison_sha256=hashlib.sha256(raw).hexdigest(),
        ranking=comparison["ranking"],
        limitations=comparison["limitations"],
        replay_requests=comparison["replay_requests"],
    )
    (args.output / "selection.json").write_text(json.dumps(selections, indent=2) + "\n")


if __name__ == "__main__":
    main()
