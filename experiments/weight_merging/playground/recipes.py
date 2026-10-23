# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Write explicit recipes for matched near donors or sparse far-donor controls."""

import argparse
import json
from pathlib import Path


def recipe(method: str, coefficients: list[float], density: float = 1, scale: float = 1) -> dict:
    return {
        "method": method,
        "coefficients": coefficients,
        "density": density,
        "scale": scale,
        "seed": 42,
        "ties_density": 1,
    }


def near_grid() -> list[dict]:
    result = [{"name": name, "source": name} for name in ("instruct", "math_sft", "code_sft")]
    for coefficient in (0.25, 0.5):
        result.append(
            {
                "name": f"near-average-c{coefficient}",
                "source": "instruct",
                "donors": ["math_sft", "code_sft"],
                "recipe": recipe("average", [coefficient] * 2),
            }
        )
    for scale in (0.5, 1):
        result.append(
            {
                "name": f"near-task_arithmetic-s{scale}",
                "source": "instruct",
                "donors": ["math_sft", "code_sft"],
                "recipe": recipe("task_arithmetic", [0.5, 0.5], scale=scale),
            }
        )
    for method in ("ties", "dare_linear", "dare_ties"):
        for density in (0.1, 0.3, 0.5, 0.7):
            for scale in (0.5, 1):
                result.append(
                    {
                        "name": f"near-{method}-d{density}-s{scale}",
                        "source": "instruct",
                        "donors": ["math_sft", "code_sft"],
                        "recipe": recipe(method, [0.5, 0.5], density, scale),
                    }
                )
    result.append(
        {
            "name": "near-ties-d0.3-s1-bf16-donors",
            "source": "instruct",
            "donors": ["math_sft", "code_sft"],
            "donor_precision": "bfloat16_rounded",
            "recipe": recipe("ties", [0.5, 0.5], 0.3, 1),
        }
    )
    return result


def far_controls() -> list[dict]:
    result = [
        {"name": "instruct", "source": "instruct"},
        {"name": "coder", "source": "coder"},
        {"name": "math-native", "source": "math", "rope_theta": 10_000},
        {"name": "math-anchor-rope", "source": "math"},
    ]
    for donor in ("coder", "math"):
        for coefficient in (0.1, 0.3, 0.5):
            result.append(
                {
                    "name": f"far-{donor}-average-c{coefficient}",
                    "source": "instruct",
                    "donors": [donor],
                    "recipe": recipe("average", [coefficient]),
                }
            )
        result.append(
            {
                "name": f"far-{donor}-ties-d0.3-s0.3",
                "source": "instruct",
                "donors": [donor],
                "recipe": recipe("ties", [1], 0.3, 0.3),
            }
        )
        result.append(
            {
                "name": f"far-{donor}-dare_linear-d0.3-c0.3",
                "source": "instruct",
                "donors": [donor],
                "recipe": recipe("dare_linear", [0.3], 0.3),
            }
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=["near", "far"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    values = near_grid() if args.arm == "near" else far_controls()
    args.output.write_text(json.dumps(values, indent=2) + "\n")


if __name__ == "__main__":
    main()
