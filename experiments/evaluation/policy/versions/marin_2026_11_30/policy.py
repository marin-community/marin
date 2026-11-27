# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The November 2026 eval policy: Mochi's comparison models on the in-distribution benchmarks of #12156.

Benchmarks own their prompts, thinking mode, repeats, output caps, metrics and any window they enforce;
``evaluations.yaml`` only chooses them and the settings each runs with. README.md lists the differences from
the October policy.
"""

from pathlib import Path

from experiments.evaluation.curated import load_evaluations
from experiments.evaluation.policy.schema import Policy

POLICY = Policy(
    version="marin-2026-11-30",
    baselines=(
        # Tier 0: Marin.
        "marin-snowball-67b-a2b-v0.1",
        # Tier 1: strong models in a smaller weight class.
        "muse-glimmer-30b",
        "nvidia-nemotron-3.5-super-vl-120b-a12b-bf16",
        "qwen3.8-27b",
        "gemma-4-31b-it",
        "granite-4.2-30b",
        # Tier 2: peer models from other open providers.
        "k2-horizon-375b-a23b",
        "command-a-plus-05-2026-bf16",
        "mistral-large-4-1t-a52b",
        "trinity-large-thinking",
        # Tier 3: the open frontier in Mochi's weight class.
        "beam-1.0-501b-a23b",
        "inkling-2-small",
        "glm-5.3-flash",
        "deepseek-v4.1-flash-1017",
    ),
    evaluations=load_evaluations(Path(__file__).with_name("evaluations.yaml")),
    # SWE-bench Pro's protocol: a 131,072-token agent input and 16,384-token replies. Glimmer and Granite serve at
    # their own 131,072.
    serving_window=147_456,
)
