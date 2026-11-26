# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""A trial eval policy for exercising the policy launcher.

It is the closest approximation of the October 2026 policy that ``main`` runs. Benchmarks own their prompts,
thinking mode, repeats, output caps, metrics and any window they enforce; ``evaluations.yaml`` only chooses them
and the settings each runs with. README.md lists the differences from October.
"""

from pathlib import Path

from experiments.evaluation.curated import load_evaluations
from experiments.evaluation.policy.schema import Policy

POLICY = Policy(
    version="marin-2026-11-20-trial",
    models=("muse-glimmer-30b", "nvidia-nemotron-3.5-super-vl-120b-a12b-bf16", "qwen3.6-35b-a3b"),
    evaluations=load_evaluations(Path(__file__).with_name("evaluations.yaml")),
    # October served every model at this window.
    serving_window=73_728,
)

# Not declared:
# - τ³-bench and SOTOPIA-hard: their Harbor agents were removed from main.
# - BixBench: its package grader asks a MiniMax-M3 judge served beside the model, which the policy does not
#   declare yet; it joins once the policy launcher can start a judge for one benchmark.
