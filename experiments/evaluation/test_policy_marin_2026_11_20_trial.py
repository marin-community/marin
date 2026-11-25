# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The 2026-11-20 trial policy shows what each candidate runs with."""

from experiments.evaluation.policy.cli import policy_rows
from experiments.evaluation.policy.versions.marin_2026_11_20_trial.policy import POLICY


def test_show_prints_the_thinking_arguments_each_candidate_is_sent():
    # Glimmer replaces the standard thinking-off switch with its own argument, as the worker sends it.
    think = {
        candidate.name: next(row.think for row in policy_rows(POLICY, candidate) if row.evaluation == "gsm8k-0shot")
        for candidate in POLICY.candidates
    }
    assert think == {
        "muse-glimmer-30b": "reasoning_strength=low",
        "nvidia-nemotron-3.5-super-vl-120b-a12b-bf16": "off",
        "qwen3.6-35b-a3b": "off",
    }
