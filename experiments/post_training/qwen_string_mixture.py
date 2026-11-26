# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compare fixed two-template mixtures at the same gross response-token budgets."""

from dataclasses import replace

from skyrlrecipe import RecipePatch, sections

from experiments.post_training import rl
from experiments.post_training.qwen_sampler import EVAL, LAYOUT, WITHIN_SOURCE_PASS, budget_preset
from experiments.post_training.rl_presets import ModelPresets
from experiments.post_training.task_curation.qwen_string_mixture import DISCOVERY, TRAIN, VERSION

MIXTURES = {"balanced": (0.5, 0.5), "easy80": (0.8, 0.2), "odd80": (0.2, 0.8)}
SEEDS = (107, 109)
RELEASE = "2026.11.26.3"
MIDPOINT = 1_048_576
BUDGET = 2 * MIDPOINT


def static_preset(arm: str, seed: int):
    preset = budget_preset(
        f"static-{arm}-s{seed}",
        WITHIN_SOURCE_PASS,
        1.0,
        seed,
        revision=RELEASE,
        layout=LAYOUT,
        max_updates=128,
        tokens=BUDGET,
        evaluation=replace(EVAL, name="qwen_string_transfer"),
        held_out_tasks=f"{DISCOVERY}@{VERSION}",
    )
    easy, intermediate = MIXTURES[arm]
    config = preset.recipe.config.merge(
        RecipePatch(
            data=sections.Data(sampling=sections.Sampling(domain_weights={"easy": easy, "intermediate": intermediate})),
            trainer=sections.Trainer(rollout_token_eval_milestones=(MIDPOINT,)),
        )
    )
    return replace(preset, tasks=f"{TRAIN}@{VERSION}", recipe=replace(preset.recipe, config=config))


STUDY = ModelPresets(
    model="qwen3-0.6b",
    namespace="qwen-string",
    presets=tuple(static_preset(arm, seed) for arm in MIXTURES for seed in SEEDS),
    control_pins={},
)

if __name__ == "__main__":
    raise SystemExit(rl.main(STUDY))
