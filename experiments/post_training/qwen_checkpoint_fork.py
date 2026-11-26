# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Prepare one balanced, development-only source for a Qwen allocation fork."""

from dataclasses import replace

from skyrlrecipe import RecipePatch, sections

from experiments.post_training import rl
from experiments.post_training.qwen_probe_curriculum import BLOCK_TOKENS, DEVELOPMENT, TRAIN
from experiments.post_training.qwen_sampler import EVAL, LAYOUT, WITHIN_SOURCE_PASS, budget_preset
from experiments.post_training.rl_presets import ModelPresets

SEED = 157
REVISION = "2026.11.26.5"
WARMUP_NAME = "warmup"


def warmup_preset():
    """Stop at the first 1 Mi-token crossing and retain the complete native state."""
    preset = budget_preset(
        WARMUP_NAME,
        WITHIN_SOURCE_PASS,
        1.0,
        SEED,
        revision=REVISION,
        layout=LAYOUT,
        max_updates=48,
        tokens=BLOCK_TOKENS,
        evaluation=replace(EVAL, name="development"),
        held_out_tasks=DEVELOPMENT,
    )
    config = preset.recipe.config.merge(
        RecipePatch(data=sections.Data(sampling=sections.Sampling(domain_weights={"easy": 0.5, "intermediate": 0.5})))
    )
    return replace(
        preset,
        tasks=TRAIN,
        checkpoints="latest",
        accepts_settings=False,
        recipe=replace(preset.recipe, config=config, layout=replace(preset.recipe.layout, job_timeout_seconds=2_400)),
    )


QWEN_WARMUP = ModelPresets(
    model="qwen3-0.6b",
    namespace="qwen-checkpoint-fork",
    presets=(warmup_preset(),),
    control_pins={},
)


if __name__ == "__main__":
    raise SystemExit(rl.main(QWEN_WARMUP))
