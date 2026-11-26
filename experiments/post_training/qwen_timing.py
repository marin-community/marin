# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Time fresh random Qwen training prompts from two fixed East02 pools."""

from dataclasses import replace

from skyrlrecipe import RecipePatch, sections

from experiments.post_training import rl
from experiments.post_training.qwen_sampler import LAYOUT, TASKS, WITHIN_SOURCE_PASS, sampler_preset
from experiments.post_training.rl_presets import ModelPresets, Preset

RELEASE = "2026.11.26.2"
STRING_TASKS = "qwen-timing-string-ei@2026.11.26.1"
POOLS = (("full", TASKS), ("string-ei", STRING_TASKS))
SEEDS = (101, 103)
UPDATES = 12


def timing_preset(pool: str, tasks: str, seed: int) -> Preset:
    # Epsilon 1 keeps the source-pass draws uniform within source and retains paid lease telemetry.
    preset = sampler_preset(
        f"east02-timing-{pool}-seed{seed}",
        UPDATES,
        WITHIN_SOURCE_PASS,
        1.0,
        seed,
        layout=LAYOUT,
        revision=RELEASE,
        evaluation_interval=None,
    )
    # The native dataset loader opens configured validation paths even when evaluation is disabled.
    config = preset.recipe.config.merge(RecipePatch(data=sections.Data(val_data=())))
    return replace(
        preset,
        recipe=replace(preset.recipe, config=config),
        tasks=tasks,
        evaluations=(),
        held_out_tasks=None,
        accepts_settings=False,
    )


QWEN_TIMING = ModelPresets(
    model="qwen3-0.6b",
    namespace="qwen-timing",
    presets=tuple(timing_preset(pool, tasks, seed) for pool, tasks in POOLS for seed in SEEDS),
    control_pins={},
)


if __name__ == "__main__":
    raise SystemExit(rl.main(QWEN_TIMING))
