# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Check whether two identical Qwen jobs reproduce native training requests."""

from dataclasses import replace

from skyrlrecipe import RecipePatch, sections

from experiments.post_training import rl
from experiments.post_training.qwen_timing import QWEN_TIMING, STRING_TASKS, timing_preset

RELEASE = "2026.11.26.6"
PRESET_NAMES = ("east02-paired-seed-a", "east02-paired-seed-b")
SEED = 137
UPDATES = 4
RETENTION_BYTES = 32 * 1024**2

base = timing_preset("string-ei", STRING_TASKS, SEED).with_updates(UPDATES)
config = base.recipe.config.merge(
    RecipePatch(
        data=sections.Data(sampling=sections.Sampling(paired_rollout_seed=SEED)),
        generator=sections.Generator(
            trajectory_retention=sections.TrajectoryRetention(
                enabled=True,
                phases=("train",),
                sample_count_per_step=0,
                sample_fraction=1.0,
                max_bytes_per_step=RETENTION_BYTES // UPDATES,
                max_bytes_per_run=RETENTION_BYTES,
                required=True,
            )
        ),
    )
)
SMOKE = replace(
    QWEN_TIMING,
    presets=tuple(
        replace(base, name=name, revision=RELEASE, recipe=replace(base.recipe, config=config)) for name in PRESET_NAMES
    ),
)

if __name__ == "__main__":
    raise SystemExit(rl.main(SMOKE))
