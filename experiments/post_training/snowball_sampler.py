# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compare static and feedback-weighted task sampling on a shared Snowball selection."""

from dataclasses import replace

from skyrlrecipe import RecipePatch, sections

from experiments.post_training import rl
from experiments.post_training.rl_presets import ModelPresets, Preset
from experiments.post_training.snowball_rl import DATAKIT_SFT_2026_09_21, SNOWBALL
from experiments.post_training.task_curation.snowball_sampler_selection import EVAL_LITE, PILOT


def sampler_preset(stage: str, updates: int, arm: str, exploration: float) -> Preset:
    base = SNOWBALL.by_name["train-long"].with_updates(updates)
    sampling = sections.Sampling(
        kind="adaptive-source-thompson" if arm == "thompson" else "adaptive-source",
        source_column="family",
        domain_weights=dict.fromkeys(PILOT.family_limits, 1.0),
        decay=0.95,
        epsilon=exploration,
        prior_alpha=1.0,
        prior_beta=1.0,
    )
    # Leave dispatches after feedback arrives even in the four-update smoke.
    settings = RecipePatch(
        data=sections.Data(sampling=sampling),
        trainer=sections.Trainer(rollout_buffer=sections.RolloutBuffer(max_in_flight=32)),
    )
    recipe = replace(base.recipe, config=base.recipe.config.merge(settings))
    # All arms score identical prompts at fixed optimizer steps, including the initializer.
    cadence = updates // 2
    return replace(
        base,
        name=f"{stage}-{arm}",
        revision=PILOT.version,
        recipe=recipe,
        tasks=f"{PILOT.name}@{PILOT.version}",
        tasks_input="forbidden",
        model=DATAKIT_SFT_2026_09_21,
        accepts_init=False,
        held_out_tasks=f"eval-lite@{EVAL_LITE.version}",
        evaluations=(replace(base.evaluations[0], every=cadence),),
        checkpoints="latest",
        checkpoint_every=cadence,
        export="latest",
    )


SAMPLER = ModelPresets(
    model=SNOWBALL.model,
    namespace="snowball-sampler",
    presets=(
        sampler_preset("smoke", 4, "fixed", 1.0),
        sampler_preset("smoke", 4, "adaptive", 0.2),
        sampler_preset("pilot", 20, "fixed", 1.0),
        sampler_preset("pilot", 20, "adaptive", 0.2),
        sampler_preset("pilot", 20, "thompson", 0.05),
    ),
    control_pins={},
)

if __name__ == "__main__":
    raise SystemExit(rl.main(SAMPLER))
