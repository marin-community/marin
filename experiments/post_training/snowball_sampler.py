# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compare fixed and learned prompt orders on the four-source Snowball selection."""

from dataclasses import replace

from skyrlrecipe import RecipePatch, sections

from experiments.post_training import rl
from experiments.post_training.rl_presets import ModelPresets, Preset
from experiments.post_training.snowball_rl import DATAKIT_SFT_2026_09_21, SEED, SNOWBALL
from experiments.post_training.task_curation.snowball_sampler_selection import EVAL_LITE, PILOT

REVISION = "2026.11.25.2"
SECOND_SEED = 20261021
PROMPT_EXPLORATION = 0.05
MATCHED_ARMS = (
    ("fixed", "adaptive-source", 1.0),
    ("source-only", "prompt-thompson-source", PROMPT_EXPLORATION),
    ("contextual", "prompt-thompson-contextual", PROMPT_EXPLORATION),
    ("cost-aware", "prompt-thompson-cost-aware", PROMPT_EXPLORATION),
)
SOURCE_ADAPTIVE = ("adaptive", "adaptive-source", 0.20)
SOURCE_THOMPSON = ("thompson", "adaptive-source-thompson", 0.05)


def sampler_preset(name: str, updates: int, kind: str, epsilon: float, seed: int) -> Preset:
    base = SNOWBALL.by_name["train-long"].with_updates(updates)
    sampling = sections.Sampling(
        kind=kind,
        source_column="family",
        domain_weights=dict.fromkeys(PILOT.family_limits, 1.0),
        decay=0.95,
        epsilon=epsilon,
        prior_alpha=1.0,
        prior_beta=1.0,
    )
    # Leave dispatches after feedback arrives even in the four-update smoke.
    settings = RecipePatch(
        data=sections.Data(sampling=sampling),
        trainer=sections.Trainer(rollout_buffer=sections.RolloutBuffer(max_in_flight=32)),
    )
    recipe = replace(base.recipe, config=base.recipe.config.merge(settings))
    # All arms score the same EvalLite100 prompts at fixed optimizer steps.
    cadence = updates // 2
    return replace(
        base,
        name=name,
        revision=REVISION,
        recipe=recipe,
        seed=seed,
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
    presets=tuple(
        sampler_preset(f"{stage}-{arm}{suffix}", updates, kind, epsilon, seed)
        for stage, updates, seed, suffix, arms in (
            ("smoke", 4, SEED, "", (*MATCHED_ARMS, SOURCE_ADAPTIVE)),
            ("pilot", 20, SEED, "", (*MATCHED_ARMS, SOURCE_ADAPTIVE, SOURCE_THOMPSON)),
            ("pilot", 20, SECOND_SEED, f"-seed{SECOND_SEED}", MATCHED_ARMS),
        )
        for arm, kind, epsilon in arms
    ),
    control_pins={},
)

if __name__ == "__main__":
    raise SystemExit(rl.main(SAMPLER))
