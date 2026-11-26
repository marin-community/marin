# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compare fixed and learned prompt orders on the four-source Snowball selection."""

from dataclasses import replace

from skyrlrecipe import RecipePatch, sections
from skyrlrecipe.declared import PromptAcquisition

from experiments.post_training import rl
from experiments.post_training.rl_presets import ModelPresets, Preset
from experiments.post_training.snowball_rl import DATAKIT_SFT_2026_09_21, SEED, SNOWBALL
from experiments.post_training.task_curation.snowball_sampler_selection import EVAL_LITE, PILOT

REVISION = "2026.11.25.2"
SECOND_SEED = 20261021
SMOKE_UPDATES = 4
PROMPT_EXPLORATION = 0.05
WITHIN_SOURCE = PromptAcquisition.WITHIN_SOURCE_COST_AWARE.value
MATCHED_ARMS = (
    ("fixed", "adaptive-source", 1.0),
    ("source-only", PromptAcquisition.SOURCE.value, PROMPT_EXPLORATION),
    ("contextual", PromptAcquisition.CONTEXTUAL.value, PROMPT_EXPLORATION),
    ("cost-aware", PromptAcquisition.COST_AWARE.value, PROMPT_EXPLORATION),
    ("within-source", WITHIN_SOURCE, PROMPT_EXPLORATION),
    ("within-source-random", WITHIN_SOURCE, 1.0),
)
SOURCE_ADAPTIVE = ("adaptive", "adaptive-source", 0.20)
SOURCE_THOMPSON = ("thompson", "adaptive-source-thompson", 0.05)
CHUNKED_PROJECTION = RecipePatch(
    trainer=sections.Trainer(
        policy=sections.Policy(megatron_config=sections.PolicyMegatronConfig(output_projection_chunk_size=1024))
    )
)


def sampler_preset(name: str, updates: int, kind: str, epsilon: float, seed: int) -> Preset:
    base = SNOWBALL.by_name["train-long"].with_updates(updates)
    sampling = sections.Sampling(
        kind=kind,
        source_column="family",
        domain_weights={} if kind == WITHIN_SOURCE else dict.fromkeys(PILOT.family_limits, 1.0),
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
    if kind == WITHIN_SOURCE:
        settings = settings.merge(CHUNKED_PROJECTION)
    layout = base.recipe.layout
    if updates == SMOKE_UPDATES:
        # Completed Datakit P40 runs reached 957 GiB of sampled learner memory.
        layout = replace(layout, host_memory="1280GB")
    recipe = replace(base.recipe, layout=layout, config=base.recipe.config.merge(settings))
    # All arms score the same EvalLite100 prompts at fixed optimizer steps.
    cadence = updates // 2
    return replace(
        base,
        name=name,
        revision="2026.11.25.3" if kind == WITHIN_SOURCE else REVISION,
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


def fixed_calibration_smoke() -> Preset:
    base = sampler_preset("smoke-fixed-sync-chunked", SMOKE_UPDATES, "adaptive-source", 1.0, SEED)
    # Close all training leases at the optimizer barrier to measure the gross four-update budget.
    sync = RecipePatch(trainer=sections.Trainer(rollout_buffer=sections.RolloutBuffer(max_staleness_steps=0)))
    return replace(
        base,
        revision="2026.11.25.4",
        recipe=replace(base.recipe, config=base.recipe.config.merge(CHUNKED_PROJECTION).merge(sync)),
        evaluations=(replace(base.evaluations[0], every=None),),
        checkpoint_every=None,
    )


SAMPLER = ModelPresets(
    model=SNOWBALL.model,
    namespace="snowball-sampler",
    presets=(
        *(
            sampler_preset(f"{stage}-{arm}{suffix}", updates, kind, epsilon, seed)
            for stage, updates, seed, suffix, arms in (
                ("smoke", SMOKE_UPDATES, SEED, "", (*MATCHED_ARMS, SOURCE_ADAPTIVE)),
                ("pilot", 20, SEED, "", (*MATCHED_ARMS, SOURCE_ADAPTIVE, SOURCE_THOMPSON)),
                ("pilot", 20, SECOND_SEED, f"-seed{SECOND_SEED}", MATCHED_ARMS),
            )
            for arm, kind, epsilon in arms
        ),
        fixed_calibration_smoke(),
    ),
    control_pins={},
)

if __name__ == "__main__":
    raise SystemExit(rl.main(SAMPLER))
