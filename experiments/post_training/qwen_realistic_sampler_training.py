# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Short paired wiring check on the admitted realistic Qwen mixture."""

from dataclasses import replace
from fractions import Fraction

from skyrlrecipe import RecipePatch, sections

from experiments.post_training import rl
from experiments.post_training.qwen_realistic_sampler import readiness
from experiments.post_training.qwen_sampler import LAYOUT, ROLE_PLAN, WITHIN_SOURCE, sampler_preset
from experiments.post_training.rl_presets import BatchMix, ModelPresets, Preset, Training
from experiments.post_training.task_curation.qwen_realistic_sampler import NAME, TRAIN_FAMILIES, VERSION

REVISION = "2026.11.26.3"
UPDATES = 4
SEED = 26631


def realistic_smoke(arm: str, epsilon: float) -> Preset:
    probe = readiness("dev", 1, "direct")
    plan = replace(ROLE_PLAN, micro_train_batch_size_per_gpu=1)
    layout = replace(
        LAYOUT,
        workload=Training(plan),
        container_profile=probe.recipe.layout.container_profile,
        job_timeout_seconds=3600,
    )
    preset = sampler_preset(
        f"realistic-direct-smoke-{arm}",
        UPDATES,
        WITHIN_SOURCE,
        epsilon,
        SEED,
        layout=layout,
        revision=REVISION,
        evaluation_interval=None,
    )
    config = preset.recipe.config.replace(
        environment=probe.recipe.config.environment,
        context_budget=probe.recipe.config.context_budget,
    ).merge(
        RecipePatch(
            trainer=sections.Trainer(
                rollout_buffer=sections.RolloutBuffer(batch_policy="full_batch"),
                policy=sections.Policy(megatron_config=sections.PolicyMegatronConfig(output_projection_chunk_size=1024)),
            ),
            generator=sections.Generator(
                chat_template_kwargs={"enable_thinking": False},
                sampling_params=sections.SamplingParams(temperature=0.7, top_p=0.8, top_k=20, logprobs=None),
            ),
        )
    )
    mix = BatchMix(dict.fromkeys(TRAIN_FAMILIES, Fraction(1, len(TRAIN_FAMILIES))))
    return replace(
        preset,
        tasks=f"{NAME}@{VERSION}",
        held_out_tasks=probe.held_out_tasks,
        evaluations=probe.evaluations,
        recipe=replace(preset.recipe, config=config),
        checkpoints="latest",
        export="latest",
        batch_mix=mix,
    )


QWEN_REALISTIC_TRAINING = ModelPresets(
    model="qwen-realistic-sampler",
    namespace="qwen-realistic-sampler",
    control_pins={},
    presets=(realistic_smoke("fixed", 1.0), realistic_smoke("ranked", 0.15)),
)


if __name__ == "__main__":
    raise SystemExit(rl.main(QWEN_REALISTIC_TRAINING))
