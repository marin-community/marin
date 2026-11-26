# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compare realistic sampling interventions at matched gross response spend."""

from dataclasses import replace
from fractions import Fraction

from skyrlrecipe import RecipePatch, sections

from experiments.post_training import rl
from experiments.post_training.qwen_realistic_sampler import readiness
from experiments.post_training.qwen_realistic_sampler_training import MAX_RESPONSE_TOKENS, realistic_smoke
from experiments.post_training.qwen_sampler import ROLE_PLAN, with_rollout_token_budget
from experiments.post_training.rl_presets import BatchMix, ModelPresets, Preset
from experiments.post_training.task_curation.qwen_realistic_sampler import TRAIN_FAMILIES

REVISION = "2026.11.26.9"
SEEDS = (26633, 26639)
GROSS_TOKENS = 2_097_152
MAX_UPDATES = 120
KV_CACHE_BYTES_PER_REPLICA = 11 * 1024**3
CHECKPOINT_EVERY = 16
# Frozen from the independent development probe's mixed rewards (10, 2, 1, 7 out of 24).
REFERENCE_COUNTS = (4, 4, 4, 4)
VARIANCE_COUNTS = (7, 2, 2, 5)
ARMS = (
    ("fixed", 1.0, REFERENCE_COUNTS),
    ("ranked", 0.15, REFERENCE_COUNTS),
    ("reweighted", 1.0, VARIANCE_COUNTS),
)


def realistic_study(arm: str, epsilon: float, counts: tuple[int, ...], seed: int) -> Preset:
    """Bind one arm to the same policy, verifier, independent eval and token stop."""
    smoke = realistic_smoke(arm, epsilon)
    panel = readiness("confirmation", 1, "native")
    (evaluation,) = panel.evaluations
    params = evaluation.profile.sampling_params
    assert params is not None
    evaluation = replace(
        evaluation,
        profile=evaluation.profile.replace(sampling_params=params.replace(max_generate_length=MAX_RESPONSE_TOKENS)),
    )
    mix = BatchMix(
        {
            family: Fraction(count, ROLE_PLAN.train_batch_size)
            for family, count in zip(TRAIN_FAMILIES, counts, strict=True)
        }
    )
    config = smoke.recipe.config.merge(
        RecipePatch(
            generator=sections.Generator(
                gpu_memory_utilization=0.35,
                engine_init_kwargs={"kv_cache_memory_bytes": KV_CACHE_BYTES_PER_REPLICA},
            )
        )
    )
    recipe = replace(
        smoke.recipe, config=config, layout=replace(smoke.recipe.layout, host_memory="128GB", job_timeout_seconds=7200)
    )
    preset = replace(
        smoke,
        name=f"realistic-study-{arm}-seed{seed}",
        revision=REVISION,
        seed=seed,
        recipe=recipe,
        updates=MAX_UPDATES,
        measured=range(1, MAX_UPDATES + 1),
        evaluations=(evaluation,),
        held_out_tasks=panel.held_out_tasks,
        batch_mix=mix,
        checkpoint_every=CHECKPOINT_EVERY,
        retention=replace(smoke.retention, resume_checkpoint_count=2),
        accepts_settings=False,
    )
    return with_rollout_token_budget(preset, GROSS_TOKENS)


QWEN_REALISTIC_STUDY = ModelPresets(
    model="qwen-realistic-sampler",
    namespace="qwen-realistic-sampler",
    control_pins={},
    presets=tuple(realistic_study(arm, epsilon, counts, seed) for seed in SEEDS for arm, epsilon, counts in ARMS),
)


if __name__ == "__main__":
    raise SystemExit(rl.main(QWEN_REALISTIC_STUDY))
