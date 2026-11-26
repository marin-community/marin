# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compare two Qwen task allocations from one native recovery checkpoint."""

from dataclasses import dataclass, replace
from typing import Literal

from skyrlrecipe import RecipePatch, sections
from skyrlrecipe.declared import Callback, EvaluationProfile, ForkSourceQuotaCallback

from experiments.post_training import rl
from experiments.post_training.qwen_probe_curriculum import BLOCK_TOKENS, DEVELOPMENT, TRAIN
from experiments.post_training.qwen_sampler import EVAL, LAYOUT, WITHIN_SOURCE_PASS, budget_preset
from experiments.post_training.rl_presets import Evaluation, ModelPresets, Preset, Training

SEED = 157
REVISION = "2026.11.26.5"
WARMUP_NAME = "warmup"
FORK_REVISION = "2026.11.26.6"
ENDPOINT = "qwen-string-fork384@2026.11.26.1"
FORKS: tuple[tuple[str, Literal[6, 10]], ...] = (("easy10", 10), ("easy06", 6))
FORK_SEEDS = (SEED, 163, 167)
EXTENSION_SEEDS = (173, 179)
EXTENSION_REVISION = "2026.11.26.7"


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


@dataclass(frozen=True)
class ForkTraining(Training):
    easy_quota: Literal[6, 10]

    def callbacks(
        self,
        preset: Preset,
        evaluation: Evaluation | None,
        profiles: dict[str, EvaluationProfile],
        expected_tasks: int,
        initial_model_uri: str | None,
    ) -> tuple[Callback, ...]:
        callbacks = super().callbacks(preset, evaluation, profiles, expected_tasks, initial_model_uri)
        if preset.revision == EXTENSION_REVISION:
            # The fork restores the shared donor and publishes terminal evaluations only.
            callbacks = tuple(callback for callback in callbacks if callback.type != "checkpoint")
        return (*callbacks, ForkSourceQuotaCallback(type="fork_source_quota", easy_quota=self.easy_quota))


def fork_preset(name: str, easy_quota: Literal[6, 10], seed: int, *, revision: str = FORK_REVISION) -> Preset:
    """Continue the same source pass to a shared absolute two Mi-token cap."""
    source = warmup_preset()
    layout = source.recipe.layout
    assert isinstance(layout.workload, Training)
    if revision == EXTENSION_REVISION:
        layout = replace(layout, host_memory="128GiB")
    config = source.recipe.config.merge(
        RecipePatch(
            data=sections.Data(sampling=sections.Sampling(seed=SEED)),
            trainer=sections.Trainer(rollout_token_budget=2 * BLOCK_TOKENS),
        )
    )
    return replace(
        source,
        name=f"{name}-s{seed}",
        revision=revision,
        seed=seed,
        evaluations=(
            replace(EVAL, name="development", at_start=False),
            replace(EVAL, name="confirmation", at_start=False, selection=ENDPOINT),
        ),
        recipe=replace(
            source.recipe,
            config=config,
            layout=replace(layout, workload=ForkTraining(layout.workload.plan, easy_quota)),
        ),
    )


QWEN_FORK = ModelPresets(
    model="qwen3-0.6b",
    namespace="qwen-checkpoint-fork",
    presets=(
        warmup_preset(),
        *(fork_preset(name, quota, seed) for seed in FORK_SEEDS for name, quota in FORKS),
        *(
            fork_preset(name, quota, seed, revision=EXTENSION_REVISION)
            for seed in EXTENSION_SEEDS
            for name, quota in FORKS
        ),
    ),
    control_pins={},
)


if __name__ == "__main__":
    raise SystemExit(rl.main(QWEN_FORK))
