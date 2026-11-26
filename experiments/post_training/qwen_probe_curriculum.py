# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compare mild transfer-driven Qwen string mixtures at a fixed rollout budget."""

from dataclasses import dataclass, replace
from typing import Literal

from skyrlrecipe import RecipePatch, sections
from skyrlrecipe.declared import Callback, DevelopmentProbeMixCallback, EvaluationProfile

from experiments.post_training import rl
from experiments.post_training.qwen_sampler import EVAL, LAYOUT, WITHIN_SOURCE_PASS, budget_preset
from experiments.post_training.rl_presets import Evaluation, ModelPresets, Preset, Training

TRAIN = "qwen-string-mixture2048@2026.11.26.1"
DEVELOPMENT = "qwen-string-transfer-dev96@2026.11.26.1"
DEVELOPMENT_SHA256 = "f14f9c0dfec16cfac2205c2b20ec8ac2b6ca7fb7d542563e3f127a6839206203"
DEVELOPMENT_TASKS = 96
CONFIRMATION = "qwen-string-confirm480@2026.11.26.1"
CONFIRMATION_SHA256 = "022faa2f761cf3e7782e4814cde82a0a731165b5a3082d40cb519a3813927e8c"
CONFIRMATION_TASKS = 480
BLOCK_TOKENS = 1_048_576
TRAIN_TOKENS = 4 * BLOCK_TOKENS
MILESTONES = tuple(BLOCK_TOKENS * block for block in range(1, 4))
REVISION = "2026.11.26.4"
SEEDS = (137, 139, 149)
ARMS = ("balanced", "randomized", "adaptive")


@dataclass(frozen=True)
class ProbeTraining(Training):
    policy: Literal["adaptive", "randomized"]

    def callbacks(
        self,
        preset: Preset,
        evaluation: Evaluation | None,
        profiles: dict[str, EvaluationProfile],
        expected_tasks: int,
        initial_model_uri: str | None,
    ) -> tuple[Callback, ...]:
        if expected_tasks != DEVELOPMENT_TASKS or set(profiles) != {"confirmation"}:
            raise ValueError("The probe curriculum requires its complete development and confirmation selections")
        callback = DevelopmentProbeMixCallback(
            type="development_probe_mix",
            panel_sha256=DEVELOPMENT_SHA256,
            expected_tasks=expected_tasks,
            policy=self.policy,
            confirmation_sha256=CONFIRMATION_SHA256,
            confirmation_tasks=CONFIRMATION_TASKS,
        )
        return (*super().callbacks(preset, evaluation, profiles, expected_tasks, initial_model_uri), callback)


def probe_preset(seed: int, arm: Literal["balanced", "randomized", "adaptive"]) -> Preset:
    assert isinstance(LAYOUT.workload, Training)
    layout = LAYOUT if arm == "balanced" else replace(LAYOUT, workload=ProbeTraining(LAYOUT.workload.plan, arm))
    preset = budget_preset(
        f"mild-{arm}-s{seed}",
        WITHIN_SOURCE_PASS,
        1.0,
        seed,
        revision=REVISION,
        layout=layout,
        max_updates=128,
        tokens=TRAIN_TOKENS,
        evaluation=replace(EVAL, name="development"),
        held_out_tasks=DEVELOPMENT,
    )
    config = preset.recipe.config.merge(
        RecipePatch(
            data=sections.Data(sampling=sections.Sampling(domain_weights={"easy": 0.5, "intermediate": 0.5})),
            trainer=sections.Trainer(rollout_token_eval_milestones=MILESTONES),
        )
    )
    return replace(
        preset,
        tasks=TRAIN,
        checkpoints="latest",
        checkpoint_every=8,
        accepts_settings=False,
        evaluations=(*preset.evaluations, replace(EVAL, name="confirmation", at_start=False, selection=CONFIRMATION)),
        recipe=replace(preset.recipe, config=config, layout=replace(preset.recipe.layout, job_timeout_seconds=4_200)),
    )


QWEN_PROBE = ModelPresets(
    model="qwen3-0.6b",
    namespace="qwen-probe-curriculum",
    presets=tuple(probe_preset(seed, arm) for seed in SEEDS for arm in ARMS),
    control_pins={},
)


if __name__ == "__main__":
    raise SystemExit(rl.main(QWEN_PROBE))
