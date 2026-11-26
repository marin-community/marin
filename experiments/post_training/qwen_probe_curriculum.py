# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compare a single development-driven quota update with a fixed string mixture."""

from dataclasses import replace

from skyrlrecipe import RecipePatch, sections
from skyrlrecipe.declared import Callback, DevelopmentProbeMixCallback, EvaluationProfile

from experiments.post_training import rl
from experiments.post_training.qwen_sampler import EVAL, LAYOUT, WITHIN_SOURCE_PASS, budget_preset
from experiments.post_training.rl_presets import Evaluation, ModelPresets, Preset, Training

TRAIN = "qwen-string-mixture@2026.11.26.1"
DEVELOPMENT = "qwen-string-curriculum-dev-smoke@2026.11.26.1"
DEVELOPMENT_SHA256 = "11c7882a4a188528fa3e3d5100a2ff4086de3e3451631eb55f6bdc07f86cf308"
DEVELOPMENT_TASKS = 48
TRAIN_TOKENS = 786_432
REVISION = "2026.11.26.1"


class ProbeTraining(Training):
    def callbacks(
        self,
        preset: Preset,
        evaluation: Evaluation | None,
        profiles: dict[str, EvaluationProfile],
        expected_tasks: int,
        initial_model_uri: str | None,
    ) -> tuple[Callback, ...]:
        if expected_tasks != DEVELOPMENT_TASKS:
            raise ValueError("The probe curriculum requires the complete development selection")
        callback = DevelopmentProbeMixCallback(
            type="development_probe_mix", panel_sha256=DEVELOPMENT_SHA256, expected_tasks=expected_tasks
        )
        return (*super().callbacks(preset, evaluation, profiles, expected_tasks, initial_model_uri), callback)


def probe_preset(seed: int, *, adaptive: bool) -> Preset:
    name = f"east02-probe-{'adaptive' if adaptive else 'balanced'}-seed{seed}"
    assert isinstance(LAYOUT.workload, Training)
    layout = replace(LAYOUT, workload=ProbeTraining(LAYOUT.workload.plan)) if adaptive else LAYOUT
    preset = budget_preset(
        name,
        WITHIN_SOURCE_PASS,
        1.0,
        seed,
        revision=REVISION,
        layout=layout,
        max_updates=32,
        tokens=TRAIN_TOKENS,
        evaluation=replace(EVAL, name="development"),
        held_out_tasks=DEVELOPMENT,
    )
    config = preset.recipe.config.merge(
        RecipePatch(
            data=sections.Data(sampling=sections.Sampling(domain_weights={"easy": 0.5, "intermediate": 0.5})),
            trainer=sections.Trainer(rollout_token_eval_milestones=(TRAIN_TOKENS // 2,)),
        )
    )
    return replace(
        preset,
        tasks=TRAIN,
        checkpoints="latest",
        checkpoint_every=4,
        accepts_settings=False,
        recipe=replace(preset.recipe, config=config),
    )


QWEN_PROBE = ModelPresets(
    model="qwen3-0.6b",
    namespace="qwen-probe-curriculum",
    presets=tuple(probe_preset(127, adaptive=adaptive) for adaptive in (False, True)),
    control_pins={},
)


if __name__ == "__main__":
    raise SystemExit(rl.main(QWEN_PROBE))
