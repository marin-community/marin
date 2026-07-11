# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Qwen3 130M learning-rate and feedback-gain sweep for error-aware Muon."""

from __future__ import annotations

import dataclasses
import logging
import os
from dataclasses import dataclass

from marin.execution.step_runner import StepRunner

from experiments.speedrun.prism_berkeley_qwen3_scaling.muon_error_feedback_optimizer import (
    ErrorAwareMuonConfig,
    ErrorAwareMuonPolicy,
)
from experiments.speedrun.prism_berkeley_qwen3_scaling.prism_berkeley_sweep import build_config
from experiments.speedrun.prism_berkeley_qwen3_scaling.submission_support import SpeedrunConfig, default_speedrun

LEARNING_RATES = (0.008, 0.012, 0.016, 0.020, 0.024)
ADAM_LR_RATIO = 0.2

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FeedbackVariant:
    policy: ErrorAwareMuonPolicy
    gain: float

    @property
    def slug(self) -> str:
        if self.policy == "muon":
            return "muon"
        return f"{self.policy}-g{_float_slug(self.gain)}"


# Zero-gain blend and hesscorr are exactly Muon, so keep only one baseline.
FEEDBACK_VARIANTS = (
    FeedbackVariant("muon", 0.0),
    FeedbackVariant("blend", 0.05),
    FeedbackVariant("blend", 0.15),
    FeedbackVariant("blend", 0.3),
    FeedbackVariant("blend", 0.5),
    FeedbackVariant("hesscorr", 0.1),
    FeedbackVariant("hesscorr", 0.3),
    FeedbackVariant("hesscorr", 1.0),
)


def _float_slug(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def build_optimizer(learning_rate: float, variant: FeedbackVariant) -> ErrorAwareMuonConfig:
    """Construct the handoff optimizer at one archived 130M learning-rate point."""
    return ErrorAwareMuonConfig(
        learning_rate=learning_rate,
        adam_lr=ADAM_LR_RATIO * learning_rate,
        momentum=0.95,
        nesterov=False,
        policy=variant.policy,
        blend_gain=variant.gain if variant.policy == "blend" else 0.0,
        correction_gain=variant.gain if variant.policy == "hesscorr" else 0.0,
        quintic_steps=5,
        cubic_steps=30,
        weight_decay=0.1,
        adam_weight_decay=None,
        beta1=0.8,
        beta2=0.98,
        epsilon=1e-15,
        muon_epsilon=1e-12,
        adamc_weight_decay=True,
        max_grad_norm=1.0,
        use_kimi_scaling=False,
        min_matrix_dim=8,
        lr_schedule="linear",
        warmup=0,
        decay=0.8,
        rewarmup=0,
        min_lr_ratio=0,
    )


def build_sweep_configs(
    *,
    learning_rates: tuple[float, ...] = LEARNING_RATES,
    variants: tuple[FeedbackVariant, ...] = FEEDBACK_VARIANTS,
) -> list[tuple[str, SpeedrunConfig]]:
    """Build the deduplicated handoff gain grid crossed with the archived LR grid."""
    _, base_config = build_config("130m")
    sweep_configs = []
    for variant in variants:
        for learning_rate in learning_rates:
            optimizer = build_optimizer(learning_rate, variant)
            train_config = dataclasses.replace(
                base_config.train_config,
                learning_rate=learning_rate,
                optimizer_config=optimizer,
            )
            config = dataclasses.replace(
                base_config,
                description=(
                    "Qwen3 ~130M error-aware Muon sweep with "
                    f"policy={variant.policy}, gain={variant.gain:g}, Muon lr={learning_rate:g}, "
                    f"and Adam lr={optimizer.adam_lr:g}."
                ),
                train_config=train_config,
            )
            name = f"qwen3_130m_error_aware_muon_{variant.slug}_lr{_float_slug(learning_rate)}"
            sweep_configs.append((name, config))
    return sweep_configs


def main() -> None:
    if os.getenv("CI") is not None:
        logger.info("Skipping experiment execution on CI environment, needs HF access.")
        return

    result_steps = []
    for name, config in build_sweep_configs():
        config.print_run_info()
        _, result_step = default_speedrun(name, config, tags=["error-aware-muon", "130m-sweep"])
        result_steps.append(result_step.lower())

    StepRunner().run(result_steps)


if __name__ == "__main__":
    main()
