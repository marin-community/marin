# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Qwen3 learning-rate and feedback-gain sweeps for error-aware Muon."""

from __future__ import annotations

import argparse
import dataclasses
import logging
import os
from dataclasses import dataclass

from marin.execution.step_runner import StepRunner

from experiments.speedrun.prism_berkeley_qwen3_scaling.muon_error_feedback_optimizer import (
    DEFAULT_INVERSE_NEWTON_STEPS,
    DEFAULT_SYLVESTER_STEPS,
    ErrorAwareMuonConfig,
    ErrorAwareMuonPolicy,
)
from experiments.speedrun.prism_berkeley_qwen3_scaling.prism_berkeley_sweep import build_config
from experiments.speedrun.prism_berkeley_qwen3_scaling.submission_support import SpeedrunConfig, default_speedrun

LEARNING_RATES = (0.008, 0.012, 0.016, 0.020, 0.024)
ADAM_LR_RATIO = 0.2
ARCHIVED_HOST_CPU = 32
ARCHIVED_HOST_RAM = "128g"
ARCHIVED_HOST_DISK = "50g"
SPECTRAL_CUBIC_STEPS = 15

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SweepSettings:
    display_name: str
    learning_rates: tuple[float, ...]
    adam_lr_ratio: float
    momentum: float


SWEEP_SETTINGS = {
    "130m": SweepSettings("130M", LEARNING_RATES, ADAM_LR_RATIO, 0.95),
    "300m": SweepSettings("300M", (0.004, 0.006, 0.008, 0.010, 0.012), 0.3, 0.98),
}


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


def build_optimizer(
    learning_rate: float,
    variant: FeedbackVariant,
    *,
    adam_lr_ratio: float = ADAM_LR_RATIO,
    momentum: float = 0.95,
) -> ErrorAwareMuonConfig:
    """Construct the handoff optimizer at one archived learning-rate point."""
    return ErrorAwareMuonConfig(
        learning_rate=learning_rate,
        adam_lr=adam_lr_ratio * learning_rate,
        momentum=momentum,
        nesterov=False,
        policy=variant.policy,
        blend_gain=variant.gain if variant.policy == "blend" else 0.0,
        correction_gain=variant.gain if variant.policy == "hesscorr" else 0.0,
        quintic_steps=5,
        cubic_steps=SPECTRAL_CUBIC_STEPS,
        sylvester_steps=DEFAULT_SYLVESTER_STEPS,
        inverse_steps=DEFAULT_INVERSE_NEWTON_STEPS,
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
    size: str = "130m",
    learning_rates: tuple[float, ...] | None = None,
    variants: tuple[FeedbackVariant, ...] = FEEDBACK_VARIANTS,
) -> list[tuple[str, SpeedrunConfig]]:
    """Build the deduplicated handoff gain grid crossed with the archived LR grid."""
    try:
        settings = SWEEP_SETTINGS[size]
    except KeyError:
        raise ValueError(f"Unsupported sweep size {size!r}; expected one of {sorted(SWEEP_SETTINGS)}") from None

    learning_rates = learning_rates if learning_rates is not None else settings.learning_rates
    _, base_config = build_config(size)
    resources = dataclasses.replace(
        base_config.train_config.resources,
        cpu=ARCHIVED_HOST_CPU,
        ram=ARCHIVED_HOST_RAM,
        disk=ARCHIVED_HOST_DISK,
        preemptible=True,
    )
    sweep_configs = []
    for variant in variants:
        for learning_rate in learning_rates:
            optimizer = build_optimizer(
                learning_rate,
                variant,
                adam_lr_ratio=settings.adam_lr_ratio,
                momentum=settings.momentum,
            )
            train_config = dataclasses.replace(
                base_config.train_config,
                learning_rate=learning_rate,
                optimizer_config=optimizer,
                resources=resources,
            )
            config = dataclasses.replace(
                base_config,
                description=(
                    f"Qwen3 ~{settings.display_name} error-aware Muon sweep with "
                    f"policy={variant.policy}, gain={variant.gain:g}, Muon lr={learning_rate:g}, "
                    f"and Adam lr={optimizer.adam_lr:g}."
                ),
                train_config=train_config,
            )
            name = f"qwen3_{size}_error_aware_muon_{variant.slug}_lr{_float_slug(learning_rate)}"
            sweep_configs.append((name, config))
    return sweep_configs


def main(*, size: str = "130m", version: str = "dev", max_concurrent: int = 8) -> None:
    if os.getenv("CI") is not None:
        logger.info("Skipping experiment execution on CI environment, needs HF access.")
        return

    result_steps = []
    for name, config in build_sweep_configs(size=size):
        config.print_run_info()
        _, result_step = default_speedrun(
            name,
            config,
            tags=["error-aware-muon", f"{size}-sweep"],
            version=version,
        )
        result_steps.append(result_step.lower())

    StepRunner().run(result_steps, max_concurrent=max_concurrent)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", choices=tuple(SWEEP_SETTINGS), default="130m")
    parser.add_argument("--version", default="dev")
    parser.add_argument("--max-concurrent", type=int, default=8)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    main(size=args.size, version=args.version, max_concurrent=args.max_concurrent)
