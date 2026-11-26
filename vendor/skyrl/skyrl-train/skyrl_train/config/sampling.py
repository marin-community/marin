# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Sampler kinds and policy-specific configuration checks used by the launcher."""

from enum import StrEnum

from omegaconf import DictConfig


class PromptAcquisition(StrEnum):
    SOURCE = "prompt-thompson-source"
    CONTEXTUAL = "prompt-thompson-contextual"
    COST_AWARE = "prompt-thompson-cost-aware"
    WITHIN_SOURCE_COST_AWARE = "prompt-thompson-within-source-cost-aware"
    WITHIN_SOURCE_PASS_COST_AWARE = "prompt-thompson-within-source-pass-cost-aware"

    @property
    def within_source(self) -> bool:
        return self in (self.WITHIN_SOURCE_COST_AWARE, self.WITHIN_SOURCE_PASS_COST_AWARE)


class SamplingKind(StrEnum):
    DOMAIN_WEIGHTED = "domain-weighted"
    ADAPTIVE_SOURCE = "adaptive-source"
    ADAPTIVE_SOURCE_THOMPSON = "adaptive-source-thompson"
    PROMPT_THOMPSON_SOURCE = PromptAcquisition.SOURCE.value
    PROMPT_THOMPSON_CONTEXTUAL = PromptAcquisition.CONTEXTUAL.value
    PROMPT_THOMPSON_COST_AWARE = PromptAcquisition.COST_AWARE.value
    PROMPT_THOMPSON_WITHIN_SOURCE_COST_AWARE = PromptAcquisition.WITHIN_SOURCE_COST_AWARE.value
    PROMPT_THOMPSON_WITHIN_SOURCE_PASS_COST_AWARE = PromptAcquisition.WITHIN_SOURCE_PASS_COST_AWARE.value
    NAIVE = "naive"
    THOMPSON = "thompson"
    LEARNABILITY = "learnability"
    GRADE_UNIFORM = "grade-uniform"
    GRADE_ADAPTIVE = "grade-adaptive"
    GRADE_PRIOR = "grade-prior"


ADAPTIVE_SOURCE_KINDS = (SamplingKind.ADAPTIVE_SOURCE, SamplingKind.ADAPTIVE_SOURCE_THOMPSON)


def validate_sampling_reversion(config: DictConfig) -> None:
    """Reject curriculum pseudo-evidence on sampler policies that use their own priors."""
    if config.reversion_mass and (config.kind in ADAPTIVE_SOURCE_KINDS or config.kind in PromptAcquisition):
        raise ValueError(
            f"data.sampling.reversion_mass applies to bin curricula. Set it to 0 for {config.kind}. "
            "data.sampling.decay controls how quickly older observations lose weight."
        )
