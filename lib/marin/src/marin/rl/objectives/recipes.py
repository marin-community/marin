# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Named objective recipes built on the compositional objective surface."""

from .spec import (
    BatchView,
    ObjectiveSpec,
    PolicyGradientTermConfig,
    RLOOSignalConfig,
    ReductionConfig,
    ReductionKind,
    ReferenceKLTermConfig,
    TruncationPolicy,
)


def make_rloo_objective(
    *,
    kl_coef: float = 0.1,
    clip_epsilon_low: float = 0.2,
    clip_epsilon_high: float = 0.2,
    tis_importance_sampling_ratio_max: float = 2.0,
    synchronous: bool = False,
    do_trainer_inference_mismatch_importance_sampling: bool = False,
    do_overlong_filtering: bool = False,
    reduction_kind: ReductionKind = ReductionKind.DAPO,
) -> ObjectiveSpec:
    """Return the current RLOO recipe on the new objective surface."""
    terms = [
        PolicyGradientTermConfig(
            clip_epsilon_low=clip_epsilon_low,
            clip_epsilon_high=clip_epsilon_high,
            tis_importance_sampling_ratio_max=tis_importance_sampling_ratio_max,
            do_trainer_inference_mismatch_importance_sampling=do_trainer_inference_mismatch_importance_sampling,
            synchronous=synchronous,
        )
    ]
    if kl_coef > 0:
        terms.append(ReferenceKLTermConfig(kl_coef=kl_coef))

    truncation_policy = TruncationPolicy.KEEP
    if do_overlong_filtering:
        truncation_policy = TruncationPolicy.FILTER_ENTIRE_RESPONSE

    return ObjectiveSpec(
        batch_view=BatchView.SEQUENCE,
        signal_builder=RLOOSignalConfig(),
        terms=tuple(terms),
        reduction=ReductionConfig(kind=reduction_kind),
        truncation_policy=truncation_policy,
    )
