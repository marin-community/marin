# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Reusable data mixtures for midtraining experiments."""

import dataclasses

from levanter.data.text import LmDataConfig

from experiments.midtraining_data_buckets import BUCKET_2
from experiments.pretraining_datasets.nemotron import nemotron_mix
from marin.processing.tokenize import lm_mixture_data_config
from marin.processing.tokenize.data_configs import step_to_lm_mixture_component

FULL_HIGHQUALITY_NEMO_MATH_NAME = "full_highquality_nemo_math"
PRETRAIN_70P_MATH_30P_HIGHQUALITY_NEMO_MATH_NAME = "70p_30m_highquality_nemo_math"
PRETRAIN_67P_MATH_33P_HIGHQUALITY_NEMO_MATH_NAME = "67p_33m_highquality_nemo_math"
PRETRAIN_33P_MATH_67P_HIGHQUALITY_NEMO_MATH_NAME = "33p_67m_highquality_nemo_math"

HIGHQUALITY_NEMO_MATH_KEY = "nemotron_cc_math_v1/4plus"
PRETRAIN_REPLAY_FRACTION = 0.7
HIGHQUALITY_MATH_FRACTION = 0.3
BALANCED_REPLAY_FRACTION = 0.67
BALANCED_MATH_FRACTION = 0.33
LOW_REPLAY_FRACTION = 0.33
HIGHQUALITY_MATH_HIGH_FRACTION = 0.67

_highquality_nemo_math_step = BUCKET_2[HIGHQUALITY_NEMO_MATH_KEY]

full_highquality_nemo_math = lm_mixture_data_config(
    components={HIGHQUALITY_NEMO_MATH_KEY: _highquality_nemo_math_step},
    weights={HIGHQUALITY_NEMO_MATH_KEY: 1.0},
)


def _fixed_train_weights(config: LmDataConfig, *, name: str) -> dict[str, float]:
    weights = config.train_weights
    if not isinstance(weights, dict):
        raise ValueError(f"{name} must use fixed train_weights, got {type(weights)}")
    return weights


def _scale_weights(weights: dict[str, float], target_sum: float) -> dict[str, float]:
    total = sum(weights.values())
    if total <= 0:
        raise ValueError("Cannot scale mixture weights with non-positive total")
    return {name: weight * target_sum / total for name, weight in weights.items()}


_nemotron_pretraining_weights = _fixed_train_weights(nemotron_mix, name="nemotron_mix")


def _nemotron_replay_math_mix(*, pretrain_fraction: float, math_fraction: float) -> LmDataConfig:
    weights = {
        **_scale_weights(_nemotron_pretraining_weights, pretrain_fraction),
        HIGHQUALITY_NEMO_MATH_KEY: math_fraction,
    }

    return dataclasses.replace(
        nemotron_mix,
        components={
            **nemotron_mix.components,
            HIGHQUALITY_NEMO_MATH_KEY: step_to_lm_mixture_component(
                _highquality_nemo_math_step,
                include_raw_paths=True,
            ),
        },
        train_weights=weights,
    )


pretrain_70p_math_30p_highquality_nemo_math = _nemotron_replay_math_mix(
    pretrain_fraction=PRETRAIN_REPLAY_FRACTION,
    math_fraction=HIGHQUALITY_MATH_FRACTION,
)

pretrain_67p_math_33p_highquality_nemo_math = _nemotron_replay_math_mix(
    pretrain_fraction=BALANCED_REPLAY_FRACTION,
    math_fraction=BALANCED_MATH_FRACTION,
)

pretrain_33p_math_67p_highquality_nemo_math = _nemotron_replay_math_mix(
    pretrain_fraction=LOW_REPLAY_FRACTION,
    math_fraction=HIGHQUALITY_MATH_HIGH_FRACTION,
)

MIDTRAINING_MIXES = {
    FULL_HIGHQUALITY_NEMO_MATH_NAME: full_highquality_nemo_math,
    PRETRAIN_70P_MATH_30P_HIGHQUALITY_NEMO_MATH_NAME: pretrain_70p_math_30p_highquality_nemo_math,
    PRETRAIN_67P_MATH_33P_HIGHQUALITY_NEMO_MATH_NAME: pretrain_67p_math_33p_highquality_nemo_math,
    PRETRAIN_33P_MATH_67P_HIGHQUALITY_NEMO_MATH_NAME: pretrain_33p_math_67p_highquality_nemo_math,
}


def midtraining_mix_by_name(name: str) -> LmDataConfig:
    """Return a registered midtraining mixture by stable string name."""
    if name not in MIDTRAINING_MIXES:
        valid_names = ", ".join(sorted(MIDTRAINING_MIXES))
        raise ValueError(f"Unknown midtraining mix {name!r}; valid names: {valid_names}")
    return MIDTRAINING_MIXES[name]
