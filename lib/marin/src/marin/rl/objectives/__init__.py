# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Composable objective runtime for RL/post-training."""

from .recipes import make_rloo_objective
from .runtime import ObjectiveBatch, ObjectiveRuntime, ObjectiveRuntimeConfig, build_objective_runtime
from .signals import PreparedSignals
from .spec import (
    BatchView,
    NoRewardSignalConfig,
    ObjectiveSpec,
    PolicyGradientTermConfig,
    RLOOSignalConfig,
    ReductionConfig,
    ReductionKind,
    ReferenceKLTermConfig,
    TruncationPolicy,
)
