# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Eval policies by version."""

from collections.abc import Mapping

from experiments.evaluation.policies.marin_2026_11_20_trial.policy import POLICY as MARIN_2026_11_20_TRIAL
from experiments.evaluation.policy.schema import EvalPolicy

POLICIES: Mapping[str, EvalPolicy] = {policy.version: policy for policy in (MARIN_2026_11_20_TRIAL,)}
