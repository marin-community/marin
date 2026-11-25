# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Eval policies by version, one ``<version>/policy.yaml`` per directory here."""

from collections.abc import Mapping
from pathlib import Path

from experiments.evaluation.policy.load import load_policy
from experiments.evaluation.policy.schema import EvalPolicy


def _load_versions() -> dict[str, EvalPolicy]:
    policies: dict[str, EvalPolicy] = {}
    for path in sorted(Path(__file__).parent.glob("*/policy.yaml")):
        policy = load_policy(path)
        if policy.version in policies:
            raise ValueError(f"{path}: version {policy.version} is declared twice")
        policies[policy.version] = policy
    return policies


POLICIES: Mapping[str, EvalPolicy] = _load_versions()
