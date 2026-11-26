# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace

import pytest

from experiments.evaluation import launch as evaluation_launch
from experiments.evaluation.policy import launch as policy_launch
from experiments.evaluation.policy.launch import PolicyModel, policy_model
from experiments.evaluation.policy.versions import POLICIES

POLICY = POLICIES["marin-2026-11-20-trial"]
QWEN_NAME = "qwen3.6-35b-a3b"


@pytest.fixture
def served(monkeypatch):
    """Resolve models without the Hugging Face Hub: catalog revisions are already pinned, and the serving config is
    the catalog's, clamped to a window a test can set to stand for the checkpoint's own."""
    monkeypatch.setattr(policy_launch, "pinned_model", lambda model: model)
    windows = {"own": None}

    def resolved(model):
        own = windows["own"]
        window = model.serve.max_model_len if own is None else min(own, model.serve.max_model_len or own)
        return replace(model.serve, max_model_len=window)

    monkeypatch.setattr(policy_launch, "resolved_serve_config", resolved)

    def set_own_window(window: int) -> None:
        windows["own"] = window

    return set_own_window


@pytest.fixture
def qwen(served) -> PolicyModel:
    return policy_model(POLICY, POLICY.model(QWEN_NAME), declared=True)


@pytest.fixture
def records(monkeypatch, tmp_path) -> str:
    """Write and find records under a temporary root instead of the fleet's."""
    root = str(tmp_path / "records")
    monkeypatch.setattr(evaluation_launch, "records_prefix_for", lambda _accelerator, _spec: root)
    return root
