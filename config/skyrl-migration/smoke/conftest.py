# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Pin the existing Megatron gate's model input without changing its assertions."""

import json
from pathlib import Path

import pytest
from huggingface_hub import snapshot_download
from tests.gpu import test_megatron_worker


@pytest.fixture(autouse=True)
def pinned_model(monkeypatch):
    receipt = json.loads((Path(__file__).resolve().parent.parent / "receipt.json").read_text())
    model = snapshot_download(repo_id=receipt["model"], revision=receipt["model_revision"])
    original = test_megatron_worker.get_test_actor_config

    def config(model_name=model, logger=None):
        return original(model_name=model_name, logger=logger)

    monkeypatch.setattr(test_megatron_worker, "MODEL_NAME", model)
    monkeypatch.setattr(test_megatron_worker, "get_test_actor_config", config)
