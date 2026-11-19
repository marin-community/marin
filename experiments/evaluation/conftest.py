# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Offline Hub metadata and submitter credentials at the launcher boundary."""

import json
from types import SimpleNamespace

import pytest
from huggingface_hub import HfApi


@pytest.fixture(autouse=True)
def hub_metadata(tmp_path_factory, monkeypatch):
    config = tmp_path_factory.mktemp("hub-metadata") / "config.json"
    config.write_text(
        json.dumps(
            {
                "architectures": ["LlamaForCausalLM"],
                "model_type": "llama",
                "max_position_embeddings": 1048576,
                "hidden_size": 1024,
                "num_attention_heads": 16,
                "num_key_value_heads": 8,
                "num_hidden_layers": 24,
            }
        )
    )
    monkeypatch.setattr(HfApi, "model_info", lambda *_args, **_kwargs: SimpleNamespace(sha="f" * 40, siblings=()))
    monkeypatch.setattr("marin.evaluation.serving_config.hf_hub_download", lambda *_args, **_kwargs: str(config))


@pytest.fixture(autouse=True)
def coreweave_credentials(monkeypatch):
    """Stand in for the submitter's storage keys that CoreWeave launches require."""
    monkeypatch.setenv("CW_KEY_ID", "local-test-id")
    monkeypatch.setenv("CW_KEY_SECRET", "local-test-secret")
