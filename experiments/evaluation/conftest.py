# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Offline Hub metadata and submitter credentials at the launcher boundary."""

import json
from types import SimpleNamespace

import pytest
from huggingface_hub import HfApi

from experiments.evaluation.catalog import evaluation_benchmarks
from experiments.post_training.task_curation import license_binding
from experiments.post_training.task_curation.tests.license_fixtures import fixture_license
from infra.marina.apps.rl_data_catalog.server import licensing


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


@pytest.fixture(autouse=True)
def reviewed_evaluation_licenses(tmp_path_factory, monkeypatch):
    """Give launcher fixtures explicit grants without resolving real upstream terms."""
    receipts = {}
    for benchmark in evaluation_benchmarks().values():
        for variant in benchmark.variants:
            for source in benchmark.build(variant):
                dataset = source.dataset
                assert dataset is not None and dataset.revision is not None
                receipts[source.info.id] = fixture_license(source.info.id, dataset.name, dataset.revision).model_dump(
                    mode="json"
                )
    directory = tmp_path_factory.mktemp("reviewed-license-registry")
    (directory / "part-000.json").write_text(json.dumps(receipts, sort_keys=True) + "\n")
    monkeypatch.setattr(license_binding, "LICENSE_RECEIPT_INDEX", directory)
    monkeypatch.setattr(licensing, "LICENSE_BANS", '{"version":"fixture","bans":[]}')
    licensing.license_bans.cache_clear()
    yield
    licensing.license_bans.cache_clear()
