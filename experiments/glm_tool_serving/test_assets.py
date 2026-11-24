# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json

import pytest

from experiments.glm_tool_serving import assets
from experiments.post_training.task_curation import reference_data


@pytest.fixture
def pinned_assets(tmp_path, monkeypatch):
    data = tmp_path / "production_data"
    serving = data / "serving_tokenizer"
    serving.mkdir(parents=True)
    payloads = {
        "sft_tokenizer": b"SFT vocabulary",
        "tokenizer.json": b"serving vocabulary",
        "chat_template.jinja": b"template",
    }
    downloads = tmp_path / "downloads"
    downloads.mkdir()
    for name, raw in payloads.items():
        (downloads / name).write_bytes(raw)
    (data / "reference-index.json").write_text(
        json.dumps(
            {
                "repo": "test/assets",
                "revision": "a" * 40,
                "references": [
                    {
                        "repo": "test/assets",
                        "revision": "a" * 40,
                        "path": "sft_tokenizer",
                        "sha256": hashlib.sha256(payloads["sft_tokenizer"]).hexdigest(),
                    }
                ],
            }
        )
    )
    (serving / "pins.json").write_text(
        json.dumps(
            {
                "repository": "test/serving",
                "declared_revision": "b" * 40,
                "files": {
                    name: hashlib.sha256(raw).hexdigest() for name, raw in payloads.items() if name != "sft_tokenizer"
                },
            }
        )
    )

    def download(*, repo_id, revision, filename, repo_type="model"):
        return str(downloads / filename)

    monkeypatch.setattr(assets, "hf_hub_download", download)
    monkeypatch.setattr(reference_data, "hf_hub_download", download)
    return data, payloads


def test_provision_tokenizers_fresh_checkout_materializes_pinned_bytes(pinned_assets):
    data, payloads = pinned_assets
    assets.provision_tokenizers(data)
    assert (data / "sft_tokenizer").read_bytes() == payloads["sft_tokenizer"]
    assert (data / "serving_tokenizer/tokenizer.json").read_bytes() == payloads["tokenizer.json"]
    assert (data / "serving_tokenizer/chat_template.jinja").read_bytes() == payloads["chat_template.jinja"]


@pytest.mark.parametrize("path", ["sft_tokenizer", "serving_tokenizer/tokenizer.json"])
def test_provision_tokenizers_rejects_changed_cached_vocabulary(pinned_assets, path):
    data, _ = pinned_assets
    assets.provision_tokenizers(data)
    (data / path).write_bytes(b"changed vocabulary")
    with pytest.raises(ValueError, match="differs from the committed pin"):
        assets.provision_tokenizers(data)
    assert (data / path).read_bytes() == b"changed vocabulary"
