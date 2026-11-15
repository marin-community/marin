# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json

import draccus
import jax.random as jrandom
import numpy as np
import pytest
from haliax import Axis
from levanter.data.text.datasets import DatasetComponent, LmDataConfig, NamedLmDataset
from levanter.data.text.formats import PrebuiltLmDatasetFormat
from levanter.store.cache import SerialCacheWriter
from levanter.utils.thread_utils import blocking_wait
from rigging.fsutil.verified_copy import verified_copy_prefix
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import PreTrainedTokenizerFast

from experiments.post_training.long_loop.exp3_initializer_smoke import initializer_recipe
from experiments.post_training.long_loop.exp3_recipe import exp3_sft_spec, verified_initializer_model
from experiments.post_training.long_loop.sft import BASE_MODEL, BASE_REVISION
from experiments.post_training.long_loop.stage_guard import StageGuard


@pytest.fixture
def staged_initializer(tmp_path):
    blobs = tmp_path / "blobs"
    blobs.mkdir()
    weight = blobs / "weight"
    weight.write_bytes(b"unchanged pretrained tensor bytes")
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "model.safetensors").symlink_to(weight)
    destination = tmp_path / "regional"
    copied = verified_copy_prefix(str(snapshot), str(destination), workers=1)
    manifest_path = destination / ".verified-copy-manifest.json"
    manifest_raw = manifest_path.read_bytes()
    digest = hashlib.sha256(weight.read_bytes()).hexdigest()
    receipt = {
        "repo": BASE_MODEL,
        "revision": BASE_REVISION,
        "model_ref": str(destination),
        "verified_copy_manifest": copied.manifest_url,
        "verified_copy_manifest_sha256": hashlib.sha256(manifest_raw).hexdigest(),
        "files": [
            {"path": "model.safetensors", "size": weight.stat().st_size, "sha256": digest, "lfs": {"sha256": digest}}
        ],
    }
    receipt_path = tmp_path / "source.json"
    raw = json.dumps(receipt).encode()
    receipt_path.write_bytes(raw)
    receipt_sha256 = hashlib.sha256(raw).hexdigest()
    return receipt_path, receipt_sha256, destination, weight


def test_verified_hub_snapshot_resolves_regional_model_and_rejects_changed_copy(staged_initializer):
    receipt_path, receipt_sha256, destination, weight = staged_initializer
    assert verified_initializer_model(str(receipt_path), receipt_sha256) == str(destination)
    assert (destination / "model.safetensors").read_bytes() == weight.read_bytes()
    manifest_path = destination / ".verified-copy-manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    manifest["files"][0]["sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="completion manifest SHA256 mismatch"):
        verified_initializer_model(str(receipt_path), receipt_sha256)


def test_exp3_reserved_validation_remains_disjoint_under_native_lookahead(tmp_path, staged_initializer):
    receipt_path, receipt_sha256, _, _ = staged_initializer
    tokens = {"<pad>": 0, **{f"doc{i}": i for i in range(1, 65)}, "<eos>": 65}
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel(tokens, unk_token="<pad>")), eos_token="<eos>", pad_token="<pad>"
    )
    tokenizer.save_pretrained(tmp_path / "tokenizer")
    cache = tmp_path / "cache"
    lengths = (3, 5, 12, 2, 11, 7, 4, 13)
    with SerialCacheWriter(
        str(cache), {"input_ids": np.empty(0, np.int32), "loss_weights": np.empty(0, np.float32)}
    ) as writer:
        for index in range(1, 65):
            length = lengths[index % len(lengths)]
            weights = np.ones(length, np.float32)
            weights[-1] = 0
            writer.write_batch([{"input_ids": np.full(length, index, np.int32), "loss_weights": weights}])
    data = LmDataConfig(
        tokenizer=str(tmp_path / "tokenizer"),
        components={
            "source": DatasetComponent(
                cache_dir=str(cache),
                flat_cache=True,
                pack=True,
                format=PrebuiltLmDatasetFormat(loss_weights_key="loss_weights"),
            )
        },
        train_weights={"source": 1.0},
        num_validation_sequences={"source": 8},
        shuffle_before_trainval_split=True,
        enforce_eos=False,
        auto_build_caches=False,
    )
    for filename in ("data.json", "c4-data.json"):
        (tmp_path / filename).write_text(json.dumps(draccus.encode(data)))
    (tmp_path / "source-loss.json").write_text(
        json.dumps(
            {
                "version": 1,
                "sources": {
                    "source": {
                        "loss_mode": "full_token",
                        "loss_weights_key": "loss_weights",
                        "loss_weights_indexing": "prediction",
                        "teacher_only": False,
                        "source_revision": "fixture-v1",
                    }
                },
            }
        )
    )
    (tmp_path / "initializer.json").write_text(
        json.dumps(
            {
                "initializer_receipt": str(receipt_path),
                "initializer_receipt_sha256": receipt_sha256,
            }
        )
    )
    recipe = initializer_recipe(
        prepared=str(tmp_path), output=str(tmp_path / "output"), artifact_version="fixture", policy=StageGuard(1, 30, 2)
    )
    spec = exp3_sft_spec("fixture", "fixture", recipe)
    train = NamedLmDataset(
        spec.prepared_data.train_sets(Axis("position", 16), key=jrandom.PRNGKey(0))["source"], Axis("position", 16)
    )
    validation = spec.evaluation_data.validation_sets(Axis("position", 16))["source"]

    def document_ids(dataset):
        count = blocking_wait(dataset.async_len())
        rows = blocking_wait(dataset.get_batch(list(range(count))))
        return {int(token) for row in rows for token in np.asarray(row.tokens.array) if token not in (0, 65)}

    training_ids, validation_ids = document_ids(train), document_ids(validation)
    assert blocking_wait(validation.async_len()) == 8
    assert training_ids.isdisjoint(validation_ids)
    assert training_ids | validation_ids == set(range(1, 65))
