# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import io
from pathlib import Path

import equinox as eqx
import jax.random as jr
import numpy as np
import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

from experiments.datakit.cluster.quality.fast_transformer.model import (
    FastEmbeddingTransformer,
    FastTransformerConfig,
)
from experiments.datakit.embeddings.fast_transformer.embedder import (
    MANIFEST_FILENAME,
    FastEmbeddingBundleManifest,
    FastEmbeddingModel,
    document_view,
    payload_sha256,
)


def write_test_bundle(root: Path) -> str:
    config = FastTransformerConfig(
        vocab_size=5,
        max_tokens=8,
        pool_window=4,
        pool_kind="meanmaxmin",
        embed_dim=8,
        hidden_dim=8,
        num_layers=1,
        num_heads=2,
        dropout=0.0,
    )
    model = FastEmbeddingTransformer(config, output_dim=6, key=jr.PRNGKey(7))
    model_buffer = io.BytesIO()
    eqx.tree_serialise_leaves(model_buffer, model)
    model_payload = model_buffer.getvalue()
    remap_buffer = io.BytesIO()
    np.save(remap_buffer, np.asarray([1, 2, 3, 4], dtype=np.int32))
    remap_payload = remap_buffer.getvalue()
    tokenizer = Tokenizer(WordLevel({"[UNK]": 0, "alpha": 1, "beta": 2, "gamma": 3}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer_payload = tokenizer.to_str().encode()
    (root / "model.eqx").write_bytes(model_payload)
    (root / "raw-to-compact.npy").write_bytes(remap_payload)
    (root / "tokenizer.json").write_bytes(tokenizer_payload)
    manifest = FastEmbeddingBundleManifest(
        model_filename="model.eqx",
        model_sha256=payload_sha256(model_payload),
        token_remap_filename="raw-to-compact.npy",
        token_remap_sha256=payload_sha256(remap_payload),
        tokenizer_filename="tokenizer.json",
        tokenizer_sha256=payload_sha256(tokenizer_payload),
        tokenizer_name="test-word-level",
        raw_vocab_size=4,
        config=config,
        output_dimension=6,
        characters_per_region=8,
        training_report_url="memory://training.json",
        training_report_sha256="0" * 64,
        evaluation_report_url="memory://evaluation.json",
        evaluation_report_sha256="1" * 64,
        speed_report_url="memory://speed.json",
        speed_report_sha256="2" * 64,
        blind_review_report_url="memory://blind-review.json",
        blind_review_report_sha256="3" * 64,
        blind_review_package_url="memory://blind-package.json.gz",
        blind_review_package_sha256="4" * 64,
    )
    manifest_payload = manifest.model_dump_json().encode()
    (root / MANIFEST_FILENAME).write_bytes(manifest_payload)
    return payload_sha256(manifest_payload)


def test_fast_embedding_bundle_round_trip_returns_distinct_unit_vectors(tmp_path: Path) -> None:
    manifest_sha256 = write_test_bundle(tmp_path)

    embedder = FastEmbeddingModel.load(str(tmp_path), manifest_sha256)
    vectors = embedder(["alpha beta", "gamma alpha"])

    assert vectors.shape == (2, 6)
    assert np.isfinite(vectors).all()
    assert np.linalg.norm(vectors, axis=1) == pytest.approx([1.0, 1.0], abs=1e-6)
    assert not np.allclose(vectors[0], vectors[1])


def test_fast_embedding_bundle_rejects_changed_model(tmp_path: Path) -> None:
    manifest_sha256 = write_test_bundle(tmp_path)
    model_path = tmp_path / "model.eqx"
    model_path.write_bytes(model_path.read_bytes() + b"changed")

    with pytest.raises(ValueError, match=r"digest for model\.eqx"):
        FastEmbeddingModel.load(str(tmp_path), manifest_sha256)


def test_document_view_keeps_fixed_head_middle_and_tail() -> None:
    text = "a" * 20 + "b" * 20 + "c" * 20

    assert document_view(text, 8) == "a" * 8 + "\n" + "b" * 8 + "\n" + "c" * 8
