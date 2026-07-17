# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the SFT checkpoint-preparation step and its wiring into ``sft_step``.

Coverage splits three ways: the ArtifactStep wiring (the SFT spec depends on the preparation step
and resolves its model + tokenizer from that step's output; identity tracks the inputs;
``override_path`` pins an existing artifact); the shard surgery (``_reinit_embedding_shards``
reseeds the embedding rows and leaves every other shard byte-for-byte identical); and the tokenizer
rename against the real base tokenizer.
"""
from __future__ import annotations

import hashlib
import json
import os

import numpy as np
import pytest
from fray.types import ResourceConfig
from marin.execution.lazy import materialized_config
from safetensors.numpy import load_file, save_file
from transformers import AutoTokenizer

from experiments.marin_tokenizer import inject_special_tokens
from experiments.sft.configs.delphi_1e22 import DELPHI_1E22_BASE_MODEL, DELPHI_1E22_BASE_REVISION
from experiments.sft.delphi_chat_template import DELPHI_RESERVED_TOKEN_RENAMES
from experiments.sft.launcher import DatasetSpec, HFModel, PreparedModel, SFTSpec, sft_step
from experiments.sft.prepare_checkpoint import (
    PrepareCheckpointConfig,
    _reinit_embedding_shards,
    _reinit_rows,
    prepare_checkpoint_step,
)

_PREFIX = "gs://test-prefix"
_DATASET = DatasetSpec(
    slug="ds",
    hf_dataset_id="some-org/some-dataset",
    revision="main",
    adapter_kwargs=dict(conversation_column="messages"),
    weight=1.0,
)


def _prepared_step(**overrides):
    kwargs = dict(
        name="checkpoints/base-prepared",
        version="2026.07.16",
        source_model="some-org/base-model",
        source_revision="deadbeef",
        token_renames=DELPHI_RESERVED_TOKEN_RENAMES,
    )
    kwargs.update(overrides)
    return prepare_checkpoint_step(**kwargs)


def _spec(model) -> SFTSpec:
    return SFTSpec(
        name="checkpoints/test-sft",
        version="2026.07.16-dev",
        model=model,
        chat_template="{% for m in messages %}{% generation %}{{ m['content'] }}{% endgeneration %}{% endfor %}",
        datasets=[_DATASET],
        num_train_steps=1,
    )


def test_prepared_model_is_a_dependency_and_resolves_both_refs():
    """A PreparedModel makes the prep step a dep; init + tokenizer resolve to its output dir."""
    prep = _prepared_step()
    step = sft_step(_spec(PreparedModel(prep)), ResourceConfig.with_cpu())

    assert prep in step.deps
    prepared_path = prep.path(_PREFIX)
    train_config = materialized_config(step, _PREFIX).train_config
    assert train_config.initialize_from_hf == prepared_path
    assert train_config.data.tokenizer == prepared_path


def test_hf_model_has_no_prep_dep_and_uses_literal_refs():
    """An HFModel adds no model dependency and passes its ids straight through."""
    step = sft_step(_spec(HFModel("Qwen/Qwen3-0.6B")), ResourceConfig.with_cpu())

    # Only the dataset transform is a dependency — no preparation step.
    assert [d.name for d in step.deps] == ["documents/ds"]
    train_config = materialized_config(step, _PREFIX).train_config
    assert train_config.initialize_from_hf == "Qwen/Qwen3-0.6B"
    assert train_config.data.tokenizer == "Qwen/Qwen3-0.6B"


def test_hf_model_separate_tokenizer_path():
    """A distinct tokenizer_path is honored for the data tokenizer, not the init ref."""
    step = sft_step(_spec(HFModel("org/model", tokenizer_path="org/tokenizer")), ResourceConfig.with_cpu())
    train_config = materialized_config(step, _PREFIX).train_config
    assert train_config.initialize_from_hf == "org/model"
    assert train_config.data.tokenizer == "org/tokenizer"


def test_prepare_config_carries_preparation_inputs():
    """The prep step's run config carries the base checkpoint pin and the rename map."""
    prep = _prepared_step()
    config = materialized_config(prep, _PREFIX)
    assert isinstance(config, PrepareCheckpointConfig)
    assert config.source_model == "some-org/base-model"
    assert config.source_revision == "deadbeef"
    assert config.token_renames == DELPHI_RESERVED_TOKEN_RENAMES
    assert config.output_path == prep.path(_PREFIX)


def test_fingerprint_tracks_preparation_inputs():
    """Identity changes when the renames / revision / seed change, and is stable otherwise."""
    base = _prepared_step().fingerprint()
    assert _prepared_step().fingerprint() == base  # deterministic

    renamed = _prepared_step(token_renames={**DELPHI_RESERVED_TOKEN_RENAMES, 128002: "<|other|>"})
    assert renamed.fingerprint() != base
    assert _prepared_step(source_revision="feedface").fingerprint() != base
    assert _prepared_step(seed=1).fingerprint() != base


def test_prepare_resources_do_not_affect_identity():
    """Where the prep runs is a runtime choice, not part of the fingerprint."""
    base = _prepared_step().fingerprint()
    big = _prepared_step(resources=ResourceConfig.with_cpu(cpu=32, ram="256g"))
    assert big.fingerprint() == base


def test_override_path_adopts_existing_checkpoint():
    """override_path pins an existing prepared checkpoint (adopted, not recomputed)."""
    pinned = _prepared_step(override_path="gs://staged/base-prepared")
    assert pinned.path(_PREFIX) == "gs://staged/base-prepared"
    assert pinned.adopt_source == "gs://staged/base-prepared"

    step = sft_step(_spec(PreparedModel(pinned)), ResourceConfig.with_cpu())
    train_config = materialized_config(step, _PREFIX).train_config
    assert train_config.initialize_from_hf == "gs://staged/base-prepared"
    assert train_config.data.tokenizer == "gs://staged/base-prepared"


def _sha256(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def _write_tiny_checkpoint(path: str, vocab: int = 16, hidden: int = 8) -> None:
    """A two-shard checkpoint: embed + lm_head + a layer tensor in shard 1, another in shard 2."""
    rng = np.random.default_rng(123)
    shard1 = "model-00001-of-00002.safetensors"
    shard2 = "model-00002-of-00002.safetensors"
    save_file(
        {
            "model.embed_tokens.weight": rng.standard_normal((vocab, hidden), dtype=np.float32),
            "lm_head.weight": rng.standard_normal((vocab, hidden), dtype=np.float32),
            "model.layers.0.mlp.weight": rng.standard_normal((hidden, hidden), dtype=np.float32),
        },
        os.path.join(path, shard1),
    )
    save_file(
        {"model.layers.1.mlp.weight": rng.standard_normal((hidden, hidden), dtype=np.float32)},
        os.path.join(path, shard2),
    )
    weight_map = {
        "model.embed_tokens.weight": shard1,
        "lm_head.weight": shard1,
        "model.layers.0.mlp.weight": shard1,
        "model.layers.1.mlp.weight": shard2,
    }
    with open(os.path.join(path, "model.safetensors.index.json"), "w") as f:
        json.dump({"metadata": {"total_size": 0}, "weight_map": weight_map}, f)


def test_reinit_rows_reseeds_only_target_rows():
    """``_reinit_rows`` changes exactly the target rows, preserves dtype, and is deterministic."""
    matrix = np.arange(16 * 8, dtype=np.float32).reshape(16, 8)
    out = _reinit_rows(matrix, [2, 5], np.random.default_rng(0))

    assert out.dtype == matrix.dtype
    assert np.array_equal(matrix, np.arange(16 * 8, dtype=np.float32).reshape(16, 8))  # input untouched
    changed = {i for i in range(matrix.shape[0]) if not np.array_equal(out[i], matrix[i])}
    assert changed == {2, 5}
    assert np.array_equal(out, _reinit_rows(matrix, [2, 5], np.random.default_rng(0)))  # deterministic


def test_reinit_embedding_shards_rewrites_only_the_embedding_shard(tmp_path):
    """Only the shard holding embed/lm_head is rewritten; every other shard stays byte-for-byte."""
    d = str(tmp_path)
    _write_tiny_checkpoint(d)
    shard1, shard2 = "model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"
    before = load_file(os.path.join(d, shard1))
    sha2_before = _sha256(os.path.join(d, shard2))

    _reinit_embedding_shards(d, ids=[2, 5], seed=0)

    after = load_file(os.path.join(d, shard1))
    # The shard with no embedding tensors is never rewritten -> identical bytes on disk.
    assert _sha256(os.path.join(d, shard2)) == sha2_before
    # A non-embedding tensor inside the rewritten shard is preserved exactly.
    assert np.array_equal(after["model.layers.0.mlp.weight"], before["model.layers.0.mlp.weight"])
    # Exactly rows 2 and 5 of both embed and lm_head are reseeded.
    for name in ("model.embed_tokens.weight", "lm_head.weight"):
        changed = {i for i in range(before[name].shape[0]) if not np.array_equal(after[name][i], before[name][i])}
        assert changed == {2, 5}, name


def test_delphi_renames_produce_single_ids_on_the_real_tokenizer():
    """Regenerate the Delphi tokenizer half: the think/tool strings become single ids.

    Downloads the public base tokenizer; skipped when the Hub is unreachable. This is the
    reproducibility check for the rename half of the preparation.
    """
    try:
        raw = AutoTokenizer.from_pretrained(DELPHI_1E22_BASE_MODEL, revision=DELPHI_1E22_BASE_REVISION)
    except Exception as e:  # gated/offline — this test needs the real base tokenizer
        pytest.skip(f"Delphi base tokenizer unavailable: {e}")

    # In the raw tokenizer the canonical strings are not single ids (the bug this fixes).
    assert len(raw.tokenize("<|start_think|>")) > 1

    prepared = inject_special_tokens(raw, dict(DELPHI_RESERVED_TOKEN_RENAMES))
    for token_id, token_str in DELPHI_RESERVED_TOKEN_RENAMES.items():
        assert prepared.encode(token_str, add_special_tokens=False) == [token_id]
        assert prepared.decode([token_id]) == token_str
