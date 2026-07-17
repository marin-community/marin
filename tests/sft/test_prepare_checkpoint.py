# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the SFT checkpoint-preparation step and its wiring into ``sft_step``.

Coverage splits three ways: the ArtifactStep wiring (the SFT spec depends on the preparation step
and resolves its model + tokenizer from that step's output; identity tracks the inputs;
``override_path`` pins an existing artifact); the shard surgery (``_reinit_shard_bytes`` reseeds the
embedding rows and round-trips every other tensor unchanged); and the tokenizer rename against the
real base tokenizer.
"""
from __future__ import annotations

import numpy as np
import pytest
from fray.types import ResourceConfig
from marin.execution.lazy import materialized_config
from safetensors.numpy import load, save
from transformers import AutoTokenizer

from experiments.marin_tokenizer import inject_special_tokens
from experiments.sft.configs.delphi_1e22 import DELPHI_1E22_BASE_MODEL, DELPHI_1E22_BASE_REVISION
from experiments.sft.delphi_chat_template import DELPHI_RESERVED_TOKEN_RENAMES
from experiments.sft.launcher import DatasetSpec, HFModel, PreparedModel, SFTSpec, sft_step
from experiments.sft.prepare_checkpoint import (
    PrepareCheckpointConfig,
    _reinit_rows,
    _reinit_shard_bytes,
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


def test_reinit_rows_reseeds_only_target_rows():
    """``_reinit_rows`` changes exactly the target rows, preserves dtype, and is deterministic."""
    matrix = np.arange(16 * 8, dtype=np.float32).reshape(16, 8)
    out = _reinit_rows(matrix, [2, 5], np.random.default_rng(0))

    assert out.dtype == matrix.dtype
    assert np.array_equal(matrix, np.arange(16 * 8, dtype=np.float32).reshape(16, 8))  # input untouched
    changed = {i for i in range(matrix.shape[0]) if not np.array_equal(out[i], matrix[i])}
    assert changed == {2, 5}
    assert np.array_equal(out, _reinit_rows(matrix, [2, 5], np.random.default_rng(0)))  # deterministic


def test_reinit_shard_bytes_reseeds_embeddings_and_round_trips_the_rest():
    """Only the embed/lm_head rows change; other tensors and the metadata survive the rewrite."""
    rng = np.random.default_rng(123)
    vocab, hidden = 16, 8
    tensors = {
        "model.embed_tokens.weight": rng.standard_normal((vocab, hidden), dtype=np.float32),
        "lm_head.weight": rng.standard_normal((vocab, hidden), dtype=np.float32),
        "model.layers.0.mlp.weight": rng.standard_normal((hidden, hidden), dtype=np.float32),
    }
    out = _reinit_shard_bytes(save(tensors, metadata={"format": "pt"}), ids=[2, 5], seed=0)

    reloaded = load(out)
    # A non-embedding tensor is round-tripped unchanged.
    assert np.array_equal(reloaded["model.layers.0.mlp.weight"], tensors["model.layers.0.mlp.weight"])
    # Exactly rows 2 and 5 of both embed and lm_head are reseeded.
    for name in ("model.embed_tokens.weight", "lm_head.weight"):
        changed = {i for i in range(vocab) if not np.array_equal(reloaded[name][i], tensors[name][i])}
        assert changed == {2, 5}, name
    # Deterministic for a given seed.
    assert _reinit_shard_bytes(save(tensors, metadata={"format": "pt"}), ids=[2, 5], seed=0) == out


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
