# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for generation config validation and normalization."""

import json
from types import SimpleNamespace
from typing import Any, cast

import haliax as hax
import pytest
from jax import random
from test_utils import use_test_mesh

from transformers import Gemma3TextConfig as HFGemma3Config

from levanter.compat.hf_checkpoints import (
    HFCheckpointConverter,
    _save_tokenizer_pretrained,
    build_generation_config,
    save_hf_checkpoint_callback,
)
from levanter.layers.rotary import DefaultRotaryEmbeddingsConfig, Llama3RotaryEmbeddingsConfig
from levanter.models.gemma import Gemma3Config, Gemma3LMHeadModel
from levanter.models.llama import LlamaConfig, LlamaLMHeadModel


class _FakeTokenizer:
    """Minimal tokenizer stub for testing build_generation_config."""

    def __init__(self, vocab_size: int = 200, eos_token_id: int | None = 2, bos_token_id: int | None = 1):
        self._vocab_size = vocab_size
        self.eos_token_id = eos_token_id
        self.bos_token_id = bos_token_id

    def __len__(self):
        return self._vocab_size

    def convert_ids_to_tokens(self, tid: int) -> str | None:
        if 0 <= tid < self._vocab_size:
            return f"<tok_{tid}>"
        return None


class TestBuildGenerationConfig:
    def test_none_returns_none(self):
        tok = _FakeTokenizer()
        assert build_generation_config(tok, None) is None

    def test_valid_ids(self):
        tok = _FakeTokenizer(vocab_size=200, eos_token_id=2, bos_token_id=1)
        result = build_generation_config(tok, [50])
        assert result is not None
        assert result["eos_token_id"] == [2, 50]
        assert result["bos_token_id"] == 1

    def test_deduplication(self):
        tok = _FakeTokenizer(vocab_size=200, eos_token_id=2)
        result = build_generation_config(tok, [50, 50, 2])
        assert result is not None
        assert result["eos_token_id"] == [2, 50]

    def test_sorted_output(self):
        tok = _FakeTokenizer(vocab_size=200, eos_token_id=2)
        result = build_generation_config(tok, [100, 50, 75])
        assert result is not None
        assert result["eos_token_id"] == [2, 50, 75, 100]

    def test_deterministic(self):
        tok = _FakeTokenizer(vocab_size=200, eos_token_id=2)
        r1 = build_generation_config(tok, [128, 64, 128])
        r2 = build_generation_config(tok, [64, 128, 64])
        assert r1 == r2

    def test_auto_adds_tokenizer_eos(self):
        tok = _FakeTokenizer(vocab_size=200, eos_token_id=2)
        result = build_generation_config(tok, [50])
        assert result is not None
        assert 2 in result["eos_token_id"]

    def test_eos_already_included(self):
        tok = _FakeTokenizer(vocab_size=200, eos_token_id=2)
        result = build_generation_config(tok, [2, 50])
        assert result is not None
        assert result["eos_token_id"] == [2, 50]

    def test_tokenizer_eos_none(self):
        tok = _FakeTokenizer(vocab_size=200, eos_token_id=None)
        result = build_generation_config(tok, [50])
        assert result is not None
        assert result["eos_token_id"] == [50]

    def test_bos_included_when_present(self):
        tok = _FakeTokenizer(vocab_size=200, eos_token_id=2, bos_token_id=1)
        result = build_generation_config(tok, [50])
        assert result is not None
        assert result["bos_token_id"] == 1

    def test_bos_omitted_when_none(self):
        tok = _FakeTokenizer(vocab_size=200, eos_token_id=2, bos_token_id=None)
        result = build_generation_config(tok, [50])
        assert result is not None
        assert "bos_token_id" not in result

    def test_empty_list_raises(self):
        tok = _FakeTokenizer()
        with pytest.raises(ValueError, match="non-empty"):
            build_generation_config(tok, [])

    def test_non_int_raises(self):
        tok = _FakeTokenizer()
        with pytest.raises(ValueError, match="non-int"):
            build_generation_config(tok, [1, "two"])  # type: ignore[list-item]

    def test_out_of_range_raises(self):
        tok = _FakeTokenizer(vocab_size=100)
        with pytest.raises(ValueError, match="out of range"):
            build_generation_config(tok, [999])

    def test_negative_id_raises(self):
        tok = _FakeTokenizer(vocab_size=100)
        with pytest.raises(ValueError, match="out of range"):
            build_generation_config(tok, [-1])


class _CapturingConverter:
    def __init__(self):
        self.calls = []

    def save_pretrained(self, model, path, **kwargs):
        self.calls.append((model, path, kwargs))


def test_save_hf_checkpoint_callback_passes_generation_config():
    converter = _CapturingConverter()
    generation_config = {"eos_token_id": [2, 50]}
    callback = save_hf_checkpoint_callback("/tmp/export", converter, generation_config=generation_config)

    model = object()
    callback(SimpleNamespace(step=1, eval_model=model))

    assert len(converter.calls) == 1
    saved_model, saved_path, saved_kwargs = converter.calls[0]
    assert saved_model is model
    assert saved_path == "/tmp/export/step-1"
    assert saved_kwargs["generation_config"] == generation_config


class _FakeChatTemplateTokenizer:
    def __init__(self, chat_template: str):
        self.chat_template = chat_template

    def save_pretrained(self, path: str) -> None:
        with open(f"{path}/tokenizer_config.json", "w") as f:
            json.dump({"tokenizer_class": "FakeTokenizer"}, f)


def test_save_tokenizer_pretrained_embeds_chat_template(tmp_path):
    tokenizer = _FakeChatTemplateTokenizer("{{ bos_token }}{{ messages[0]['content'] }}")

    _save_tokenizer_pretrained(cast(Any, tokenizer), str(tmp_path))

    tokenizer_config = json.loads((tmp_path / "tokenizer_config.json").read_text())
    assert tokenizer_config["chat_template"] == tokenizer.chat_template


def _exported_config_json(rope, tmp_path):
    config = LlamaConfig(
        max_seq_len=128,
        hidden_dim=32,
        intermediate_dim=64,
        num_layers=2,
        num_heads=4,
        num_kv_heads=4,
        rope=rope,
        reference_checkpoint=None,
        tokenizer=None,
    )
    converter = config.hf_checkpoint_converter()
    out = str(tmp_path / "model")
    with use_test_mesh():
        model = LlamaLMHeadModel.init(hax.Axis("vocab", 64), config, key=random.PRNGKey(0))
        converter.save_pretrained(model, out, save_reference_code=False, save_tokenizer=False)
    return json.loads((tmp_path / "model" / "config.json").read_text())


def test_export_emits_both_new_and_legacy_rope_keys(tmp_path):
    cfg = _exported_config_json(Llama3RotaryEmbeddingsConfig(theta=500000.0), tmp_path)

    assert "rope_parameters" in cfg  # new-style (5.x) key
    assert cfg["rope_theta"] == 500000.0  # legacy key, at the trained value rather than the 10000 default
    assert cfg["rope_scaling"]["rope_type"] == "llama3"
    assert cfg["rope_scaling"]["factor"] == 8.0


def test_export_default_rope_omits_legacy_scaling(tmp_path):
    cfg = _exported_config_json(DefaultRotaryEmbeddingsConfig(theta=10000.0), tmp_path)

    assert cfg["rope_theta"] == 10000.0
    assert "rope_scaling" not in cfg


def test_export_linear_rope_keeps_scaling_factor(tmp_path):
    cfg = _exported_config_json(DefaultRotaryEmbeddingsConfig(theta=10000.0, factor=2.0), tmp_path)

    assert cfg["rope_theta"] == 10000.0
    assert cfg["rope_scaling"] == {"rope_type": "linear", "factor": 2.0}


def test_export_per_layer_rope_backfills_from_full_attention_block(tmp_path):
    config = Gemma3Config(
        max_seq_len=128,
        hidden_dim=16,
        intermediate_dim=64,
        num_layers=2,
        num_heads=4,
        num_kv_heads=4,
        head_dim=4,
        query_pre_attn_scalar=4,
        sliding_window=128,
        rope=DefaultRotaryEmbeddingsConfig(theta=1000000.0),
    )
    converter = HFCheckpointConverter(Gemma3Config, reference_checkpoint=None, HfConfigClass=HFGemma3Config)
    out = str(tmp_path / "model")
    with use_test_mesh():
        model = Gemma3LMHeadModel.init(hax.Axis("vocab", 64), config, key=random.PRNGKey(0))
        converter.save_pretrained(model, out, save_reference_code=False, save_tokenizer=False)
    cfg = json.loads((tmp_path / "model" / "config.json").read_text())

    # Gemma3 stores per-layer rope_parameters with no flat rope_theta. The legacy top-level
    # rope_theta corresponds to the full_attention block, so the back-fill must pull the trained
    # value from there — never inject rope_theta: null or a rope_scaling block built from the
    # nested per-layer dicts.
    assert cfg["rope_parameters"]["full_attention"]["rope_theta"] == 1000000.0
    assert cfg["rope_theta"] == 1000000.0
    assert "rope_scaling" not in cfg
