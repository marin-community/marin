# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for generation config validation and normalization."""

import json
from types import SimpleNamespace
from typing import Any, cast

import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.normalizers import Lowercase
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from transformers import AutoTokenizer, PreTrainedTokenizerFast

from levanter.compat.hf_checkpoints import (
    _save_tokenizer_pretrained,
    build_generation_config,
    save_hf_checkpoint_callback,
)


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


def test_save_tokenizer_pretrained_exports_portable_generic_fast_tokenizer(tmp_path):
    backend = Tokenizer(
        WordLevel(
            vocab={"[UNK]": 0, "<s>": 1, "</s>": 2, "hello": 3, "world": 4, "user": 5, ":": 6},
            unk_token="[UNK]",
        )
    )
    backend.normalizer = Lowercase()
    backend.pre_tokenizer = Whitespace()
    backend.post_processor = TemplateProcessing(
        single="<s> $A </s>",
        special_tokens=[("<s>", 1), ("</s>", 2)],
    )
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="[UNK]",
        bos_token="<s>",
        eos_token="</s>",
    )
    tokenizer.add_tokens(["portable"])
    tokenizer.chat_template = (
        "{% for message in messages %}{{ message['role'] }}: " "{{ message['content'] }}{{ eos_token }}{% endfor %}"
    )

    text = "HELLO portable mystery"
    encoding = tokenizer(text, return_offsets_mapping=True, return_special_tokens_mask=True)
    messages = [{"role": "user", "content": "HELLO world"}]

    _save_tokenizer_pretrained(tokenizer, str(tmp_path))

    tokenizer_config = json.loads((tmp_path / "tokenizer_config.json").read_text())
    assert tokenizer_config["tokenizer_class"] == "PreTrainedTokenizerFast"
    assert json.loads((tmp_path / "tokenizer.json").read_text()) == json.loads(tokenizer.backend_tokenizer.to_str())

    reloaded = AutoTokenizer.from_pretrained(tmp_path, local_files_only=True)
    assert reloaded(text, return_offsets_mapping=True, return_special_tokens_mask=True) == encoding
    assert reloaded.get_vocab() == tokenizer.get_vocab()
    assert reloaded.special_tokens_map == tokenizer.special_tokens_map
    assert reloaded.apply_chat_template(messages, tokenize=True) == tokenizer.apply_chat_template(
        messages, tokenize=True
    )
    assert reloaded.decode(encoding["input_ids"]) == tokenizer.decode(encoding["input_ids"])


def test_save_tokenizer_pretrained_preserves_specialized_tokenizer_class(local_gpt2_tokenizer_path, tmp_path):
    tokenizer = AutoTokenizer.from_pretrained(local_gpt2_tokenizer_path, local_files_only=True)
    assert tokenizer.__class__ is not PreTrainedTokenizerFast
    text = "Marin tokenizer compatibility"
    encoding = tokenizer(text, return_offsets_mapping=True, return_special_tokens_mask=True)

    _save_tokenizer_pretrained(tokenizer, str(tmp_path))

    tokenizer_config = json.loads((tmp_path / "tokenizer_config.json").read_text())
    assert tokenizer_config["tokenizer_class"] == "GPT2Tokenizer"
    reloaded = AutoTokenizer.from_pretrained(tmp_path, local_files_only=True)
    assert reloaded(text, return_offsets_mapping=True, return_special_tokens_mask=True) == encoding
