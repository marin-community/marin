# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import os
from pathlib import Path

import huggingface_hub
import pytest
from rigging.filesystem import StoragePath
from test_utils import skip_if_hf_model_not_accessible
from transformers import GPT2Config as HfGpt2Config

from levanter.compat.hf_checkpoints import (
    HFCheckpointConverter,
    _load_hf_config,
    _patch_hf_hub_download,
    _stage_fsspec_config_with_retry,
    _stage_fsspec_tokenizer,
    load_tokenizer,
)
from levanter.models.gpt2 import Gpt2Config
from levanter.utils.hf_utils import byte_length_of_token


def test_load_tokenizer_in_memory_fs():
    # sort of like a gs:// path insasmuch as it uses fsspec machinery
    directory_of_this_test = os.path.dirname(os.path.abspath(__file__))
    StoragePath("memory://foo/tokenizer.json").upload_from(f"{directory_of_this_test}/gpt2_tokenizer.json")
    StoragePath("memory://foo/tokenizer_config.json").upload_from(
        f"{directory_of_this_test}/gpt2_tokenizer_config.json"
    )

    StoragePath("memory://foo/config.json").write_text(
        """{
     "model_type": "gpt2",
     "vocab_size": 5027
     }"""
    )
    tokenizer = load_tokenizer("memory://foo/")
    assert len(tokenizer) == 5027


def test_load_tokenizers_backend_from_memory_fs(tmp_path):
    directory_of_this_test = os.path.dirname(os.path.abspath(__file__))
    checkpoint = "memory://tokenizers-backend-checkpoint"
    StoragePath(f"{checkpoint}/tokenizer.json").upload_from(f"{directory_of_this_test}/gpt2_tokenizer.json")
    StoragePath(f"{checkpoint}/tokenizer_config.json").write_text(
        """{
        "tokenizer_class": "TokenizersBackend",
        "bos_token": "<|endoftext|>",
        "eos_token": "<|endoftext|>",
        "unk_token": "<|endoftext|>"
        }"""
    )
    StoragePath(f"{checkpoint}/model.safetensors").write_bytes(b"must not be staged")

    staged_dir = tmp_path / "staged"
    staged_dir.mkdir()
    _stage_fsspec_tokenizer(checkpoint, str(staged_dir), trust_remote_code=False)
    tokenizer = load_tokenizer(checkpoint)

    assert (staged_dir / "tokenizer.json").is_file()
    assert (staged_dir / "tokenizer_config.json").is_file()
    assert not (staged_dir / "model.safetensors").exists()
    assert len(tokenizer) == 5027
    assert tokenizer.eos_token_id == 5026


def test_hf_hub_download_handles_long_fsspec_url():
    checkpoint = "memory://" + "long-checkpoint-segment-" * 16
    StoragePath(f"{checkpoint}/config.json").write_text('{"model_type": "gpt2"}')

    with _patch_hf_hub_download() as download:
        local_path = Path(download(checkpoint, "config.json"))
        assert local_path.read_text() == '{"model_type": "gpt2"}'
        assert max(len(part) for part in local_path.parts) < 256


def test_load_hf_config_from_memory_fs(tmp_path):
    checkpoint = "memory://hf-config-checkpoint"
    StoragePath(f"{checkpoint}/config.json").write_text(
        """{
        "model_type": "gpt2",
        "n_embd": 48,
        "n_head": 4,
        "n_layer": 2,
        "vocab_size": 5027
        }"""
    )
    StoragePath(f"{checkpoint}/model.safetensors").write_bytes(b"must not be staged")

    staged_dir = tmp_path / "config"
    staged_dir.mkdir()
    _stage_fsspec_config_with_retry(checkpoint, str(staged_dir), trust_remote_code=False)
    inferred = _load_hf_config(
        checkpoint,
        revision=None,
        trust_remote_code=False,
    )
    converter = HFCheckpointConverter(
        Gpt2Config,
        reference_checkpoint=checkpoint,
        HfConfigClass=HfGpt2Config,
        tokenizer=object(),
    )
    explicit = converter.hf_config_from_hf_checkpoint()

    assert isinstance(inferred, HfGpt2Config)
    assert isinstance(explicit, HfGpt2Config)
    assert inferred.n_embd == explicit.n_embd == 48
    assert inferred.vocab_size == explicit.vocab_size == 5027
    assert (staged_dir / "config.json").is_file()
    assert not (staged_dir / "model.safetensors").exists()


def test_load_hf_config_requires_root_config():
    checkpoint = "memory://nested-hf-config-checkpoint"
    StoragePath(f"{checkpoint}/nested/config.json").write_text('{"model_type": "gpt2"}')

    with pytest.raises(FileNotFoundError, match="No root config.json"):
        _load_hf_config(
            checkpoint,
            revision=None,
            trust_remote_code=False,
        )


def test_load_hf_config_rejects_revision_for_url():
    with pytest.raises(ValueError, match="Revisions not supported for explicit URLs"):
        _load_hf_config(
            "memory://revisioned-hf-config-checkpoint",
            revision="main",
            trust_remote_code=False,
        )


def test_model_info_patch_for_fsspec_urls():
    """transformers calls model_info() in _patch_mistral_regex to check if a model is a base Mistral model."""

    with _patch_hf_hub_download():
        # This should NOT raise or make a network call - it should return a mock
        result = huggingface_hub.hf_api.model_info("memory://some/path")
        assert result.id == "monkeypatched"
        assert result.tags is None


@skip_if_hf_model_not_accessible("NousResearch/Llama-2-7b-hf")
def test_byte_length_of_token():
    tok = load_tokenizer("NousResearch/Llama-2-7b-hf")
    ids = tok("this is hello a test", add_special_tokens=False)["input_ids"]
    assert byte_length_of_token(tok, ids[2]) == len(" hello".encode("utf-8"))
    assert byte_length_of_token(tok, 25) == 1
    # llama prepends a space to the string. ideally it wouldn't b/c it technically throws off our bpb calculations
    # but it's a small difference
    assert byte_length_of_token(tok, ids[0]) == len(" this".encode("utf-8"))

    bos = tok.bos_token_id
    assert byte_length_of_token(tok, bos) == 0

    # 632: "▁▁▁▁▁▁▁▁▁▁▁▁" which is just 12 spaces
    # assert byte_length_of_token(tok, 632) == len("            ".encode("utf-8"))
    # 8535: "ными"
    # assert byte_length_of_token(tok, 8535) == len("ными".encode("utf-8"))

    checks = {
        632: " " * 12,
        8535: "ными",
        25: " ",
    }

    for token_id, expected_length in checks.items():
        assert byte_length_of_token(tok, token_id) == len(expected_length.encode("utf-8"))

    # now just test all tokens and print the ones that aren't expected
    # the ones less than 259 are bytes or special tokens
    for i in range(3, 259):
        byte_length = byte_length_of_token(tok, i)
        assert byte_length == 1, f"Token {i} has length {byte_length} but expected 1"

    for i in range(259, tok.vocab_size):
        byte_length = byte_length_of_token(tok, i)
        expected_length = len(tok.convert_ids_to_tokens(i).replace("▁", " ").encode("utf-8"))
        assert byte_length == expected_length, f"Token {i} has length {byte_length} but expected {expected_length}"


@skip_if_hf_model_not_accessible("NousResearch/Llama-2-7b-hf")
def test_byte_length_of_token_multi():
    tok = load_tokenizer("NousResearch/Llama-2-7b-hf")
    multi_checks = [
        "👍你好",
    ]

    for expr in multi_checks:
        # stupid llama adds a prefix space
        token_ids = tok.encode(expr, add_special_tokens=False)[1:]
        total_length = sum(byte_length_of_token(tok, token_id) for token_id in token_ids)
        assert total_length == len(expr.encode("utf-8"))


@skip_if_hf_model_not_accessible("gpt2")
def test_byte_length_of_token_gpt2():
    tok = load_tokenizer("gpt2")
    ids = tok("this is hello a test", add_special_tokens=False)["input_ids"]
    assert byte_length_of_token(tok, ids[2]) == len(" hello".encode("utf-8"))

    eos = tok.eos_token_id
    assert byte_length_of_token(tok, eos) == 0
