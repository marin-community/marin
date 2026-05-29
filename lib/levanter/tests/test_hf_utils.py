# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import os

import fsspec
import huggingface_hub
import pytest
from fsspec import AbstractFileSystem
from test_utils import skip_if_hf_model_not_accessible

from levanter.compat.hf_checkpoints import _patch_hf_hub_download, load_tokenizer
from levanter.utils.hf_utils import byte_length_of_token


class _FakeTokenMonsterTokenizer:
    _tokens = {
        37: ("D", "", "single"),
        517: ("DC", "", "regular"),
        82: ("\ufffd", "\ufffd", "single"),
        114: ("\ufffd", "\ufffd", "single"),
        171: ("\ufffd", "\ufffd", "single"),
        8231: (" hello", " hello", "regular"),
        12709: ("C world", " World", "regular"),
        999: ("<eos>", "<eos>", "special"),
    }
    all_special_ids = [999]

    def convert_ids_to_tokens(self, idx):
        return self._tokens[idx][0]

    def id_to_token_decoded(self, idx):
        return self._tokens[idx][1]

    def get_dictionary(self):
        return {
            idx: {"id": idx, "token": token, "token_decoded": decoded, "type": token_type}
            for idx, (token, decoded, token_type) in self._tokens.items()
        }


def test_byte_length_of_token_tokenmonster_capcode_markers():
    tok = _FakeTokenMonsterTokenizer()

    assert byte_length_of_token(tok, 37) == -1
    assert byte_length_of_token(tok, 517) == -1
    assert byte_length_of_token(tok, 8231) == len(" hello".encode("utf-8"))
    assert byte_length_of_token(tok, 12709) == len(" World".encode("utf-8"))
    assert byte_length_of_token(tok, 999) == 0

    # TokenMonster decodes "DC" + " hello" + "C world" as "Hello World".
    assert sum(byte_length_of_token(tok, token_id) for token_id in [517, 8231, 12709]) == len(
        "Hello World".encode("utf-8")
    )
    # Byte fallback tokens can show up as replacement characters individually.
    assert sum(byte_length_of_token(tok, token_id) for token_id in [171, 82, 114]) == len("⇧".encode("utf-8"))


def test_tokenmonster_byte_offsets_handle_normalized_unicode():
    pytest.importorskip("tokenmonster")
    from levanter.tokenizers import TokenizerBackend
    from levanter.tokenizers import load_tokenizer as load_marin_tokenizer

    tok = load_marin_tokenizer(
        "tokenmonster:englishcode-32000-consistent-v1",
        backend=TokenizerBackend.TOKENMONSTER,
    )
    text = "Stefan Talijanović"

    ids, starts, ends, num_bytes = tok.tokenize_with_byte_offsets(text)

    assert tok.decode(ids) == "Stefan Talijanovic\u0301"
    assert num_bytes == len(text.encode("utf-8"))
    assert all((start < 0 and end < 0) or (0 <= start < end <= num_bytes) for start, end in zip(starts, ends))
    assert max(end for end in ends if end >= 0) == num_bytes


def test_load_tokenizer_in_memory_fs():
    # sort of like a gs:// path insasmuch as it uses fsspec machinery
    fs: AbstractFileSystem = fsspec.filesystem("memory")
    directory_of_this_test = os.path.dirname(os.path.abspath(__file__))
    fs.put(f"{directory_of_this_test}/gpt2_tokenizer_config.json", "memory://foo/tokenizer_config.json")
    fs.put(f"{directory_of_this_test}/gpt2_tokenizer_config.json", "memory://foo/tokenizer.json")

    with fsspec.open("memory://foo/config.json", "w") as f:
        f.write(
            """{
         "model_type": "gpt2",
         "vocab_size": 5027
         }"""
        )
    tokenizer = load_tokenizer("memory://foo/")
    assert len(tokenizer) == 5027


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
