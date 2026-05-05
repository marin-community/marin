# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import json
import os
import shutil
import tempfile
from pathlib import Path

import pytest

from levanter.tokenizers import HfMarinTokenizer, load_tokenizer

pytest_plugins = ["tests.test_utils"]


def _gpt2_tokenizer_dir(tmp_path_factory):
    """Create a local directory with the GPT-2 tokenizer files."""
    config_src = Path(__file__).parent / "gpt2_tokenizer_config.json"
    tmpdir = tmp_path_factory.mktemp("gpt2_tok")
    shutil.copy(config_src, tmpdir / "tokenizer.json")
    shutil.copy(config_src, tmpdir / "tokenizer_config.json")
    (tmpdir / "config.json").write_text(json.dumps({"model_type": "gpt2", "vocab_size": 5027}))
    return tmpdir


@pytest.fixture(scope="session")
def local_gpt2_tokenizer(tmp_path_factory):
    """Load a GPT2 MarinTokenizer from a local directory to avoid network downloads."""
    tmpdir = _gpt2_tokenizer_dir(tmp_path_factory)
    return load_tokenizer(str(tmpdir))


@pytest.fixture(scope="session")
def local_gpt2_marin_tokenizer(tmp_path_factory) -> HfMarinTokenizer:
    """Load a GPT2 MarinTokenizer from a local JSON file to avoid network downloads."""
    from tokenizers import Tokenizer as HfBaseTokenizer

    config_src = Path(__file__).parent / "gpt2_tokenizer_config.json"
    tmpdir = tmp_path_factory.mktemp("gpt2_marin_tok")
    shutil.copy(config_src, tmpdir / "tokenizer.json")
    shutil.copy(config_src, tmpdir / "tokenizer_config.json")

    tok = HfBaseTokenizer.from_file(str(tmpdir / "tokenizer.json"))

    # GPT-2 uses <|endoftext|> as both BOS and EOS
    vocab = tok.get_vocab()
    eos_token = "<|endoftext|>"
    eos_id = vocab.get(eos_token)

    all_special_ids = [eos_id] if eos_id is not None else []

    return HfMarinTokenizer(
        _tokenizer=tok,
        _name_or_path=str(tmpdir),
        _bos_id=None,
        _eos_id=eos_id,
        _pad_id=None,
        _bos_token=None,
        _eos_token=eos_token,
        _chat_template=None,
        _vocab_size=tok.get_vocab_size(),
        _all_special_ids=all_special_ids,
        _id_to_token={v: k for k, v in vocab.items()},
        _vocab=vocab,
    )


@pytest.fixture(autouse=True)
def _configure_marin_prefix():
    """Set MARIN_PREFIX to a temp directory for tests that rely on it."""
    if "MARIN_PREFIX" in os.environ:
        yield
        return

    with tempfile.TemporaryDirectory(prefix="marin_prefix") as temp_dir:
        os.environ["MARIN_PREFIX"] = temp_dir
        yield
        del os.environ["MARIN_PREFIX"]
