# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
import shutil
from pathlib import Path

import pytest
from levanter.tokenizers import load_tokenizer

_GPT2_TOKENIZER_CONFIG = (
    Path(__file__).resolve().parents[2] / "lib" / "levanter" / "tests" / "gpt2_tokenizer_config.json"
)


@pytest.fixture(scope="session")
def gpt2_tokenizer(tmp_path_factory):
    """Local GPT-2 tokenizer fixture without network access."""
    tmpdir = tmp_path_factory.mktemp("tts_gpt2_tok")
    shutil.copy(_GPT2_TOKENIZER_CONFIG, tmpdir / "tokenizer.json")
    shutil.copy(_GPT2_TOKENIZER_CONFIG, tmpdir / "tokenizer_config.json")
    (tmpdir / "config.json").write_text(json.dumps({"model_type": "gpt2", "vocab_size": 5027}))
    return load_tokenizer(str(tmpdir))
