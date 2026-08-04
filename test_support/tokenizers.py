# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
import shutil
from pathlib import Path

import pytest


def stage_gpt2_tokenizer(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Stage the checked-in GPT-2 tokenizer without accessing Hugging Face."""
    tokenizer_source = Path(__file__).resolve().parents[1] / "lib" / "levanter" / "tests"
    tokenizer_dir = tmp_path_factory.mktemp("gpt2_tokenizer")
    shutil.copy(tokenizer_source / "gpt2_tokenizer.json", tokenizer_dir / "tokenizer.json")
    shutil.copy(tokenizer_source / "gpt2_tokenizer_config.json", tokenizer_dir / "tokenizer_config.json")
    (tokenizer_dir / "config.json").write_text(json.dumps({"model_type": "gpt2", "vocab_size": 5027}))
    return tokenizer_dir
