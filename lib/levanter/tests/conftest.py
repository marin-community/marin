# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import json
import os
import shutil
import tempfile
from pathlib import Path

import pytest
from transformers import AutoTokenizer

pytest_plugins = ["tests.test_utils"]


@pytest.fixture(scope="session")
def local_gpt2_tokenizer(tmp_path_factory):
    """Load a GPT2 tokenizer from a local JSON file to avoid network downloads."""

    config_src = Path(__file__).parent / "gpt2_tokenizer_config.json"
    tmpdir = tmp_path_factory.mktemp("gpt2_tok")
    shutil.copy(config_src, tmpdir / "tokenizer.json")
    shutil.copy(config_src, tmpdir / "tokenizer_config.json")
    (tmpdir / "config.json").write_text(json.dumps({"model_type": "gpt2", "vocab_size": 5027}))
    return AutoTokenizer.from_pretrained(str(tmpdir))


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
