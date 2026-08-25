# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
import os
import tempfile
from pathlib import Path

import pytest
from fray.current_client import set_current_client
from fray.local_backend import LocalClient
from levanter.testing.tokenizer import stage_gpt2_tokenizer
from levanter.tokenizers import load_tokenizer

DEFAULT_BUCKET_NAME = "marin-us-east5"
DEFAULT_DOCUMENT_PATH = "documents/test-document-path"


@pytest.fixture(scope="session")
def gpt2_tokenizer_path(tmp_path_factory) -> str:
    source_dir = Path(__file__).resolve().parents[1] / "lib" / "levanter" / "tests"
    output_dir = tmp_path_factory.mktemp("gpt2_tokenizer")
    return str(stage_gpt2_tokenizer(source_dir, output_dir))


@pytest.fixture(scope="session")
def gpt2_tokenizer(gpt2_tokenizer_path):
    """Load the checked-in GPT-2 tokenizer without accessing Hugging Face."""
    return load_tokenizer(gpt2_tokenizer_path)


@pytest.fixture(autouse=True)
def fray_client(_configure_marin_prefix):
    """Set up a v2 LocalClient for all tests.

    Depends on ``_configure_marin_prefix`` so it tears down first: shutting
    down the client joins every local-backend worker thread before that
    fixture removes the MARIN_PREFIX temp directory. Without this ordering, a
    straggler task can still be writing under MARIN_PREFIX when the temp
    directory is removed, raising ``OSError: Directory not empty`` at
    teardown.
    """
    client = LocalClient()
    with set_current_client(client):
        yield client
    client.shutdown()


@pytest.fixture(autouse=True)
def disable_wandb(monkeypatch):
    """Disable WANDB logging during tests."""
    monkeypatch.setenv("WANDB_MODE", "disabled")


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
