# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import sys
from pathlib import Path

PROJECT = Path(__file__).parents[2] / ".agents" / "projects" / "luxical-arctic-poc"
sys.path.insert(0, str(PROJECT))

from fast_student import fast_document_view, fast_student_config  # noqa: E402
from train_fast_student import parse_args  # noqa: E402


def test_large_config_stays_within_fast_transformer_compute_limit() -> None:
    config = fast_student_config("large", vocab_size=29_145)

    assert config.flops_per_token() < 1_000_000


def test_context_config_doubles_input_width_without_changing_model_width() -> None:
    baseline = fast_student_config("full", vocab_size=29_145)
    context = fast_student_config("context512", vocab_size=29_145)

    assert baseline.max_tokens == 256
    assert context.max_tokens == 512
    assert context.embed_dim == baseline.embed_dim
    assert context.hidden_dim == baseline.hidden_dim
    assert context.num_layers == baseline.num_layers


def test_context_view_keeps_512_characters_from_each_document_region() -> None:
    stored_view = "a" * 2_000 + "\n" + "b" * 2_000 + "\n" + "c" * 2_000

    view = fast_document_view(stored_view, characters_per_source_window=512)

    assert view == "a" * 512 + "\n" + "b" * 512 + "\n" + "c" * 512


def test_document_view_accepts_a_raw_document_with_the_stored_view_length() -> None:
    raw_document = "a" * 6_002

    view = fast_document_view(raw_document, characters_per_source_window=256)

    assert view == "a" * 256 + "\n" + "a" * 256 + "\n" + "a" * 256


def test_train_cli_accepts_large_student_config(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_fast_student.py",
            "--rung",
            "3m",
            "--config",
            "large",
            "--teacher",
            "arctic-medium-256",
            "--training-layout",
            "staged",
        ],
    )

    arguments = parse_args()

    assert arguments.config == "large"
