# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import sys
from pathlib import Path

PROJECT = Path(__file__).parents[2] / ".agents" / "projects" / "luxical-arctic-poc"
sys.path.insert(0, str(PROJECT))

from fast_student import fast_student_config  # noqa: E402
from train_fast_student import parse_args  # noqa: E402


def test_large_config_stays_within_fast_transformer_compute_limit() -> None:
    config = fast_student_config("large", vocab_size=29_145)

    assert config.flops_per_token() < 1_000_000


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
