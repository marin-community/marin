# Copyright 2026 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse

import pytest

from scripts.gdn import gdnctl


def test_parse_profile_env_parses_key_value_pairs() -> None:
    parsed = gdnctl._parse_profile_env(
        [
            "GDN_TRIANGULAR_SOLVE_PROBE=identity",
            "WANDB_MODE=offline",
            "EMPTY_VALUE=",
        ]
    )
    assert parsed == [
        ("GDN_TRIANGULAR_SOLVE_PROBE", "identity"),
        ("WANDB_MODE", "offline"),
        ("EMPTY_VALUE", ""),
    ]


@pytest.mark.parametrize(
    "item",
    [
        "MISSING_EQUALS",
        "1BAD=value",
        "=value",
    ],
)
def test_parse_profile_env_rejects_invalid_entries(item: str) -> None:
    with pytest.raises(SystemExit):
        gdnctl._parse_profile_env([item])


def test_validation_mode_profile_only_skips_tests(monkeypatch: pytest.MonkeyPatch) -> None:
    called = {"tests": False, "profile": False}

    def _fake_tests(_: argparse.Namespace) -> tuple[int, bool]:
        called["tests"] = True
        return 0, True

    def _fake_profile(_: argparse.Namespace, *, iteration: int) -> tuple[int, bool, dict[str, object]]:
        called["profile"] = True
        return 0, True, {"profile_prefix": f"iter-{iteration:03d}", "metrics": {"throughput/mfu": 1.23}}

    monkeypatch.setattr(gdnctl, "_run_validation_tests_once", _fake_tests)
    monkeypatch.setattr(gdnctl, "_run_validation_profile_once", _fake_profile)

    args = argparse.Namespace(
        validation_mode="profile-only",
        validation_max_attempts=1,
        validation_retry_sleep=0.0,
    )
    ok, rc, info = gdnctl._run_validation_gate_for_iteration(args, iteration=7)
    assert ok
    assert rc == 0
    assert info["profile_prefix"] == "iter-007"
    assert called["profile"]
    assert not called["tests"]
