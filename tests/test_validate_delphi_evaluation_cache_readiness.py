# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest

from experiments.domain_phase_mix import validate_delphi_evaluation_cache_readiness as readiness


def test_evaluation_cache_dirs_are_region_local(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MARIN_PREFIX", "gs://marin-us-east5")

    cache_dirs = readiness.evaluation_cache_dirs(region="us-east5")

    assert len(cache_dirs) == readiness.EXPECTED_EVALUATION_CACHES
    assert all(path.startswith("gs://marin-us-east5/") for path in cache_dirs)


def test_validate_evaluation_caches_accepts_complete_panel(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MARIN_PREFIX", "gs://marin-us-east5")
    monkeypatch.setattr(readiness, "_read_text", lambda _: "SUCCESS")
    monkeypatch.setattr(readiness, "_path_exists", lambda _: True)

    cache_dirs = readiness.validate_evaluation_caches(region="us-east5")

    assert len(cache_dirs) == readiness.EXPECTED_EVALUATION_CACHES


def test_validate_evaluation_caches_rejects_missing_status(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MARIN_PREFIX", "gs://marin-us-east5")
    cache_dirs = readiness.evaluation_cache_dirs(region="us-east5")

    def read_text(path: str) -> str:
        if path == f"{cache_dirs[0]}/.executor_status":
            raise FileNotFoundError(path)
        return "SUCCESS"

    monkeypatch.setattr(readiness, "_read_text", read_text)
    monkeypatch.setattr(readiness, "_path_exists", lambda _: True)

    with pytest.raises(ValueError, match="missing executor status"):
        readiness.validate_evaluation_caches(region="us-east5")


def test_validate_evaluation_caches_rejects_missing_stats(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MARIN_PREFIX", "gs://marin-us-east5")
    cache_dirs = readiness.evaluation_cache_dirs(region="us-east5")
    monkeypatch.setattr(readiness, "_read_text", lambda _: "SUCCESS")
    monkeypatch.setattr(readiness, "_path_exists", lambda path: not path.startswith(cache_dirs[0]))

    with pytest.raises(ValueError, match="missing validation statistics"):
        readiness.validate_evaluation_caches(region="us-east5")
