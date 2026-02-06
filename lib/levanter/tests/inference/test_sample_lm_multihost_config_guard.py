# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import pytest

from levanter.main.sample_lm_multihost import SampleLmMultihostConfig, _validate_tracker_logging_safety


def _config(**overrides) -> SampleLmMultihostConfig:
    config = SampleLmMultihostConfig()
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


def test_guard_rejects_multiround_legacy_metrics_samples():
    config = _config(
        n_rounds=2,
        defer_tracker_logs_until_end=False,
        skip_metrics_log=False,
        skip_samples_table=False,
        emit_minimal_tracker_probe=False,
        skip_leader_postprocess=False,
    )

    with pytest.raises(ValueError, match="Unsafe tracker logging configuration"):
        _validate_tracker_logging_safety(config, is_multihost=True)


def test_guard_rejects_multiround_probe_only():
    config = _config(
        n_rounds=2,
        defer_tracker_logs_until_end=False,
        skip_metrics_log=True,
        skip_samples_table=True,
        emit_minimal_tracker_probe=True,
        skip_leader_postprocess=False,
    )

    with pytest.raises(ValueError, match="Unsafe tracker logging configuration"):
        _validate_tracker_logging_safety(config, is_multihost=True)


def test_guard_allows_multiround_allhost_probe_only():
    config = _config(
        n_rounds=2,
        defer_tracker_logs_until_end=False,
        skip_metrics_log=True,
        skip_samples_table=True,
        emit_minimal_tracker_probe=False,
        emit_minimal_tracker_probe_all_hosts=True,
        skip_leader_postprocess=False,
    )

    _validate_tracker_logging_safety(config, is_multihost=True)


@pytest.mark.parametrize(
    "overrides,is_multihost",
    [
        (
            {
                "n_rounds": 2,
                "defer_tracker_logs_until_end": True,
                "skip_metrics_log": False,
                "skip_samples_table": False,
                "emit_minimal_tracker_probe": False,
                "skip_leader_postprocess": False,
            },
            True,
        ),
        (
            {
                "n_rounds": 2,
                "defer_tracker_logs_until_end": False,
                "skip_metrics_log": True,
                "skip_samples_table": True,
                "emit_minimal_tracker_probe": False,
                "skip_leader_postprocess": False,
            },
            True,
        ),
        (
            {
                "n_rounds": 1,
                "defer_tracker_logs_until_end": False,
                "skip_metrics_log": False,
                "skip_samples_table": False,
                "emit_minimal_tracker_probe": False,
                "skip_leader_postprocess": False,
            },
            True,
        ),
        (
            {
                "n_rounds": 2,
                "defer_tracker_logs_until_end": False,
                "skip_metrics_log": False,
                "skip_samples_table": False,
                "emit_minimal_tracker_probe": False,
                "skip_leader_postprocess": False,
            },
            False,
        ),
        (
            {
                "n_rounds": 2,
                "defer_tracker_logs_until_end": False,
                "skip_metrics_log": False,
                "skip_samples_table": False,
                "emit_minimal_tracker_probe": False,
                "skip_leader_postprocess": True,
            },
            True,
        ),
        (
            {
                "n_rounds": 2,
                "defer_tracker_logs_until_end": False,
                "skip_metrics_log": True,
                "skip_samples_table": True,
                "emit_minimal_tracker_probe": False,
                "emit_minimal_tracker_probe_all_hosts": True,
                "skip_leader_postprocess": True,
            },
            True,
        ),
    ],
)
def test_guard_allows_known_safe_combinations(overrides, is_multihost):
    config = _config(**overrides)
    _validate_tracker_logging_safety(config, is_multihost=is_multihost)
