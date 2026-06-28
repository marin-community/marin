# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import subprocess
import sys

import pytest

from experiments.domain_phase_mix.east5_launch_safety import (
    validate_east5_iris_command,
    validate_regional_iris_command,
)


def test_accepts_explicit_east5_parent_and_east5_paths() -> None:
    command = (
        "uv run iris --cluster=marin job run --no-wait "
        "--job-name dm-safe --region us-east5 --zone us-east5-a "
        "--enable-extra-resources --cpu 1 --memory 16GB --disk 20GB -- "
        "python -m experiments.domain_phase_mix.launch_300m_raw_ppl_evals "
        "--state-csv gs://marin-us-east5/pinlin/state.csv "
        "--executor-prefix pinlin_calvin_xu/data_mixture/safe "
        "--tpu-region us-east5 --tpu-zone us-east5-a"
    )

    result = validate_east5_iris_command(command)

    assert result.ok
    assert result.errors == []


def test_accepts_equals_form_parent_region_and_zone() -> None:
    command = (
        "uv run iris --cluster=marin job run --no-wait "
        "--job-name dm-safe --region=us-east5 --zone=us-east5-a -- "
        "python -m experiments.domain_phase_mix.launch_300m_raw_ppl_evals"
    )

    result = validate_east5_iris_command(command)

    assert result.ok
    assert result.parent_regions == ["us-east5"]
    assert result.parent_zone == "us-east5-a"


def test_rejects_explicit_parent_region_mismatch() -> None:
    command = (
        "uv run iris --cluster=marin job run --no-wait "
        "--job-name dm-unsafe --region us-central1 --zone us-east5-a -- "
        "python -m experiments.domain_phase_mix.launch_300m_raw_ppl_evals"
    )

    result = validate_east5_iris_command(command)

    assert not result.ok
    assert any("--region" in error and "us-central1" in error for error in result.errors)


def test_rejects_missing_parent_zone_even_when_children_are_east5a() -> None:
    command = (
        "uv run iris --cluster=marin job run --no-wait "
        "--job-name dm-unsafe --region us-east5 "
        "--enable-extra-resources --cpu 1 --memory 16GB --disk 20GB -- "
        "python -m experiments.domain_phase_mix.launch_proportional_controllability_300m "
        "--tpu-region us-east5 --tpu-zone us-east5-a"
    )

    result = validate_east5_iris_command(command)

    assert not result.ok
    assert any("--zone us-east5-a" in error for error in result.errors)


def test_accepts_region_only_parent_when_explicitly_allowed() -> None:
    command = (
        "uv run iris --cluster=marin job run --no-wait "
        "--job-name dm-safe --region us-east5 "
        "--enable-extra-resources --cpu 0.5 --memory 8GB --disk 20GB -- "
        "python -m experiments.domain_phase_mix.launch_starcoder_heteroskedastic_snr "
        "--eval-datasets-cache-path gs://marin-us-east5/raw/eval-datasets/code-tasks "
        "--tpu-region us-east5 --tpu-zone us-east5-a"
    )

    result = validate_regional_iris_command(
        command,
        expected_region="us-east5",
        expected_zone="us-east5-a",
        expected_bucket_prefix="gs://marin-us-east5",
        allow_region_only_parent=True,
    )

    assert result.ok
    assert result.parent_zone is None
    assert result.allows_region_only_parent is True
    assert "omits --zone" in result.warnings[0]


def test_rejects_missing_parent_region_and_zone() -> None:
    command = (
        "uv run iris --cluster=marin job run --no-wait "
        "--job-name dm-unsafe --no-preemptible -- "
        "python -m experiments.domain_phase_mix.launch_300m_raw_ppl_evals "
        "--state-csv gs://marin-us-east5/pinlin/state.csv "
        "--tpu-region us-east5 --tpu-zone us-east5-a"
    )

    result = validate_east5_iris_command(command)

    assert not result.ok
    assert any("--region us-east5" in error for error in result.errors)
    assert any("--zone us-east5-a" in error for error in result.errors)


def test_rejects_central_gcs_paths() -> None:
    command = (
        "uv run iris --cluster=marin job run --no-wait "
        "--job-name dm-unsafe --region us-east5 --zone us-east5-a -- "
        "python -m experiments.domain_phase_mix.launch_300m_raw_ppl_evals "
        "--state-csv gs://marin-us-central1/pinlin/state.csv"
    )

    result = validate_east5_iris_command(command)

    assert not result.ok
    assert any("gs://marin-us-central1" in error for error in result.errors)


def test_rejects_central_gcs_paths_in_comma_joined_values() -> None:
    command = (
        "uv run iris --cluster=marin job run --no-wait "
        "--job-name dm-unsafe --region us-east5 --zone us-east5-a -- "
        "python -m experiments.domain_phase_mix.launch_300m_raw_ppl_evals "
        "--inputs gs://marin-us-east5/pinlin/ok.csv,gs://marin-us-central1/pinlin/bad.csv"
    )

    result = validate_east5_iris_command(command)

    assert not result.ok
    assert any("gs://marin-us-central1" in error for error in result.errors)


def test_rejects_child_tpu_region_mismatch() -> None:
    command = (
        "uv run iris --cluster=marin job run --no-wait "
        "--job-name dm-unsafe --region us-east5 --zone us-east5-a -- "
        "python -m experiments.domain_phase_mix.launch_300m_raw_ppl_evals "
        "--tpu-region us-central1 --tpu-zone us-east5-a"
    )

    result = validate_east5_iris_command(command)

    assert not result.ok
    assert any("--tpu-region" in error for error in result.errors)


def test_rejects_child_tpu_zone_mismatch() -> None:
    command = (
        "uv run iris --cluster=marin job run --no-wait "
        "--job-name dm-unsafe --region us-east5 --zone us-east5-a -- "
        "python -m experiments.domain_phase_mix.launch_300m_raw_ppl_evals "
        "--tpu-region us-east5 --tpu-zone us-east5-b"
    )

    result = validate_east5_iris_command(command)

    assert not result.ok
    assert any("--tpu-zone" in error for error in result.errors)


def test_rejects_non_iris_job_run_command() -> None:
    with pytest.raises(ValueError, match="Iris job run"):
        validate_east5_iris_command("uv run python -m experiments.foo")


def test_accepts_central1_parent_children_and_paths_when_expected() -> None:
    command = (
        "uv run iris --cluster=marin job run --no-wait "
        "--job-name dm-starcoder --region us-central1 --zone us-central1-a -- "
        "python -m experiments.domain_phase_mix.launch_starcoder_heteroskedastic_snr "
        "--cache-root gs://marin-us-central1/pinlin/starcoder/cache "
        "--tpu-region us-central1 --tpu-zone us-central1-a"
    )

    result = validate_regional_iris_command(
        command,
        expected_region="us-central1",
        expected_zone="us-central1-a",
        expected_bucket_prefix="gs://marin-us-central1",
    )

    assert result.ok
    assert result.errors == []


def test_rejects_mixed_east5_paths_for_central1_expected_region() -> None:
    command = (
        "uv run iris --cluster=marin job run --no-wait "
        "--job-name dm-starcoder --region us-central1 --zone us-central1-a -- "
        "python -m experiments.domain_phase_mix.launch_starcoder_heteroskedastic_snr "
        "--cache-root gs://marin-us-central1/pinlin/starcoder/cache "
        "--checkpoint-root gs://marin-us-east5/pinlin/stale/checkpoint "
        "--tpu-region us-central1 --tpu-zone us-central1-a"
    )

    result = validate_regional_iris_command(
        command,
        expected_region="us-central1",
        expected_zone="us-central1-a",
        expected_bucket_prefix="gs://marin-us-central1",
    )

    assert not result.ok
    assert any("gs://marin-us-east5" in error for error in result.errors)


def test_cli_accepts_central1_expected_region() -> None:
    command = (
        "uv run iris --cluster=marin job run --no-wait "
        "--job-name dm-starcoder --region us-central1 --zone us-central1-a -- "
        "python -m experiments.domain_phase_mix.launch_starcoder_heteroskedastic_snr "
        "--eval-datasets-cache-path gs://marin-us-central1/raw/eval-datasets/code-tasks "
        "--tpu-region us-central1 --tpu-zone us-central1-a"
    )

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "experiments.domain_phase_mix.east5_launch_safety",
            "--command",
            command,
            "--expected-region",
            "us-central1",
            "--expected-zone",
            "us-central1-a",
            "--expected-bucket-prefix",
            "gs://marin-us-central1",
            "--json",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(completed.stdout)
    assert payload["ok"] is True
    assert payload["parent_regions"] == ["us-central1"]


def test_cli_accepts_region_only_parent_when_explicitly_allowed() -> None:
    command = (
        "uv run iris --cluster=marin job run --no-wait "
        "--job-name dm-starcoder --region us-east5 -- "
        "python -m experiments.domain_phase_mix.launch_starcoder_heteroskedastic_snr "
        "--eval-datasets-cache-path gs://marin-us-east5/raw/eval-datasets/code-tasks "
        "--tpu-region us-east5 --tpu-zone us-east5-a"
    )

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "experiments.domain_phase_mix.east5_launch_safety",
            "--command",
            command,
            "--expected-region",
            "us-east5",
            "--expected-zone",
            "us-east5-a",
            "--expected-bucket-prefix",
            "gs://marin-us-east5",
            "--allow-region-only-parent",
            "--json",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(completed.stdout)
    assert payload["ok"] is True
    assert payload["allows_region_only_parent"] is True
    assert payload["parent_zone"] is None
    assert payload["warnings"]
