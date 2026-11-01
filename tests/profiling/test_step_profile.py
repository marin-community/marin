# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import shutil
from pathlib import Path

import pytest
from marin.profiling.step_profile import profile_dir_summary, render, render_diff, wandb_metrics

FIXTURE = Path(__file__).parent / "fixtures" / "step_profile_trace.json"


@pytest.fixture
def profile(tmp_path: Path):
    shutil.copy(FIXTURE, tmp_path / "host.trace.json")
    return profile_dir_summary(tmp_path)


def test_summary_splits_compute_communication_and_exposure(profile) -> None:
    assert (profile.steps, profile.devices) == (2, 1)
    assert profile.step_time == pytest.approx(1e-3)
    assert profile.compute == pytest.approx(600e-6)
    assert profile.communication == pytest.approx(350e-6)
    # All-to-all runs 300-500 us against compute until 600 us: fully hidden. All-gather (700-850) is all exposed.
    assert profile.collectives["all-to-all"].exposed_seconds_per_step == pytest.approx(0.0)
    assert profile.collectives["all-gather"].exposed_seconds_per_step == pytest.approx(150e-6)
    # The device-to-host copy runs 560-640 us; compute ends at 600 us, so half of it is exposed.
    assert profile.offload["d2h"].seconds_per_step == pytest.approx(80e-6)
    assert profile.offload["d2h"].exposed_seconds_per_step == pytest.approx(40e-6)


def test_top_kernels_rank_by_time_with_their_share_of_the_step(profile) -> None:
    names = [k.name for k in profile.top_kernels]
    assert names[0] == "gemm_bf16_tn" and names[-1] == "MemcpyD2H"
    assert profile.top_kernels[0].share_of_step == pytest.approx(0.4)


def test_metrics_and_reports_carry_the_summary(profile) -> None:
    metrics = wandb_metrics(profile)
    assert metrics["profile/all-gather_exposed_s"] == pytest.approx(150e-6)
    assert metrics["profile/exposed_communication_s"] == pytest.approx(150e-6)
    assert "all-gather" in render(profile)
    assert "+0.00 ms" in render_diff(profile, profile)


def test_step_time_falls_back_to_the_kernel_span_over_the_sessions_steps(tmp_path: Path) -> None:
    """GPU traces have no Steps line; the session directory names the profiled steps."""
    text = FIXTURE.read_text().replace('"Steps"', '"Not steps"')
    session = tmp_path / "steps-24-to-26-process0"
    session.mkdir()
    (session / "host.trace.json").write_text(text)
    profile = profile_dir_summary(session)
    assert profile.steps == 2
    assert profile.step_time == pytest.approx(925e-6)  # kernels span 0..1850 us over two steps
