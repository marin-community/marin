# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pandas as pd

from experiments.domain_phase_mix.exploratory.paper_plots import (
    starcoder_two_phase_local_snr_landscape as local_snr,
    starcoder_two_phase_heteroskedastic_landscape as landscape,
)


def _repeat_rows() -> pd.DataFrame:
    rows = []
    values = {
        "proportional": [1.0, 1.2, 0.8],
        "late_code": [0.5, 0.5, 0.5],
        "noisy_code": [1.8, 2.0, 2.2],
    }
    coords = {
        "proportional": (0, 0.03649, 0.03649),
        "late_code": (1, 0.0, 0.25),
        "noisy_code": (2, 1.0, 1.0),
    }
    for anchor_id, metric_values in values.items():
        anchor_index, p0, p1 = coords[anchor_id]
        for repeat_index, metric_value in enumerate(metric_values):
            rows.append(
                {
                    "anchor_index": anchor_index,
                    "anchor_id": anchor_id,
                    "phase_0_starcoder": p0,
                    "phase_1_starcoder": p1,
                    "repeat_index": repeat_index,
                    "latest_step": 3813,
                    landscape.TARGET: metric_value,
                    "eval/bpb": metric_value + 0.1,
                }
            )
    rows.append(
        {
            "anchor_index": 2,
            "anchor_id": "noisy_code",
            "phase_0_starcoder": 1.0,
            "phase_1_starcoder": 1.0,
            "repeat_index": 3,
            "latest_step": 3000,
            landscape.TARGET: 8.0,
            "eval/bpb": 8.1,
        }
    )
    return pd.DataFrame(rows)


def test_load_repeat_panel_excludes_partial_rows(tmp_path):
    path = tmp_path / "repeats.csv"
    _repeat_rows().to_csv(path, index=False)

    panel = landscape.load_repeat_panel(path)

    assert panel.final_step == 3813
    assert len(panel.final_rows) == 9
    assert len(panel.excluded_rows) == 1
    assert panel.excluded_rows.iloc[0]["latest_step"] == 3000


def test_summarize_anchor_noise_adds_contrast_snr_and_local_denominator():
    summary = landscape.summarize_anchor_noise(_repeat_rows().query("latest_step == 3813"), [landscape.TARGET])
    target = summary[summary["metric"].eq(landscape.TARGET)].set_index("anchor_id")

    assert target.loc["proportional", "count"] == 3
    assert np.isclose(target.loc["proportional", "std"], 0.2)
    assert np.isclose(target.loc["late_code", "std"], 0.0)
    assert target.loc["late_code", "contrast_snr_vs_proportional"] > 0.0
    assert np.isinf(target.loc["late_code", "between_anchor_std_over_local_std"])
    assert target.loc["noisy_code", "log10_variance"] > target.loc["late_code", "log10_variance"]


def test_render_heteroskedastic_landscape_has_metric_dropdown():
    rows = pd.DataFrame(
        {
            "phase_0_starcoder": [0.0, 0.0, 1.0, 1.0],
            "phase_1_starcoder": [0.0, 1.0, 0.0, 1.0],
            landscape.TARGET: [1.0, 0.8, 1.2, 2.0],
            "status": ["completed"] * 4,
        }
    )
    summary = landscape.summarize_anchor_noise(_repeat_rows().query("latest_step == 3813"), [landscape.TARGET, "eval/bpb"])

    fig = landscape.render_heteroskedastic_landscape(
        rows,
        summary,
        final_row_count=9,
        excluded_row_count=1,
        final_step=3813,
    )

    assert len(fig.data) == 12
    assert len(fig.layout.updatemenus[0].buttons) == 2
    ci_trace_names = [
        trace.name
        for trace in fig.data
        if getattr(trace, "mode", None) == "lines" and "95% CI" in str(trace.name)
    ]
    assert ci_trace_names


def test_fit_local_gradient_recovers_linear_surface():
    rows = pd.DataFrame(
        {
            "phase_0_starcoder": [0.0, 0.0, 1.0, 1.0, 0.5],
            "phase_1_starcoder": [0.0, 1.0, 0.0, 1.0, 0.5],
            landscape.TARGET: [1.0, 4.0, 3.0, 6.0, 3.5],
            "status": ["completed"] * 5,
        }
    )

    gradient = local_snr.fit_local_gradient(rows, landscape.TARGET, 0.5, 0.5, bandwidth=10.0)

    assert np.isclose(gradient.grad_phase_0, 2.0)
    assert np.isclose(gradient.grad_phase_1, 3.0)


def test_summarize_local_snr_uses_repeat_variance_denominator():
    rows = pd.DataFrame(
        {
            "phase_0_starcoder": [0.0, 0.0, 1.0, 1.0, 0.5],
            "phase_1_starcoder": [0.0, 1.0, 0.0, 1.0, 0.5],
            landscape.TARGET: [1.0, 4.0, 3.0, 6.0, 3.5],
            "status": ["completed"] * 5,
        }
    )
    anchor_summary = landscape.summarize_anchor_noise(
        _repeat_rows().query("latest_step == 3813"),
        [landscape.TARGET],
    )

    summary = local_snr.summarize_local_snr(rows, anchor_summary, [landscape.TARGET], radius=0.1, bandwidth=10.0)
    proportional = summary[summary["anchor_id"].eq("proportional")].iloc[0]

    expected_signal = 0.1 * np.hypot(2.0, 3.0)
    expected_snr = expected_signal**2 / (0.2**2)
    assert np.isclose(proportional["local_signal_at_radius"], expected_signal)
    assert np.isclose(proportional["snr_power"], expected_snr)


def test_render_local_snr_landscape_has_metric_dropdown():
    rows = pd.DataFrame(
        {
            "phase_0_starcoder": [0.0, 0.0, 1.0, 1.0, 0.5],
            "phase_1_starcoder": [0.0, 1.0, 0.0, 1.0, 0.5],
            landscape.TARGET: [1.0, 4.0, 3.0, 6.0, 3.5],
            "eval/bpb": [2.0, 5.0, 4.0, 7.0, 4.5],
            "status": ["completed"] * 5,
        }
    )
    anchor_summary = landscape.summarize_anchor_noise(
        _repeat_rows().query("latest_step == 3813"),
        [landscape.TARGET, "eval/bpb"],
    )
    summary = local_snr.summarize_local_snr(
        rows,
        anchor_summary,
        [landscape.TARGET, "eval/bpb"],
        bandwidth=10.0,
    )

    fig = local_snr.render_local_snr_landscape(
        rows,
        summary,
        final_row_count=9,
        excluded_row_count=1,
        final_step=3813,
    )

    assert len(fig.data) == 10
    assert len(fig.layout.updatemenus[0].buttons) == 2
