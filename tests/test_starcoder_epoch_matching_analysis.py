# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import csv
import json
from dataclasses import asdict, replace

import numpy as np
import pytest

from experiments.domain_phase_mix import starcoder_epoch_matching as design_module
from experiments.domain_phase_mix.exploratory.two_phase_many import analyze_starcoder_epoch_matching as analysis


@pytest.fixture
def design(tmp_path):
    payload = design_module.load_design().to_dict()
    target_values = {0.0: 4.0, 0.1: 1.0, 0.3: 3.0, 0.5: 1.0, 0.7: 2.0, 1.0: -100.0}
    for observation in payload["target_observations"]:
        observation["observed_bpb"] = target_values.get(observation["starcoder_weight"], 2.5)
    payload.pop("design_sha256")
    payload["design_sha256"] = design_module.canonical_sha256(payload)
    path = tmp_path / "synthetic_design.json"
    path.write_text(json.dumps(payload))
    return design_module.load_design(path)


def measurements(design, stage):
    records = []
    for run in design_module.select_runs(design, stage):
        if run.arm == "target":
            value = 5.0
        elif run.arm == "unmatched":
            value = 100.0 + (run.starcoder_weight - 0.7) ** 2
        else:
            value = 10.0 + (run.starcoder_weight - 0.1) ** 2
        records.append(
            analysis.Measurement(
                run.run_name,
                run.total_steps - 1,
                design_module.METRIC,
                value,
                design.design_sha256,
                f"verified-{run.run_name}",
                "succeeded",
            )
        )
    return tuple(records)


def launch_plan(design, stage) -> dict:
    runs = design_module.select_runs(design, stage)
    return {
        "design_sha256": design.design_sha256,
        "metric": design.primary_metric,
        "stage": stage,
        "new_training_runs": len(runs),
        "runs": [{**asdict(run), "fingerprint": f"verified-{run.run_name}"} for run in runs],
    }


def test_analysis_uses_target_loss_at_proxy_choices_and_excludes_historical_endpoint(design, tmp_path):
    input_path = tmp_path / "measurements.csv"
    with input_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=analysis.MEASUREMENT_COLUMNS)
        writer.writeheader()
        writer.writerows(asdict(record) for record in measurements(design, "pilot"))
    plan_path = tmp_path / "launch_plan.json"
    plan_path.write_text(json.dumps(launch_plan(design, "pilot")))
    report, points = analysis.analyze(
        design, analysis.read_measurements(input_path), "pilot", analysis.read_plan(plan_path)
    )
    assert report["selection"]["unmatched"]["selected_weight"] == 0.7
    assert report["selection"]["unmatched"]["absolute_weight_displacement_from_target_minimum"] == pytest.approx(0.6)
    assert report["selection"]["unmatched"]["target_grid_regret"] == 1.0
    assert report["selection"]["matched"]["selected_weight"] == 0.1
    assert report["selection"]["matched"]["absolute_weight_displacement_from_target_minimum"] == 0.0
    assert report["selection"]["matched"]["target_grid_regret"] == 0.0
    assert report["target_minimum"]["bpb"] == 1.0
    assert report["delta_target_regret"] == -1.0
    endpoint = next(point for point in points if point.arm == "target" and point.starcoder_weight == 1.0)
    assert endpoint.observed_bpb == 5.0
    assert endpoint.source_kind == "new"
    analysis.write_outputs(report, points, tmp_path / "results")
    assert json.loads((tmp_path / "results/analysis.json").read_text())["grid"] == [0.0, 0.1, 0.3, 0.7, 1.0]
    with (tmp_path / "results/curves.csv").open(newline="") as handle:
        saved = list(csv.DictReader(handle))
    assert len(saved) == 15
    assert sum(row["is_alias"] == "True" for row in saved) == 1


def test_analysis_refuses_missing_corrected_endpoint_even_with_archived_value(design):
    target_name = next(run.run_name for run in design.runs if run.arm == "target")
    records = tuple(record for record in measurements(design, "pilot") if record.run_name != target_name)
    with pytest.raises(ValueError, match="missing final measurements"):
        analysis.analyze(design, records, "pilot", launch_plan(design, "pilot"))


@pytest.mark.parametrize(("matched_choice", "delta_regret"), [(0.1, -1.0), (0.7, 0.0), (0.3, 1.0)])
def test_analysis_reports_matching_improvement_tie_and_worsening(design, matched_choice, delta_regret):
    runs = {run.run_name: run for run in design.runs}
    records = tuple(
        (
            replace(record, value=10.0 + (runs[record.run_name].starcoder_weight - matched_choice) ** 2)
            if runs[record.run_name].arm == "matched"
            else record
        )
        for record in measurements(design, "pilot")
    )
    report, _ = analysis.analyze(design, records, "pilot", launch_plan(design, "pilot"))
    assert report["delta_target_regret"] == delta_regret


def test_excess_plot_subtracts_each_mean_curve_minimum_on_one_axis_without_changing_raw_values(design):
    report, points = analysis.analyze(design, measurements(design, "pilot"), "pilot", launch_plan(design, "pilot"))
    original_report = json.dumps(report, sort_keys=True)
    raw = analysis.plot_curves(report, points, ())
    excess = analysis.plot_curves(report, points, (), mode="excess")
    assert len(excess.axes) == 1
    raw_lines = raw.axes[0].lines
    excess_lines = excess.axes[0].lines
    assert len(raw_lines) == len(excess_lines) == 3
    # Independently specified target, unmatched, and matched minima have very different BPB levels.
    for raw_line, excess_line, minimum in zip(raw_lines, excess_lines, (1.0, 100.0, 10.0), strict=True):
        raw_values = np.asarray(raw_line.get_ydata(), dtype=float)
        excess_values = np.asarray(excess_line.get_ydata(), dtype=float)
        assert min(raw_values) == minimum
        assert min(excess_values) == 0.0
        assert excess_values == pytest.approx(raw_values - minimum)
        assert excess_line.get_xdata() == pytest.approx(raw_line.get_xdata())
    assert json.dumps(report, sort_keys=True) == original_report


def test_analysis_refuses_primary_selection_from_only_pilot_measurements(design):
    with pytest.raises(ValueError, match="missing final measurements"):
        analysis.analyze(design, measurements(design, "pilot"), "primary", launch_plan(design, "primary"))


@pytest.mark.parametrize(
    "change",
    [
        {"design_sha256": "stale"},
        {"metric": "eval/another_task/bpb"},
        {"step": 0},
        {"status": "running"},
        {"value": float("nan")},
        {"config_fingerprint": ""},
        {"config_fingerprint": "nonempty-but-wrong-config"},
    ],
)
def test_analysis_rejects_unverified_or_nonfinal_measurements(design, change):
    records = measurements(design, "pilot")
    corrupted = (replace(records[0], **change), *records[1:])
    with pytest.raises(ValueError):
        analysis.analyze(design, corrupted, "pilot", launch_plan(design, "pilot"))


def test_analysis_rejects_duplicate_run_instead_of_counting_it_as_replication(design):
    records = measurements(design, "pilot")
    with pytest.raises(ValueError, match="Duplicate measurement"):
        analysis.analyze(design, (*records, records[0]), "pilot", launch_plan(design, "pilot"))


def test_replicated_analysis_aliases_zero_weight_once_per_seed(design):
    report, points = analysis.analyze(
        design, measurements(design, "replicated"), "replicated", launch_plan(design, "replicated")
    )
    zeros = [point for point in points if point.arm != "target" and point.starcoder_weight == 0.0]
    assert len(zeros) == 6
    assert len({point.source_run_name for point in zeros}) == 3
    assert sum(point.is_alias for point in zeros) == 3
    assert report["new_measurements_used"] == 154
    assert report["proxy_measurements_used"] == 153
    assert report["proxy_curve_points"] == 156
    assert all(row["n_measurements"] == 3 for row in report["curves"]["matched"])
    assert all(row["n_measurements"] == 1 for row in report["curves"]["target"])
    assert report["interpretation"]["statistical_superiority_test"] is False


def test_analysis_reports_all_exact_ties_and_selects_lowest_weight(design):
    runs = {run.run_name: run for run in design.runs}
    records = tuple(
        (
            replace(record, value=1.0 if runs[record.run_name].starcoder_weight in (0.3, 0.7) else 3.0)
            if runs[record.run_name].arm == "unmatched"
            else record
        )
        for record in measurements(design, "pilot")
    )
    report, _ = analysis.analyze(design, records, "pilot", launch_plan(design, "pilot"))
    assert report["selection"]["unmatched"]["selected_weight"] == 0.3
    assert report["selection"]["unmatched"]["tied_weights"] == [0.3, 0.7]
    assert report["selection"]["unmatched"]["target_grid_regret"] == 2.0


def test_replicated_selection_uses_mean_over_all_prespecified_trainer_seeds(design):
    runs = {run.run_name: run for run in design.runs}
    values = {
        (design_module.REFERENCE_SEED, 0.3): 4.0,
        (design_module.REFERENCE_SEED, 0.7): 0.0,
        (design_module.TRAINER_SEEDS[1], 0.3): 1.0,
        (design_module.TRAINER_SEEDS[1], 0.7): 9.0,
        (design_module.TRAINER_SEEDS[2], 0.3): 1.0,
        (design_module.TRAINER_SEEDS[2], 0.7): 9.0,
    }
    records = tuple(
        (
            replace(
                record,
                value=values.get((runs[record.run_name].trainer_seed, runs[record.run_name].starcoder_weight), 20.0),
            )
            if runs[record.run_name].arm == "unmatched"
            else record
        )
        for record in measurements(design, "replicated")
    )
    report, _ = analysis.analyze(design, records, "replicated", launch_plan(design, "replicated"))
    assert report["selection"]["unmatched"]["selected_weight"] == 0.3
    assert report["selection"]["unmatched"]["proxy_mean_bpb"] == 2.0
    assert report["selection"]["unmatched"]["target_grid_regret"] == 2.0


@pytest.mark.parametrize("change", ["design", "stage", "metric", "count", "missing", "duplicate", "seed", "blank"])
def test_analysis_rejects_changed_launch_plan(design, change):
    plan = launch_plan(design, "pilot")
    if change == "design":
        plan["design_sha256"] = "old-design"
    elif change == "stage":
        plan["stage"] = "primary"
    elif change == "metric":
        plan["metric"] = "eval/another_task/bpb"
    elif change == "count":
        plan["new_training_runs"] += 1
    elif change == "missing":
        plan["runs"].pop()
    elif change == "duplicate":
        plan["runs"][-1] = plan["runs"][0]
    elif change == "seed":
        plan["runs"][0]["data_seed"] += 1
    elif change == "blank":
        plan["runs"][0]["fingerprint"] = ""
    with pytest.raises(ValueError, match=r"Launch plan|Duplicate launch plan"):
        analysis.analyze(design, measurements(design, "pilot"), "pilot", plan)


def test_analysis_rejects_results_outside_checked_stage(design):
    with pytest.raises(ValueError, match="outside the planned stage"):
        analysis.analyze(design, measurements(design, "primary"), "pilot", launch_plan(design, "pilot"))


def test_refinement_analysis_uses_expanded_grid_without_counting_alias_as_new_run(design):
    # An unmeasured-for-proxies target point cannot lower the refinement grid's regret reference.
    observations = tuple(
        replace(observation, observed_bpb=-10.0) if observation.starcoder_weight == 0.8 else observation
        for observation in design.target_observations
    )
    design = replace(design, target_observations=observations)
    payload = design.to_dict()
    payload.pop("design_sha256")
    design = replace(design, design_sha256=design_module.canonical_sha256(payload))
    runs = {run.run_name: run for run in design.runs}
    records = tuple(
        (
            replace(record, value=10.0 + (runs[record.run_name].starcoder_weight - 0.5) ** 2)
            if runs[record.run_name].arm == "matched"
            else record
        )
        for record in measurements(design, "refinement")
    )

    report, points = analysis.analyze(design, records, "refinement", launch_plan(design, "refinement"))

    assert report["grid"] == [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.9, 1.0]
    assert report["new_measurements_used"] == 20
    assert report["proxy_measurements_used"] == 19
    assert report["proxy_curve_points"] == 20
    assert len(points) == 30
    assert sum(point.is_alias for point in points) == 1
    assert report["selection"]["matched"]["selected_weight"] == 0.5
    assert report["selection"]["matched"]["target_grid_regret"] == 0.0
    assert report["target_minimum"]["bpb"] == 1.0
    assert report["interpretation"]["selection_grid"] == "adaptive refinement grid"
    endpoint = next(point for point in points if point.arm == "target" and point.starcoder_weight == 1.0)
    assert endpoint.source_kind == "new"
    assert endpoint.observed_bpb == 5.0


def test_refinement_analysis_refuses_to_select_from_pilot_results_alone(design):
    with pytest.raises(ValueError, match="missing final measurements"):
        analysis.analyze(design, measurements(design, "pilot"), "refinement", launch_plan(design, "refinement"))
