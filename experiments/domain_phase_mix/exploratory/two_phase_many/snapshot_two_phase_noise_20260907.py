# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy==2.3.5", "pandas==2.2.2"]
# ///
"""Recover joint task noise without changing the frozen link-transfer inputs.

The Table-9 anchor reference has eleven actual proportional observations. The
expanded packet replaces its baseline Table-9 row by a repeat mean, so that
packet is used only to recover the baseline's atomic Uncheatable observation.
The ten additional Uncheatable observations come from the raw metric registry.
All matrices use the existing prepared component order and objective weights.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]
REFERENCE = SCRIPT_DIR / "reference_outputs"
ROOT = REFERENCE / "two_phase_link_transfer_20260907"
OUTPUT = ROOT / "noise"
TABLE9_SOURCE = (
    REFERENCE
    / "one_phase_swarm_scores_export_300m_20260630"
    / "proportional_reference_uncheatable_table9_scores_300m.csv"
)
UNCHEATABLE_SOURCE = (
    SCRIPT_DIR / "metric_registry" / "raw_metric_matrix_300m" / "noise_baseline_proportional_variable_subset_300m_6b.csv"
)
PACKET = REFERENCE / "two_phase_solver_gap_collaborator_packet_20260701" / "data" / "all_300m_checkpoint_metrics.csv"
LAUNCHER = REPO_ROOT / "experiments" / "domain_phase_mix" / "launch_proportional_variable_subset_noise_baseline.py"


def main() -> None:
    with np.load(ROOT / "inputs" / "panel.npz", allow_pickle=False) as archive:
        panel = {name: archive[name] for name in archive.files}
    reference = pd.read_csv(TABLE9_SOURCE).set_index("run_name", verify_integrity=True)
    raw = pd.read_csv(UNCHEATABLE_SOURCE).set_index("run_name", verify_integrity=True)
    packet = pd.read_csv(PACKET).set_index("run_name", verify_integrity=True)
    components = np.concatenate([panel["uncheatable_components"], panel["table9_components"]])
    runs = reference.index.to_numpy(str)
    assert len(reference) == 11 and len(raw) == 10
    assert runs[0] == "baseline_proportional" and set(runs[1:]) == set(raw.index)
    assert np.array_equal(
        reference.loc[runs[1:], "wandb_run_id"].to_numpy(), raw.loc[runs[1:], "wandb_run_id"].to_numpy()
    )
    baseline_uncheatable = packet.loc[runs[0], [name.replace("/", "_") for name in panel["uncheatable_components"]]]
    uncheatable = np.vstack([baseline_uncheatable.to_numpy(float), raw.loc[runs[1:], panel["uncheatable_components"]]])
    table9 = reference.loc[runs, panel["table9_components"]].to_numpy(float)
    outcomes = np.column_stack([uncheatable, table9])
    assert outcomes.shape == (11, 58) and np.isfinite(outcomes).all()
    weights = np.zeros((2, 58))
    weights[0, :7] = panel["uncheatable_aggregation_weights"]
    weights[1, 7:] = panel["table9_aggregation_weights"]
    aggregate = outcomes @ weights.T
    u_error = float(np.max(np.abs(aggregate[:, 0] - reference["eval_uncheatable_eval_bpb"].to_numpy(float))))
    t_error = float(np.max(np.abs(aggregate[:, 1] - reference["table9_macro_bpb"].to_numpy(float))))
    assert max(u_error, t_error) < 3e-6
    covariance = np.cov(outcomes, rowvar=False, ddof=1)
    objective_covariance = weights @ covariance @ weights.T
    assert np.max(np.abs(objective_covariance - np.cov(aggregate, rowvar=False, ddof=1))) < 1e-16
    means = outcomes.mean(axis=0)
    deviations = outcomes.std(axis=0, ddof=1)
    comparison = pd.DataFrame(
        {
            "component": components,
            "target": ["uncheatable"] * 7 + ["table9"] * 51,
            "frozen_proportional_bpb": panel["anchor_proportional_bpb"],
            "empirical_proportional_bpb": means,
            "frozen_repeat_sd": panel["anchor_repeat_sd"],
            "empirical_repeat_sd": deviations,
            "sd_ratio_empirical_to_frozen": deviations / panel["anchor_repeat_sd"],
            "mean_shift": means - panel["anchor_proportional_bpb"],
        }
    )
    fit_paths = []
    margin_records = []
    for target in ("uncheatable", "table9"):
        offset = 0 if target == "uncheatable" else 7
        for index, component in enumerate(panel[f"{target}_components"]):
            path = ROOT / "spines" / "final" / f"{target}_c{index}.json"
            fitted = json.loads(path.read_text())
            fit_paths.append(path)
            assert fitted["component"] == component
            provenance = fitted["meta"]["provenance"]
            train = np.asarray(provenance["train_rows"], dtype=int)
            minimum = float(panel[f"{target}_outcomes"][train, index].min())
            anchor = float(provenance["anchor"]["proportional_bpb"])
            sd = float(provenance["anchor"]["repeat_sd"])
            kappa = float(fitted["kappa"])
            gap = kappa * (anchor - minimum)
            old_floor = anchor - max(gap, 3.0 * sd)
            assert abs(old_floor - fitted["floor"]) < 1e-10
            empirical_sd = float(deviations[offset + index])
            empirical_anchor = float(means[offset + index])
            updated_gap = kappa * (empirical_anchor - minimum)
            margin_records.append(
                {
                    "target": target,
                    "component": str(component),
                    "kappa": kappa,
                    "train_minimum": minimum,
                    "frozen_margin_binds": 3.0 * sd >= gap,
                    "empirical_sd_margin_binds_frozen_mean": 3.0 * empirical_sd >= gap,
                    "empirical_mean_and_sd_margin_binds": 3.0 * empirical_sd >= updated_gap,
                    "frozen_floor": old_floor,
                    "sd_only_counterfactual_floor": anchor - max(gap, 3.0 * empirical_sd),
                    "mean_and_sd_counterfactual_floor": empirical_anchor - max(updated_gap, 3.0 * empirical_sd),
                }
            )
    margins = pd.DataFrame(margin_records)
    sources = [
        Path(__file__),
        TABLE9_SOURCE,
        UNCHEATABLE_SOURCE,
        PACKET,
        LAUNCHER,
        ROOT / "inputs" / "panel.npz",
        *fit_paths,
    ]
    source_hashes = {str(path.relative_to(REPO_ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}
    if OUTPUT.exists():
        manifest = json.loads((OUTPUT / "manifest.json").read_text())
        assert manifest["source_sha256"] == source_hashes, "changed noise inputs"
        for name, digest in manifest["output_sha256"].items():
            assert hashlib.sha256((OUTPUT / name).read_bytes()).hexdigest() == digest
        print(f"Verified existing noise snapshot: {OUTPUT}")
        return
    OUTPUT.mkdir()
    (OUTPUT / "proportional_reference.csv").write_bytes(TABLE9_SOURCE.read_bytes())
    (OUTPUT / "uncheatable_raw_repeats.csv").write_bytes(UNCHEATABLE_SOURCE.read_bytes())
    rows = reference[
        [
            "panel",
            "row_kind",
            "noise_trainer_seed",
            "noise_data_seed",
            "noise_simulated_epoch_subset_seed",
            "wandb_run_id",
        ]
    ].reset_index()
    rows["shared_fixed_subset_confirmed"] = False
    pd.concat([rows, pd.DataFrame(outcomes, columns=components)], axis=1).to_csv(
        OUTPUT / "component_repeats.csv", index=False
    )
    comparison.to_csv(OUTPUT / "anchor_comparison.csv", index=False)
    margins.to_csv(OUTPUT / "final_spine_floor_margin_sensitivity.csv", index=False)
    np.savez_compressed(
        OUTPUT / "noise.npz",
        runs=runs,
        components=components,
        outcomes=outcomes,
        uncheatable_outcomes=uncheatable,
        table9_outcomes=table9,
        centered_outcomes=outcomes - means,
        component_covariance=covariance,
        objective_names=np.asarray(["uncheatable", "table9"]),
        objective_weights=weights,
        objective_outcomes=aggregate,
        objective_covariance=objective_covariance,
        empirical_proportional_bpb=means,
        empirical_repeat_sd=deviations,
    )
    summary = {
        "observations": 11,
        "atomic_tasks": 58,
        "maximum_uncheatable_parity_error": u_error,
        "maximum_table9_parity_error": t_error,
        "objective_repeat_sd": dict(
            zip(("uncheatable", "table9"), np.sqrt(np.diag(objective_covariance)).tolist(), strict=True)
        ),
        "objective_covariance": objective_covariance.tolist(),
        "component_covariance_rank": int(np.linalg.matrix_rank(covariance)),
        "floor_margin_binding": (
            margins.groupby("target")[
                ["frozen_margin_binds", "empirical_sd_margin_binds_frozen_mean", "empirical_mean_and_sd_margin_binds"]
            ]
            .sum()
            .to_dict()
        ),
        "frozen_anchors_modified": False,
        "subset_evidence": (
            "Trainer seeds 10000..10009; launcher requires data_seed=None and simulated_epoch_subset_seed=None. "
            "All share the proportional policy, but no fixed or identical data subset is established."
        ),
        "baseline_table9_warning": (
            "The expanded packet substitutes the proportional repeat mean at baseline_proportional. "
            "Its Table-9 baseline is unsuitable for noise covariance; this snapshot uses the eleven actual observations."
        ),
        "sampling_scope": (
            "Resample centered 58-task vectors together to retain empirical correlations. "
            "Independent-run pair noise requires two independent row draws. This is proportional total-run noise, "
            "not measured covariance for each policy pair or each mixture region."
        ),
        "sample_size_limit": (
            "The 58-task covariance has at most rank 10 from eleven observations; "
            "do not treat all covariance directions as precisely identified."
        ),
        "floor_sensitivity_scope": (
            "Frozen final kappa and training minima; counterfactual floor formulas only, "
            "without refitting coefficients or selecting new hyperparameters."
        ),
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    manifest = {
        "source_sha256": source_hashes,
        "output_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(OUTPUT.iterdir())},
    }
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
