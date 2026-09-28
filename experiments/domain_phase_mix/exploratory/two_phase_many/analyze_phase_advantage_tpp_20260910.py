# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["numpy==2.3.5", "pandas==2.2.2"]
# ///
"""Summarize measured phase advantages across archived training horizons."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent / "reference_outputs"
CANONICAL = BASE / "two_phase_surrogate_collaborator_packet_20260721/data/canonical"
SMALL = BASE / "60m_39bucket_checkpoint_audit_20260724"
HORIZONS = BASE / "delphi_horizon_transfer_20260910"
SWEEP = BASE / "delphi_fixed_n_tpp_phase_sweep_results_20260713"
OUTPUT = BASE / "phase_advantage_tpp_evidence_20260910"
TARGETS = ("uncheatable_bpb", "table9_macro_bpb")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    inputs = [
        Path(__file__),
        *[
            CANONICAL / f"{setting}_{phase}_phase_fit.csv"
            for setting in ("300m", "delphi_3e18")
            for phase in ("one", "two")
        ],
        SMALL / "fit_single_phase.csv",
        SMALL / "fit_two_phase.csv",
        HORIZONS / "predictions.csv",
        HORIZONS / "tpp_definitions.csv",
        HORIZONS / "artifact_hashes.json",
        SWEEP / "observed_results.csv",
    ]
    identity = {str(p): digest(p) for p in inputs}
    frozen = OUTPUT / "input_identity.json"
    if frozen.exists():
        assert json.loads(frozen.read_text()) == identity, "Source changed; use a new output directory."
    else:
        frozen.write_text(json.dumps(identity, indent=2) + "\n")
    prior = json.loads((HORIZONS / "artifact_hashes.json").read_text())
    for name in ("predictions.csv", "tpp_definitions.csv"):
        assert digest(HORIZONS / name) == prior[name]
    definitions = pd.read_csv(HORIZONS / "tpp_definitions.csv").set_index("setting")
    reference = pd.read_csv(CANONICAL / "300m_two_phase_fit.csv")
    domains = sorted(c.split("::")[1] for c in reference if c.startswith("phase_0_weight::"))
    columns = [f"phase_{phase}_weight::{domain}" for phase in (0, 1) for domain in domains]
    weights = reference[columns].to_numpy().reshape(-1, 2, len(domains))
    asymmetric = np.max(abs(weights[:, 0] - weights[:, 1]), axis=1) > 1e-10
    groups = sorted(reference.loc[asymmetric, "group_id"])
    assert len(groups) == 238
    reference_weights = reference.set_index("group_id").loc[groups, columns].to_numpy()
    summaries, paired, checks = [], [], []
    for setting, stem, alpha in [
        ("Delphi 3e18", "delphi_3e18", 2400 / 3007),
        ("Llama 160M/1.2B", "60m", 0.8),
        ("Llama 200M/6B", "300m", 0.8),
    ]:
        if stem == "60m":
            one = pd.read_csv(SMALL / "fit_single_phase.csv")
            two = pd.read_csv(SMALL / "fit_two_phase.csv")
            one["group_id"], two["group_id"] = one.paired_run_name, two.run_name
            rename = {f"phase_{p}_{d}": f"phase_{p}_weight::{d}" for p in (0, 1) for d in domains}
            one, two = one.rename(columns=rename), two.rename(columns=rename)
        else:
            one = pd.read_csv(CANONICAL / f"{stem}_one_phase_fit.csv")
            two = pd.read_csv(CANONICAL / f"{stem}_two_phase_fit.csv")
        assert one.group_id.is_unique and two.group_id.is_unique
        one, two = one.set_index("group_id").loc[groups], two.set_index("group_id").loc[groups]
        w1 = one[columns].to_numpy().reshape(-1, 2, len(domains))
        w2 = two[columns].to_numpy().reshape(-1, 2, len(domains))
        aggregate = alpha * w2[:, 0] + (1 - alpha) * w2[:, 1]
        tied_error = float(np.max(abs(w1[:, 0] - w1[:, 1])))
        aggregate_error = float(np.max(abs(w1[:, 0] - aggregate)))
        policy_error = float(np.max(abs(two[columns].to_numpy() - reference_weights)))
        assert max(tied_error, aggregate_error, policy_error) < 1e-12
        checks.append(
            dict(
                setting=setting,
                pairs=len(groups),
                tied_error=tied_error,
                aggregate_error=aggregate_error,
                policy_error=policy_error,
            )
        )
        for target in TARGETS:
            a, b = one[target].to_numpy(), two[target].to_numpy()
            assert np.isfinite(a).all() and np.isfinite(b).all()
            gain = a - b
            best = int(np.argmin(b))
            summaries.append(
                dict(
                    setting=setting,
                    tpp_total=definitions.loc[setting, "tpp_total"],
                    target=target,
                    n=len(a),
                    mean_paired_gain=gain.mean(),
                    median_paired_gain=np.median(gain),
                    fraction_asymmetric_better=np.mean(gain > 0),
                    best_tied=a.min(),
                    best_asymmetric=b.min(),
                    best_sampled_gain=a.min() - b.min(),
                    gain_at_best_asymmetric=gain[best],
                    best_asymmetric_group=groups[best],
                )
            )
            paired.append(
                pd.DataFrame(
                    dict(setting=setting, group_id=groups, target=target, tied_bpb=a, asymmetric_bpb=b, gain=gain)
                )
            )
    pd.DataFrame(summaries).to_csv(OUTPUT / "completed_swarm_summary.csv", index=False)
    pd.concat(paired, ignore_index=True).to_csv(OUTPUT / "matched_policy_gains.csv", index=False)
    observations = pd.read_csv(SWEEP / "observed_results.csv")
    fixed_rows = []
    for tpp, data in observations.groupby("tpp"):
        tied = data[data.policy.eq("tied")].iloc[0]
        treatments = data[data.policy.str.startswith("epsilon_")]
        assert len(treatments) == 4
        assert treatments.data_seed.eq(tied.data_seed).all() and treatments.trainer_seed.eq(tied.trainer_seed).all()
        for target in TARGETS:
            gain = tied[target] - treatments[target]
            assert np.max(abs(gain - treatments[f"gain_vs_tied_{target}"])) < 1e-12
            for index, row in treatments.iterrows():
                fixed_rows.append(
                    dict(
                        tpp_total=tpp,
                        target=target,
                        policy=row.policy,
                        gain=gain.loc[index],
                        tied_bpb=tied[target],
                        asymmetric_bpb=row[target],
                        data_seed=row.data_seed,
                        trainer_seed=row.trainer_seed,
                    )
                )
    pd.DataFrame(fixed_rows).to_csv(OUTPUT / "fixed_model_sweep_gains.csv", index=False)
    horizon = pd.read_csv(HORIZONS / "predictions.csv").query("context == 'full'")
    rows = []
    selected_orders = []
    for setting, data in horizon.groupby("horizon"):
        asymmetric = data[~data.tied]
        assert len(asymmetric) == 156 and len(data[data.tied]) == 2
        selected_orders.append(set(asymmetric.order))
        unimax = data.loc[data.order.eq(1), "measured"].item()
        proportional = data.loc[data.order.eq(0), "measured"].item()
        best = asymmetric.loc[asymmetric.measured.idxmin()]
        rows.append(
            dict(
                setting=setting,
                n_asymmetric=len(asymmetric),
                n_tied=2,
                best_asymmetric=best.measured,
                best_order=best.order,
                unimax=unimax,
                proportional=proportional,
                gain_vs_unimax=unimax - best.measured,
                gain_vs_proportional=proportional - best.measured,
                count_better_than_unimax=int((asymmetric.measured < unimax).sum()),
            )
        )
    assert selected_orders[0] == selected_orders[1]
    pd.DataFrame(rows).to_csv(OUTPUT / "partial_tpp40_comparison.csv", index=False)
    (OUTPUT / "CHECKS.json").write_text(
        json.dumps(
            dict(
                completed_swarm_checks=checks,
                common_partial_asymmetric_policies=156,
                fixed_sweep_treatments_per_horizon=4,
                prior_horizon_hashes_verified=True,
            ),
            indent=2,
        )
        + "\n"
    )
    print(pd.DataFrame(summaries).round(6).to_string(index=False))
    print(pd.DataFrame(rows).round(6).to_string(index=False))


if __name__ == "__main__":
    main()
