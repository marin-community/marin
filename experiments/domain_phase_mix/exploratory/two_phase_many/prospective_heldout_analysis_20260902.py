# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Prospective evaluation of frozen models on the successor-proposed Delphi coordinates of the refreshed registry."""

import numpy as np
import pandas as pd

D = (
    "experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/"
    "single_phase_observatory_benchmark_20260902"
)
HD = "experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/single_phase_heldout_benchmark_20260902"
SOURCE = "weibull_softplus_unscaled_epoch_cap"
pd.set_option("display.width", 260)
pred = pd.read_csv(f"{D}/external_heldout_predictions.csv")
pred = pred[pred.panel.eq("delphi_3e18_39bucket")].copy()
new = pred[pred.sources.astype(str).str.contains(SOURCE)]
print(
    "Delphi coordinates in the predictions table:",
    pred.coordinate_id.nunique(),
    "| new (successor-proposed):",
    new.coordinate_id.nunique(),
)
frozen = [
    "dsp_total_exposure",
    "dsp_total_exposure_concentration",
    "bucket_family_power_grp",
    "weibull_family_grp_shared_onset",
    "olmix_loglinear_taskwise",
    "grp_pair_power",
    "weibull_softplus_shared",
    "weibull_softplus_unscaled",
]
for target in ("uncheatable", "table9"):
    sub = pred[pred.target.eq(target)]
    old = sub[~sub.sources.astype(str).str.contains(SOURCE)].drop_duplicates("coordinate_id")
    best_old = float(old.measured_mean_bpb.min())
    fresh = sub[sub.sources.astype(str).str.contains(SOURCE)].drop_duplicates("coordinate_id")
    print(
        f"\n=== Delphi {target}: pre-sweep bank best {best_old:.4f}; new coordinates measured: "
        f"min {fresh.measured_mean_bpb.min():.4f}, median {fresh.measured_mean_bpb.median():.4f}; "
        f"{int((fresh.measured_mean_bpb < best_old).sum())} of {len(fresh)} beat the pre-sweep best"
    )
    rows = []
    for mid in frozen:
        m = sub[sub.model.eq(mid)]
        if m.empty:
            continue
        mn = m[m.sources.astype(str).str.contains(SOURCE)]
        err = mn.prediction - mn.measured_mean_bpb
        pick = m.loc[m.prediction.idxmin()]
        rank_all = int((m.measured_mean_bpb < pick.measured_mean_bpb).sum()) + 1
        spearman_new = (
            float(mn[["prediction", "measured_mean_bpb"]].corr(method="spearman").iloc[0, 1]) if len(mn) > 2 else np.nan
        )
        rows.append(
            {
                "model": mid,
                "new_pts": len(mn),
                "bias": float(err.mean()),
                "rmse_new": float(np.sqrt(np.mean(err**2))),
                "spearman_new": spearman_new,
                "argmin_is_new": bool(SOURCE in str(pick.sources)),
                "argmin_measured": float(pick.measured_mean_bpb),
                "argmin_rank_full_bank": rank_all,
                "regret_full_bank": float(pick.measured_mean_bpb - m.measured_mean_bpb.min()),
            }
        )
    print(pd.DataFrame(rows).round(4).to_string(index=False))
    print("new coordinates (measured, successor prediction):")
    s = sub[sub.model.eq("weibull_softplus_unscaled") & sub.sources.astype(str).str.contains(SOURCE)].sort_values(
        "measured_mean_bpb"
    )
    print(s[["coordinate_id", "measured_mean_bpb", "prediction"]].round(4).to_string(index=False))
