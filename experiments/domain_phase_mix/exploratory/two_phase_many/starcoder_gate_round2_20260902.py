# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Per-curve StarCoder gate: in-sample fit and argmin (tied-curves protocol) plus OOF metrics, candidates vs DSP.

usage: uv run python starcoder_gate.py model_a,model_b [--oof]
"""

import sys

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from experiments.domain_phase_mix.exploratory.two_phase_many import benchmark_single_phase_observatory_20260902 as h
from experiments.domain_phase_mix.exploratory.two_phase_many import single_phase_observatory_registry_20260902 as r

D = (
    "experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/"
    "single_phase_observatory_benchmark_20260902"
)
MODELS = sys.argv[1].split(",")
OOF = "--oof" in sys.argv
INSAMPLE = "--insample-objective" in sys.argv
ref = pd.read_csv(f"{D}/../starcoder_all_tied_curves_canonical_dsp_20260902/curve_reference.csv").set_index("curve_id")
plan = h.tier_plan("screen")
rows = []
for cid in plan.curves:
    panel = h.load_panel(f"{h.STARCODER_PANEL_PREFIX}{cid}")
    y = panel.groups[0].outcomes[:, 0].astype(float)
    n = panel.rows
    train = np.arange(n)
    inner = ((train, train),) if INSAMPLE else h.heldout_inner_folds(panel)
    sc = int(np.argmax(panel.features.exposures.max(0)))
    p = panel.features.weights[:, sc]
    number = int(ref.curve_number.get(cid, -1))
    for mid in MODELS:
        e = r.ENTRY_BY_ID[mid]
        f = r.apply_transform(panel.features, e)
        m = e.build(f)
        fit = m.fit(f, y, train, inner, 0)
        pred = np.asarray(m.predict(fit, f, train), float)
        row = {
            "curve": number,
            "curve_id": cid,
            "model": mid,
            "insample_rmse": float(np.sqrt(np.mean((pred - y) ** 2))),
            "insample_regret": float(y[np.argmin(pred)] - y.min()),
            "argmin_pred": float(p[np.argmin(pred)]),
            "argmin_obs": float(p[np.argmin(y)]),
        }
        rows.append(row)
out = pd.DataFrame(rows)
if OOF:
    m = pd.read_csv(f"{D}/screen/starcoder_one_dimensional_curve_metrics.csv")
    m = m[["model", "curve_id", "rmse", "regret_at_1", "spearman"]].rename(
        columns={"rmse": "oof_rmse", "regret_at_1": "oof_regret"}
    )
    out = out.merge(m, on=["model", "curve_id"], how="left")
out.to_csv(sys.argv[-1] if sys.argv[-1].endswith(".csv") else "/dev/null", index=False)
pd.set_option("display.width", 250)
base = out[out.model.eq(MODELS[0])].set_index("curve")
for mid in MODELS[1:]:
    cand = out[out.model.eq(mid)].set_index("curve")
    worse_fit = int((cand.insample_rmse > base.insample_rmse + 1e-9).sum())
    worse_reg = int((cand.insample_regret > base.insample_regret + 1e-9).sum())
    line = (
        f"{mid}: in-sample RMSE worse than {MODELS[0]} on {worse_fit}/45 curves; "
        f"in-sample regret worse on {worse_reg}/45"
    )
    if OOF:
        oof_fit = int((cand.oof_rmse > base.oof_rmse + 1e-9).sum())
        oof_reg = int((cand.oof_regret > base.oof_regret + 1e-9).sum())
        line += f"; OOF RMSE worse on {oof_fit}/45; OOF regret worse on {oof_reg}/45"
    print(line)
    worst = cand.sort_values("insample_regret", ascending=False).head(4)
    print(
        "   worst in-sample regret curves:",
        worst[["insample_regret", "argmin_pred", "argmin_obs"]].round(3).to_dict("index"),
    )
