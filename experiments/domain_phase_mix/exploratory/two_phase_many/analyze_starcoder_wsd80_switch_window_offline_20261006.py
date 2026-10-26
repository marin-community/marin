# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Switch-window offline evals: control against training-time evals, then dense double differences.

Inputs: the CSV written by ``evaluate_starcoder_wsd80_switch_window.py --collect`` and the persisted
training-time evals of the review-v9 trajectories (``gcf_v9_evals.parquet`` from
``pull_starcoder_wsd80_gradient_conflict_evals_20261006.py``).

Usage: uv run python analyze_starcoder_wsd80_switch_window_offline_20261006.py <offline.csv> <gcf_evals.parquet>
<out_dir>
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

PL_ONLINE = "eval/paloma/dolma_100_programing_languages-llama3/bpb"
C4_ONLINE = "eval/paloma/c4_en-llama3/bpb"
METRICS = {"PL": ("dolma_100_programing_languages_bpb", PL_ONLINE), "C4": ("c4_en_bpb", C4_ONLINE)}
DECAY_SWITCH = 22608
H5_SWITCH = {"boundary_beta_0p60": 16896, "boundary_beta_0p70": 19712}
P_FINAL, H5_FINAL = 28259, 28159


def ci(values) -> tuple[float, float]:
    values = np.asarray(values, float)
    n = len(values)
    half = stats.t.ppf(0.975, n - 1) * values.std(ddof=1) / np.sqrt(n) if n > 1 else float("nan")
    return float(values.mean()), float(half)


def main() -> None:
    offline = pd.read_csv(sys.argv[1])
    online = pd.read_parquet(sys.argv[2]).sort_values("step").drop_duplicates(["trajectory_id", "step"], keep="last")
    out = Path(sys.argv[3])
    out.mkdir(parents=True, exist_ok=True)

    # 1. control: offline final-step values against the persisted training-time evals
    finals = offline[offline.step.isin([P_FINAL, H5_FINAL])].merge(
        online[["trajectory_id", "step", PL_ONLINE, C4_ONLINE]], on=["trajectory_id", "step"], how="left"
    )
    for name, (off_col, on_col) in METRICS.items():
        diff = finals[off_col] - finals[on_col]
        print(
            f"control {name}: n={diff.notna().sum()} mean offline-online {diff.mean():+.5f} "
            f"max |diff| {diff.abs().max():.5f}"
        )

    rows = []
    # 2. P3 against P2 and P4: offline window plus the training-time evals at +392 and +1392
    merged = offline.copy()
    for name, (off_col, _) in METRICS.items():
        merged = merged.rename(columns={off_col: name})
    extra = online[online.trajectory_id.str.startswith("gcf_p") & online.step.isin([23000, 24000, 25000])].copy()
    extra = extra.rename(columns={PL_ONLINE: "PL", C4_ONLINE: "C4"})
    extra["arm"] = extra.trajectory_id.str.extract(r"gcf_(p\d)_")[0]
    extra["support_id"] = extra.trajectory_id.str.extract(r"r3d28260_(m100a|full)_")[0]
    extra["training_seed"] = extra.trajectory_id.str.extract(r"_s(\d+)$")[0].astype(int)
    p_rows = pd.concat(
        [
            merged[merged.arm.isin(["p2", "p3", "p4"])][["arm", "support_id", "training_seed", "step", "PL", "C4"]],
            extra[["arm", "support_id", "training_seed", "step", "PL", "C4"]],
        ]
    )
    for support in ("m100a", "full"):
        for comparator in ("p2", "p4"):
            for metric in ("PL", "C4"):
                p3 = p_rows[(p_rows.arm == "p3") & (p_rows.support_id == support)].pivot(
                    index="training_seed", columns="step", values=metric
                )
                cmp_ = p_rows[(p_rows.arm == comparator) & (p_rows.support_id == support)].pivot(
                    index="training_seed", columns="step", values=metric
                )
                seeds = sorted(set(p3.index) & set(cmp_.index))
                gap = p3.loc[seeds] - cmp_.loc[seeds]
                if 22544 not in gap.columns or 22352 not in gap.columns:
                    continue
                base = gap[22544]
                for step in sorted(gap.columns):
                    values = (gap[step] - base).dropna()
                    if step == 22544 or values.empty:
                        continue
                    mean, half = ci(values)
                    rows.append(
                        dict(
                            arm=f"p3_vs_{comparator}_{support}",
                            metric=metric,
                            step=step,
                            lag=step - DECAY_SWITCH,
                            n=len(values),
                            double_diff=mean,
                            ci=half,
                            baseline_step=22544,
                        )
                    )
    # 3. H5 constant-LR switches: within-run change relative to the -64 checkpoint
    for role, switch in H5_SWITCH.items():
        sub = merged[merged.arm == role]
        for metric in ("PL", "C4"):
            table = sub.pivot(index="training_seed", columns="step", values=metric)
            if switch - 64 not in table.columns:
                continue
            base = table[switch - 64]
            for step in sorted(table.columns):
                if step == switch - 64 or step == H5_FINAL:
                    continue
                values = (table[step] - base).dropna()
                mean, half = ci(values)
                rows.append(
                    dict(
                        arm=f"h5_{role[-4:].replace('0p', '0.')}",
                        metric=metric,
                        step=step,
                        lag=step - switch,
                        n=len(values),
                        double_diff=mean,
                        ci=half,
                        baseline_step=switch - 64,
                        note="within-run change",
                    )
                )
    result = pd.DataFrame(rows).sort_values(["arm", "metric", "lag"])
    result.to_csv(out / "switch_window_offline_double_differences.csv", index=False)
    pd.set_option("display.width", 200)
    print(result.pivot_table(index=["arm", "lag"], columns="metric", values=["double_diff", "ci"]).round(4).to_string())


if __name__ == "__main__":
    main()
