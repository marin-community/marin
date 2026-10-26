# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Collect the transition-ramp endpoints and compare them with the August abrupt and tied runs.

Uses the offline final-checkpoint evaluations written by
``evaluate_starcoder_wsd80_switch_window.py --collect`` for both the ramp-launch runs and the
August 4x confirmation runs, so every endpoint comes from one evaluation code version, then
reports ramp minus abrupt per seed (paired-t over seeds), the canary's drift against its August
twin, and the two-phase gain under each schedule. The training-time evals of the ramp runs are
read too, for the eval histories.

Usage: uv run python collect_starcoder_wsd80_transition_ramp_20261006.py <out_dir> <offline_evals.csv>
"""

import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from rigging.filesystem.buckets import filesystem_for
from scipy import stats

HERE = Path(__file__).resolve().parent
DESIGNS = (
    HERE.parents[1] / "starcoder_wsd80_transition_ramp_design_20261006.json",
    HERE.parents[1] / "starcoder_wsd80_transition_ramp_quick_design_20261007.json",
    HERE.parents[1] / "starcoder_wsd80_transition_ramp_abrupt_design_20261007.json",
)
CONFIRMATION = (
    HERE
    / "reference_outputs/starcoder_wsd80_dense_support_empirical_optimum_confirmation_results_20260811"
    / "confirmation_observations.csv"
)
RAMP_ROOT = "gs://marin-us-central1/checkpoints/pinlin_calvin_xu/data_mixture/starcoder_wsd80_transition_ramp_20261006"
VERSION = "2026.07.11"
PL = "eval/paloma/dolma_100_programing_languages-llama3/bpb"
C4 = "eval/paloma/c4_en-llama3/bpb"
FINAL_STEP = 28259


def persisted_evals(run_name: str) -> pd.DataFrame | None:
    url = f"{RAMP_ROOT}/{run_name}/{VERSION}/checkpoints/eval_metrics.jsonl"
    fs, path = filesystem_for(url)
    if not fs.exists(path):
        return None
    with fs.open(path, "rb") as handle:
        rows = [json.loads(line) for line in handle.read().decode().splitlines() if line.strip()]
    frame = pd.DataFrame(rows).sort_values("step").drop_duplicates("step", keep="last")
    frame["run_name"] = run_name
    return frame


def paired_t(values: np.ndarray) -> tuple[float, float, float]:
    values = np.asarray(values, float)
    n = len(values)
    mean = values.mean()
    half = stats.t.ppf(0.975, n - 1) * values.std(ddof=1) / np.sqrt(n) if n > 1 else float("nan")
    return mean, mean - half, mean + half


def main() -> None:
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    runs = []
    for design_path in DESIGNS:
        if design_path.exists():
            with design_path.open() as handle:
                runs.extend(json.load(handle)["runs"])
    with ThreadPoolExecutor(8) as executor:
        frames = [f for f in executor.map(persisted_evals, [r["run_name"] for r in runs]) if f is not None]
    if frames:
        pd.concat(frames).to_csv(out / "ramp_eval_history.csv", index=False)
    offline = pd.read_csv(sys.argv[2])
    offline = offline[offline.step == FINAL_STEP].set_index("trajectory_id")
    meta = pd.DataFrame(runs).set_index("run_name")
    august = pd.read_csv(CONFIRMATION)
    august = august[(august.support_id == "m400") & (august.rung == 3)]

    def offline_pl(run_name: str) -> float | None:
        return float(offline.loc[run_name, "dolma_100_programing_languages_bpb"]) if run_name in offline.index else None

    abrupt_aug = {int(r.pair_seed): offline_pl(r.run_name) for r in august[august.policy_class == "untied"].itertuples()}
    tied_aug = {int(r.pair_seed): offline_pl(r.run_name) for r in august[august.policy_class == "tied"].itertuples()}
    if any(v is None for v in [*abrupt_aug.values(), *tied_aug.values()]):
        raise SystemExit("August 4x final checkpoints are not all evaluated offline yet")

    rows = []
    for run_name, row in meta.iterrows():
        pl = offline_pl(run_name)
        if pl is None:
            rows.append({"run_name": run_name, "arm": row.arm, "data_seed": row.data_seed, "status": "pending"})
            continue
        rows.append(
            {
                "run_name": run_name,
                "arm": row.arm,
                "data_seed": row.data_seed,
                "status": "done",
                "pl_bpb": pl,
                "c4_bpb": float(offline.loc[run_name, "c4_en_bpb"]),
                "august_abrupt_pl": abrupt_aug[row.data_seed],
                "august_tied_pl": tied_aug[row.data_seed],
                "minus_august_abrupt": pl - abrupt_aug[row.data_seed],
                "gain_vs_august_tied": tied_aug[row.data_seed] - pl,
            }
        )
    table = pd.DataFrame(rows)
    october_abrupt = {
        int(row.data_seed): float(row.pl_bpb)
        for row in table[(table.arm == "abrupt") & (table.status == "done")].itertuples()
    }
    table["october_abrupt_pl"] = table.data_seed.map(october_abrupt)
    table["minus_october_abrupt"] = table.pl_bpb - table.october_abrupt_pl
    table.to_csv(out / "ramp_endpoints.csv", index=False)
    done = table[table.status == "done"]
    print(f"{len(done)}/{len(table)} runs landed")
    print(done.to_string(index=False))
    canary = done[done.arm == "abrupt"]
    if len(canary):
        print(f"\ncanary drift (new abrupt minus August twin): {canary.minus_august_abrupt.iloc[0]:+.5f} BPB")
    print("\nramp minus August abrupt, paired over seeds (95% paired-t); gain = August tied minus ramp:")
    for arm, group in done[done.arm != "abrupt"].groupby("arm"):
        m, lo, hi = paired_t(group.minus_august_abrupt)
        g, glo, ghi = paired_t(group.gain_vs_august_tied)
        print(
            f"  {arm}: n={len(group)} ramp-abrupt {m:+.5f} [{lo:+.5f}, {hi:+.5f}]   "
            f"gain {g:+.5f} [{glo:+.5f}, {ghi:+.5f}]"
        )
    g0, g0lo, g0hi = paired_t([tied_aug[seed] - abrupt_aug[seed] for seed in sorted(abrupt_aug)])
    print(f"  August abrupt gain for reference: {g0:+.5f} [{g0lo:+.5f}, {g0hi:+.5f}]")

    # within-launch contrasts between ramp widths (same code era and training stack, paired by seed)
    wide = done.pivot(index="data_seed", columns="arm", values="pl_bpb")
    arms = [a for a in wide.columns if a != "abrupt"]
    print("\nwithin-launch width contrasts, paired over seeds (95% paired-t):")
    for i, a in enumerate(arms):
        for b in arms[i + 1 :]:
            d = (wide[b] - wide[a]).dropna()
            if len(d) > 1:
                m, lo, hi = paired_t(d)
                print(
                    f"  {b} minus {a}: n={len(d)} {m:+.5f} [{lo:+.5f}, {hi:+.5f}]  "
                    f"worse in {int((d > 0).sum())}/{len(d)}"
                )

    # gains over the October abrupt switch (abrupt PL minus ramp PL; positive means the ramp is better)
    if "abrupt" in wide.columns and wide["abrupt"].notna().sum() > 1:
        wide_c4 = done.pivot(index="data_seed", columns="arm", values="c4_bpb")
        gain_rows = []
        for a in arms:
            d = (wide["abrupt"] - wide[a]).dropna()
            m, lo, hi = paired_t(d)
            c4 = (wide_c4["abrupt"] - wide_c4[a]).dropna()
            c4m, c4lo, c4hi = paired_t(c4)
            g = done[done.arm == a].gain_vs_august_tied
            gm, glo, ghi = paired_t(g)
            gain_rows.append(
                {
                    "arm": a,
                    "n": len(d),
                    "pl_gain_over_october_abrupt": m,
                    "pl_gain_ci_low": lo,
                    "pl_gain_ci_high": hi,
                    "ramp_worse_in": int((d < 0).sum()),
                    "c4_gain_over_october_abrupt": c4m,
                    "c4_gain_ci_low": c4lo,
                    "c4_gain_ci_high": c4hi,
                    "two_phase_gain_vs_august_tied": gm,
                    "two_phase_gain_ci_low": glo,
                    "two_phase_gain_ci_high": ghi,
                }
            )
        drift = done[done.arm == "abrupt"].minus_august_abrupt
        dm, dlo, dhi = paired_t(drift)
        gains = pd.DataFrame(gain_rows)
        gains.to_csv(out / "ramp_gains_over_october_abrupt.csv", index=False)
        print("\ngain over the October abrupt switch (abrupt minus ramp, positive = ramp better), paired over seeds:")
        for row in gains.itertuples():
            print(
                f"  {row.arm}: n={row.n} PL {row.pl_gain_over_october_abrupt:+.5f} "
                f"[{row.pl_gain_ci_low:+.5f}, {row.pl_gain_ci_high:+.5f}] ramp worse in {row.ramp_worse_in}/{row.n}; "
                f"C4 {row.c4_gain_over_october_abrupt:+.5f} [{row.c4_gain_ci_low:+.5f}, {row.c4_gain_ci_high:+.5f}]; "
                f"two-phase gain vs August tied {row.two_phase_gain_vs_august_tied:+.5f} "
                f"[{row.two_phase_gain_ci_low:+.5f}, {row.two_phase_gain_ci_high:+.5f}]"
            )
        print(f"  drift, October abrupt minus August abrupt: n={len(drift)} {dm:+.5f} [{dlo:+.5f}, {dhi:+.5f}]")


if __name__ == "__main__":
    main()
