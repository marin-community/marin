# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Analyze the Delphi 3e18 seed-replicate panel (2026-10-23): how far seed replicates of one fixed curriculum diverge.

Inputs: the sixteen new runs (arms B1 and C at four data seeds x two trainer seeds; training-time Uncheatable/C4 evals
every 600 steps, Table-9 at hook step 2400 and at the end) and the six archive runs of the same two arms at data seed
662009 (trainer seeds 0-2; evals at 1000/2000/3000/3006, Table-9 at the end). Outputs, under
``reference_outputs/delphi_seed_replicates_20261023/analysis/``: per-run metrics, the trajectory of the seed spread,
seed SDs per arm and readout with bootstrap intervals, the end/switch amplification ratio, the paired B1-C delta, and
``report.md``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import gcsfs
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
PANEL_DIR = SCRIPT_DIR / "reference_outputs" / "delphi_seed_replicates_20261023"
ANALYSIS_DIR = PANEL_DIR / "analysis"
NEW_ROOT = "marin-us-east5/pinlin_calvin_xu/data_mixture/delphi_seed_replicates_3e18_20261023"
ARCHIVE_ROOT = "marin-us-east5/pinlin_calvin_xu/data_mixture/delphi_sft_probe_arms_3e18_20261014"
EVAL_ROOT = "marin-us-east5/evaluation/olmo_base_eval_table9"
ARCHIVE_RUNS = {f"sftp_{arm.lower()}_t{seed}": (arm, 662_009, seed) for arm in ("B1", "C") for seed in range(3)}
ARCHIVE_POOL = "v6e-8"
CANARY = "sdrep_b1_d10230000_t0"  # the one new run trained on v6e-8; the others ran on v5p-8
SWITCH_STEP = 2400
FINAL_STEP = 3006
METRICS = {
    "uncheatable": "eval/uncheatable_eval/bpb",
    "c4": "eval/paloma/c4_100_domains/bpb",
}
UNCHEATABLE_COMPONENTS = (
    "ao3_english",
    "arxiv_computer_science",
    "arxiv_physics",
    "bbc_news",
    "github_cpp",
    "github_python",
    "wikipedia_english",
)
BOOTSTRAP_DRAWS = 4000
BETWEEN_CURRICULUM_SD = 0.027  # Table-9 SD of the 238 swarm pair deltas at a fixed aggregate (sft_midtraining_probe)
DEPLOYMENT_MARGIN = 0.002


def _read_jsonl(fs: gcsfs.GCSFileSystem, path: str) -> list[dict]:
    return [json.loads(line) for line in fs.open(path).read().decode().splitlines() if line.strip()]


def _run_dir(fs: gcsfs.GCSFileSystem, root: str, run_name: str) -> str:
    hits = [d for d in fs.ls(root) if d.split("/")[-1].startswith(f"{run_name}-")]
    if len(hits) != 1:
        raise ValueError(f"Expected one directory for {run_name} under {root}, found {hits}")
    return hits[0]


def _table9(fs: gcsfs.GCSFileSystem, eval_name: str) -> tuple[float, dict[str, float]]:
    hits = fs.glob(f"{EVAL_ROOT}/{eval_name}-*/olmo_base_eval_table9_results.json")
    if not hits:
        return float("nan"), {}
    result = json.load(fs.open(sorted(hits)[-1]))
    return float(result["table9_macro_bpb"]), {k: float(v) for k, v in result["table9_components"].items()}


def collect(fs: gcsfs.GCSFileSystem) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Per-run metrics, per-step trajectories, and per-component Table-9 values for the 22 runs."""
    panel = pd.read_csv(PANEL_DIR / "source_panel.csv")
    runs = [(r.candidate_id, r.arm, int(r.data_seed), int(r.trainer_seed), NEW_ROOT, "new") for r in panel.itertuples()]
    runs += [(name, arm, ds, ts, ARCHIVE_ROOT, "archive") for name, (arm, ds, ts) in ARCHIVE_RUNS.items()]
    run_rows, trajectory_rows, component_rows = [], [], []
    for run_name, arm, data_seed, trainer_seed, root, source in runs:
        directory = _run_dir(fs, root, run_name)
        evals = _read_jsonl(fs, f"{directory}/checkpoints/eval_metrics.jsonl")
        by_step = {int(e["step"]): e for e in evals if "step" in e}
        pool = ARCHIVE_POOL if source == "archive" or run_name == CANARY else "v5p-8"
        row = {
            "run_name": run_name,
            "arm": arm,
            "data_seed": data_seed,
            "trainer_seed": trainer_seed,
            "source": source,
            "pool": pool,
            "eval_steps": len(by_step),
        }
        for step in sorted(by_step):
            e = by_step[step]
            trajectory_rows.append(
                {"run_name": run_name, "arm": arm, "source": source, "step": step}
                | {metric: e.get(key) for metric, key in METRICS.items()}
                | {f"unch_{c}": e.get(f"eval/uncheatable_eval/{c}/bpb") for c in UNCHEATABLE_COMPONENTS}
            )
        for readout, step in (("switch", SWITCH_STEP), ("final", FINAL_STEP)):
            e = by_step.get(step, {})
            for metric, key in METRICS.items():
                row[f"{metric}_{readout}"] = e.get(key, float("nan"))
        t9_final, comps_final = _table9(fs, f"t9_{run_name}")
        row["table9_final"] = t9_final
        for component, value in comps_final.items():
            component_rows.append(
                {
                    "run_name": run_name,
                    "arm": arm,
                    "source": source,
                    "readout": "final",
                    "component": component,
                    "bpb": value,
                }
            )
        if source == "new":
            t9_switch, comps_switch = _table9(fs, f"t9_{run_name}_step{SWITCH_STEP}")
            row["table9_switch"] = t9_switch
            for component, value in comps_switch.items():
                component_rows.append(
                    {
                        "run_name": run_name,
                        "arm": arm,
                        "source": source,
                        "readout": "switch",
                        "component": component,
                        "bpb": value,
                    }
                )
        else:
            row["table9_switch"] = float("nan")
        run_rows.append(row)
    return pd.DataFrame(run_rows), pd.DataFrame(trajectory_rows), pd.DataFrame(component_rows)


def _sd_with_ci(values: np.ndarray, rng: np.random.Generator) -> tuple[float, float, float]:
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return float("nan"), float("nan"), float("nan")
    draws = [np.std(rng.choice(values, len(values), replace=True), ddof=1) for _ in range(BOOTSTRAP_DRAWS)]
    return float(np.std(values, ddof=1)), float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def seed_spread(runs: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """SD across seeds per arm, readout and metric, for the new eight-seed grid, the archive three-seed set, and both."""
    rows = []
    metrics = [c for c in runs.columns if c.endswith(("_switch", "_final"))]
    for arm in sorted(runs["arm"].unique()):
        for population, frame in (
            ("new_8_seeds", runs[(runs["arm"] == arm) & (runs["source"] == "new")]),
            ("archive_3_init_seeds", runs[(runs["arm"] == arm) & (runs["source"] == "archive")]),
            ("all_11", runs[runs["arm"] == arm]),
        ):
            for metric in metrics:
                values = frame[metric].to_numpy(dtype=float)
                sd, lo, hi = _sd_with_ci(values, rng)
                rows.append(
                    {
                        "arm": arm,
                        "population": population,
                        "metric": metric,
                        "n": int(np.isfinite(values).sum()),
                        "mean": float(np.nanmean(values)) if np.isfinite(values).any() else float("nan"),
                        "sd": sd,
                        "sd_ci_lo": lo,
                        "sd_ci_hi": hi,
                    }
                )
    return pd.DataFrame(rows)


def amplification(runs: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """End SD over switch SD per arm and metric, bootstrapped over the eight new seeds (paired)."""
    rows = []
    for arm in sorted(runs["arm"].unique()):
        frame = runs[(runs["arm"] == arm) & (runs["source"] == "new")]
        for metric in ("uncheatable", "c4", "table9"):
            switch = frame[f"{metric}_switch"].to_numpy(dtype=float)
            final = frame[f"{metric}_final"].to_numpy(dtype=float)
            mask = np.isfinite(switch) & np.isfinite(final)
            switch, final = switch[mask], final[mask]
            if len(switch) < 3:
                continue
            ratio = np.std(final, ddof=1) / np.std(switch, ddof=1)
            draws = []
            for _ in range(BOOTSTRAP_DRAWS):
                idx = rng.choice(len(switch), len(switch), replace=True)
                if np.std(switch[idx], ddof=1) > 0:
                    draws.append(np.std(final[idx], ddof=1) / np.std(switch[idx], ddof=1))
            corr = float(np.corrcoef(switch, final)[0, 1]) if len(switch) > 2 else float("nan")
            rows.append(
                {
                    "arm": arm,
                    "metric": metric,
                    "n": len(switch),
                    "sd_switch": float(np.std(switch, ddof=1)),
                    "sd_final": float(np.std(final, ddof=1)),
                    "ratio_final_over_switch": float(ratio),
                    "ratio_ci_lo": float(np.percentile(draws, 2.5)),
                    "ratio_ci_hi": float(np.percentile(draws, 97.5)),
                    "corr_switch_final": corr,
                }
            )
    return pd.DataFrame(rows)


def paired_contrast(runs: pd.DataFrame) -> pd.DataFrame:
    """B1 minus C at the same (data seed, trainer seed), for every readout and metric."""
    metrics = [c for c in runs.columns if c.endswith(("_switch", "_final"))]
    b1 = runs[runs["arm"] == "B1"].set_index(["data_seed", "trainer_seed"])
    c = runs[runs["arm"] == "C"].set_index(["data_seed", "trainer_seed"])
    common = b1.index.intersection(c.index)
    deltas = (b1.loc[common, metrics] - c.loc[common, metrics]).reset_index()
    deltas["source"] = b1.loc[common, "source"].to_numpy()
    summary = []
    for metric in metrics:
        for population, frame in (("new_8", deltas[deltas["source"] == "new"]), ("all_11", deltas)):
            values = frame[metric].to_numpy(dtype=float)
            values = values[np.isfinite(values)]
            if len(values) < 2:
                continue
            summary.append(
                {
                    "metric": metric,
                    "population": population,
                    "n_pairs": len(values),
                    "delta_mean": float(values.mean()),
                    "delta_se": float(values.std(ddof=1) / np.sqrt(len(values))),
                    "delta_sd": float(values.std(ddof=1)),
                }
            )
    deltas.to_csv(ANALYSIS_DIR / "paired_b1_minus_c_by_seed.csv", index=False)
    return pd.DataFrame(summary)


def trajectory_spread(trajectory: pd.DataFrame) -> pd.DataFrame:
    """SD across the eight new seeds at every logged step, per arm and metric."""
    new = trajectory[trajectory["source"] == "new"]
    metrics = ["uncheatable", "c4", *(f"unch_{c}" for c in UNCHEATABLE_COMPONENTS)]
    return new.groupby(["arm", "step"])[metrics].agg(["mean", "std", "count"]).round(5)


def component_spread(components: pd.DataFrame) -> pd.DataFrame:
    """Per Table-9 component: seed SD at the switch and at the end over the eight new seeds, per arm."""
    new = components[components["source"] == "new"]
    table = new.groupby(["arm", "component", "readout"])["bpb"].agg(["mean", "std", "count"]).unstack("readout")
    table.columns = [f"{stat}_{readout}" for stat, readout in table.columns]
    table["sd_ratio_final_over_switch"] = table["std_final"] / table["std_switch"]
    return table.round(5)


def seed_shift(runs: pd.DataFrame) -> pd.DataFrame:
    """Data-seed means (both trainer seeds pooled) against the archive's 662009 mean, per arm and final metric."""
    rows = []
    for arm in sorted(runs["arm"].unique()):
        frame = runs[runs["arm"] == arm]
        for metric in ("table9_final", "uncheatable_final", "c4_final"):
            by_seed = frame.groupby("data_seed")[metric].agg(["mean", "std", "count"])
            for data_seed, r in by_seed.iterrows():
                rows.append(
                    {
                        "arm": arm,
                        "metric": metric,
                        "data_seed": int(data_seed),
                        "mean": float(r["mean"]),
                        "sd_over_trainer_seeds": float(r["std"]),
                        "n": int(r["count"]),
                    }
                )
    return pd.DataFrame(rows)


def _md(frame: pd.DataFrame, **kwargs) -> str:
    return frame.to_markdown(**kwargs)


def render_report(runs, spread, amp, paired, shift, components) -> str:
    new = runs[runs["source"] == "new"]
    parts = [f"# Seed replicates of a fixed two-phase curriculum at Delphi 3e18 (analysis of {len(runs)} runs)", ""]
    parts += [
        f"New runs: {len(new)} ({new['table9_final'].notna().sum()} with a final Table-9, "
        f"{new['table9_switch'].notna().sum()} with a switch Table-9); archive runs at data seed 662009: "
        f"{len(runs) - len(new)}.",
        "",
    ]
    parts += [
        "## Seed SD per arm, readout and metric (bootstrap 95% interval on the SD)",
        "",
        _md(spread.round(5), index=False),
        "",
    ]
    parts += [
        "## Does phase 2 amplify or contract the divergence present at the switch?",
        "",
        "Ratio of the SD across the eight new seeds at the end to the SD at the switch (hook step 2400), "
        "bootstrapped over seeds.",
        "",
        _md(amp.round(4), index=False),
        "",
    ]
    parts += ["## Paired B1 minus C (same data seed and trainer seed)", "", _md(paired.round(5), index=False), ""]
    parts += ["## Data-seed means (trainer seeds pooled)", "", _md(shift.round(5), index=False), ""]
    parts += [
        "## Table-9 components: seed SD at the switch and the end (eight new seeds)",
        "",
        _md(components, index=True),
        "",
    ]
    parts += [
        "## Reference scales",
        "",
        f"- Between-curriculum spread at a fixed aggregate (238 swarm pairs, Table-9): SD {BETWEEN_CURRICULUM_SD}.",
        f"- Deployment margin used for two-phase claims: {DEPLOYMENT_MARGIN} BPB.",
        "- Archive seed components (Table-9): init only 0.001-0.006 over six two-phase arms; data order only "
        "0.0015-0.0067 over six Dolmino-late schedules; single-phase controls 0.003-0.006.",
        "",
    ]
    parts += ["## Per-run table", "", _md(runs.round(5), index=False), ""]
    return "\n".join(parts)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    fs = gcsfs.GCSFileSystem()
    rng = np.random.default_rng(20261023)
    runs, trajectory, components = collect(fs)
    runs.to_csv(ANALYSIS_DIR / "runs.csv", index=False)
    trajectory.to_csv(ANALYSIS_DIR / "trajectory.csv", index=False)
    components.to_csv(ANALYSIS_DIR / "table9_components.csv", index=False)
    spread = seed_spread(runs, rng)
    spread.to_csv(ANALYSIS_DIR / "seed_sd.csv", index=False)
    amp = amplification(runs, rng)
    amp.to_csv(ANALYSIS_DIR / "amplification.csv", index=False)
    paired = paired_contrast(runs)
    paired.to_csv(ANALYSIS_DIR / "paired_b1_minus_c.csv", index=False)
    shift = seed_shift(runs)
    shift.to_csv(ANALYSIS_DIR / "data_seed_means.csv", index=False)
    traj = trajectory_spread(trajectory)
    traj.to_csv(ANALYSIS_DIR / "trajectory_spread.csv")
    comp = component_spread(components) if not components.empty else pd.DataFrame()
    comp.to_csv(ANALYSIS_DIR / "table9_component_spread.csv")
    report = render_report(runs, spread, amp, paired, shift, comp)
    (ANALYSIS_DIR / "report.md").write_text(report)
    print(report[:6000])


if __name__ == "__main__":
    main()
