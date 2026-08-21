# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["fsspec", "gcsfs", "numpy", "pandas", "scipy", "wandb"]
# ///
"""Fit the phase-TV response from the 3e18 ladder and test its preregistered optimum.

The composite-proposal panel measured one point of this response at phase total variation 0.24 and
implied a quadratic: writing ``gain(t) = -kappa t + (rho/2) t^2`` gave ``kappa = 0.008671``,
``rho = 0.105104``, an interior optimum at ``t* = 0.0825`` and a gain there of ``-0.000358`` BPB. Those
four numbers went into this panel's manifest before any of its runs existed, so the ladder is a test of
them rather than a fit to them.

Four levels, both orientations, three matched seed blocks, one tied control per block, and an aggregate
identical across all 27 rows to 2e-12. That design gives the odd/even decomposition at every level:
``odd = (L+ - L-)/2`` is the part of the response that phase ordering controls, and
``cost = (L+ + L-)/2 - L0`` is what any asymmetry costs regardless of direction. A two-phase policy
beats its tied control only where the odd part exceeds the cost, so the ladder's real question is where
that crossing happens, not whether any single level looks good.

**The orientation is preregistered as ``plus``** -- from the composite panel, which called it correctly
in both seed blocks on both objectives. Scoring the better orientation after seeing both outcomes is a
selection statistic that wins about half the time on noise, so the preregistered gain is the headline
and the post-hoc best is reported beside it only to show the size of the curse.

Bootstrapping resamples whole seed blocks, since the seed is the nuisance the design pairs against and
resampling rows within a block would break that pairing.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path

import fsspec
import numpy as np
import pandas as pd
import wandb
from scipy import stats

logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
REFERENCE_OUTPUTS = SCRIPT_DIR / "reference_outputs"
PANEL_DIR = REFERENCE_OUTPUTS / "delphi_3e18_uncheatable_phase_tv_ladder_20260727"
DEFAULT_OUTPUT_DIR = REFERENCE_OUTPUTS / "delphi_3e18_uncheatable_phase_tv_ladder_results_20260727"

TRAIN_PROJECT = "marin-community/marin"
EVAL_PROJECT = "marin-community/marin-eval"
TRAIN_TAG = "delphi-3e18-uncheatable-phase-tv-ladder"
EVAL_GROUP = "olmo_base_eval_table9_delphi_3e18_uncheatable_phase_tv_ladder_20260727"
UNCHEATABLE_METRIC = "eval/uncheatable_eval/bpb"
TABLE9_METRIC = "olmo_base_easy/table9_51_component_macro_bpb"
EXPECTED_ROWS = 27
PREREGISTERED_SIGN = "plus"
RUN_SIGMA = {"uncheatable": 0.000913, "table9": 0.003772}
TARGETS = (("uncheatable", "uncheatable_bpb"), ("table9", "table9_macro_bpb"))
BOOTSTRAP_DRAWS = 4000
BOOTSTRAP_SEED = 20260727


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel-dir", type=Path, default=PANEL_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--wandb-timeout", type=int, default=180)
    parser.add_argument("--draws", type=int, default=BOOTSTRAP_DRAWS)
    return parser.parse_args()


def collect_results(panel: pd.DataFrame, timeout: int) -> pd.DataFrame:
    """Join the panel to its training and native Table-9 runs by the launcher's short run name."""
    api = wandb.Api(timeout=timeout)
    training = list(api.runs(TRAIN_PROJECT, filters={"tags": {"$in": [TRAIN_TAG]}}, per_page=EXPECTED_ROWS + 60))
    evaluations = list(api.runs(EVAL_PROJECT, filters={"group": EVAL_GROUP}, per_page=EXPECTED_ROWS + 120))
    logger.info("found %d training and %d evaluation runs", len(training), len(evaluations))

    rows = []
    for index, entry in panel.reset_index(drop=True).iterrows():
        phase_tv = float(entry["phase_tv"])
        short = f"tvladder_{index:02d}_{entry['sign']}_tv{phase_tv:g}_s{int(entry['seed_block'])}"
        train = next((run for run in training if run.name.startswith(short)), None)
        # Retried evaluations appear more than once; only the finished attempt carries the metric.
        attempts = [run for run in evaluations if run.name == f"t9_{short}"]
        finished = [run for run in attempts if run.state == "finished"]
        native = finished[0] if finished else None
        table9, source = float("nan"), "missing"
        if native is not None:
            table9, source = float(native.summary.get(TABLE9_METRIC, np.nan)), "wandb_finished_summary"
        else:
            for attempt in sorted(attempts, key=lambda run: str(run.created_at), reverse=True):
                recovered = persisted_table9(attempt)
                if recovered is not None:
                    table9, source = recovered[0], recovered[1]
                    native = attempt
                    break
        rows.append(
            {
                "candidate_id": str(entry["candidate_id"]),
                "wandb_name": short,
                "sign": entry["sign"],
                "phase_tv": phase_tv,
                "seed_block": int(entry["seed_block"]),
                "uncheatable_bpb": float(train.summary.get(UNCHEATABLE_METRIC, np.nan)) if train else np.nan,
                "table9_macro_bpb": table9,
                "table9_metric_source": source,
                "training_wandb_url": train.url if train else None,
                "eval_wandb_url": native.url if native else None,
            }
        )
    return pd.DataFrame(rows)


def persisted_table9(run) -> tuple[float, str] | None:
    """Recover a Table-9 macro from the step's GCS output when its W&B run died before logging.

    An evaluation can finish its computation, write results, and still lose the W&B run -- that is what
    happened to one row here, where Iris reports the task succeeded while the W&B run is `crashed` with
    an empty summary. The persisted result is authoritative in that case, but only if the executor
    marked the step SUCCESS, so a partially written directory is never read as a result.
    """
    output_path = str(run.config.get("output_path") or "").rstrip("/")
    if not output_path.startswith("gs://marin-us-east5/"):
        return None
    result_path = f"{output_path}/olmo_base_eval_table9_results.json"
    status_path = f"{output_path}/.executor_status"
    fs, _, paths = fsspec.get_fs_token_paths(result_path)
    status_fs, _, status_paths = fsspec.get_fs_token_paths(status_path)
    if not paths or not status_paths or not fs.exists(paths[0]) or not status_fs.exists(status_paths[0]):
        return None
    with status_fs.open(status_paths[0], "r") as source:
        if source.read().strip() != "SUCCESS":
            return None
    with fs.open(paths[0], "r") as source:
        value = float(json.load(source)["table9_macro_bpb"])
    if not math.isfinite(value):
        raise ValueError(f"Non-finite persisted Table-9 value for {run.name!r}: {value}")
    return value, result_path


def decompose_levels(results: pd.DataFrame, column: str) -> pd.DataFrame:
    """Odd effect, asymmetry cost and preregistered gain at every level and seed block."""
    rows = []
    for (seed_block, phase_tv), group in results[results["sign"] != "center"].groupby(["seed_block", "phase_tv"]):
        control = results[(results["seed_block"] == seed_block) & (results["sign"] == "center")]
        if group[column].isna().any() or control[column].isna().any() or set(group["sign"]) != {"plus", "minus"}:
            continue
        plus = float(group.loc[group["sign"] == "plus", column].iloc[0])
        minus = float(group.loc[group["sign"] == "minus", column].iloc[0])
        tied = float(control[column].iloc[0])
        preregistered = plus if PREREGISTERED_SIGN == "plus" else minus
        rows.append(
            {
                "seed_block": int(seed_block),
                "phase_tv": float(phase_tv),
                "plus": plus,
                "minus": minus,
                "tied": tied,
                "odd_effect": 0.5 * (plus - minus),
                "asymmetry_cost": 0.5 * (plus + minus) - tied,
                "preregistered_gain": preregistered - tied,
                "posthoc_best_gain": min(plus, minus) - tied,
                "orientation_call_correct": bool(plus < minus),
            }
        )
    return pd.DataFrame(rows)


def fit_quadratic(levels: pd.DataFrame) -> dict[str, float]:
    """Least squares on ``gain(t) = -kappa t + (rho/2) t^2`` through the origin.

    No intercept: a tied policy is the control, so the response is zero at zero tilt by construction and
    fitting an intercept would let the curve absorb control noise into the mechanism.
    """
    tilt = levels["phase_tv"].to_numpy()
    gain = levels["preregistered_gain"].to_numpy()
    design = np.column_stack([-tilt, 0.5 * tilt**2])
    solution, *_ = np.linalg.lstsq(design, gain, rcond=None)
    kappa, rho = float(solution[0]), float(solution[1])
    optimum = kappa / rho if rho > 0 else float("nan")
    return {
        "kappa": kappa,
        "rho": rho,
        "optimum_tv": optimum,
        "gain_at_optimum": -(kappa**2) / (2.0 * rho) if rho > 0 else float("nan"),
        "has_interior_optimum": bool(rho > 0 and kappa > 0),
    }


def bootstrap_fit(levels: pd.DataFrame, draws: int, seed: int) -> dict[str, float]:
    """Resample whole seed blocks and refit, since the design pairs against the seed."""
    generator = np.random.default_rng(seed)
    blocks = sorted(levels["seed_block"].unique())
    optima, gains, kappas = [], [], []
    for _draw in range(draws):
        drawn = generator.choice(blocks, size=len(blocks), replace=True)
        sample = pd.concat([levels[levels["seed_block"] == block] for block in drawn], ignore_index=True)
        fit = fit_quadratic(sample)
        kappas.append(fit["kappa"])
        if fit["has_interior_optimum"]:
            optima.append(fit["optimum_tv"])
            gains.append(fit["gain_at_optimum"])
    return {
        "interior_optimum_share": len(optima) / max(draws, 1),
        "optimum_p05": float(np.quantile(optima, 0.05)) if optima else float("nan"),
        "optimum_p95": float(np.quantile(optima, 0.95)) if optima else float("nan"),
        "gain_p05": float(np.quantile(gains, 0.05)) if gains else float("nan"),
        "gain_p95": float(np.quantile(gains, 0.95)) if gains else float("nan"),
        "kappa_positive_share": float(np.mean(np.asarray(kappas) > 0.0)),
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    panel_files = sorted(args.panel_dir.glob("ladder_panel-*.csv"))
    assert len(panel_files) == 1, f"expected one ladder panel, found {panel_files}"
    panel = pd.read_csv(panel_files[0])
    manifest = json.loads((args.panel_dir / "ladder_manifest.json").read_text())
    implied = manifest["implied_response"]
    assert len(panel) == EXPECTED_ROWS, f"panel has {len(panel)} rows, expected {EXPECTED_ROWS}"

    results = collect_results(panel, args.wandb_timeout)
    results.to_csv(args.output_dir / "observed_results.csv", index=False)
    for target, column in TARGETS:
        missing = int(results[column].isna().sum())
        if missing:
            logger.warning("%s is missing %d of %d rows", target, missing, len(results))

    print("\nPREREGISTERED from the composite panel, before any ladder run existed:")
    print(f"  kappa {implied['kappa_per_tv']:.6f}   rho {implied['rho_per_tv_squared']:.6f}")
    print(f"  t* {implied['implied_optimum_tv']:.4f}   gain at t* {implied['implied_gain_at_optimum_bpb']:+.6f} BPB")
    print(f"  orientation preregistered as '{PREREGISTERED_SIGN}'\n")

    records, fits = [], []
    for target, column in TARGETS:
        sigma = RUN_SIGMA[target]
        levels = decompose_levels(results, column)
        if levels.empty:
            logger.warning("no complete blocks for %s", target)
            continue
        levels.insert(0, "target", target)
        records.append(levels)

        print("=" * 104)
        print(f"{target.upper()}   (run sigma {sigma:.6f})")
        print("=" * 104)
        for phase_tv, group in levels.groupby("phase_tv"):
            odd = group["odd_effect"].mean()
            cost = group["asymmetry_cost"].mean()
            gain = group["preregistered_gain"].mean()
            correct = int(group["orientation_call_correct"].sum())
            print(
                f"  tv {phase_tv:.2f}  odd {odd / sigma:+6.2f}s  cost {cost / sigma:+6.2f}s  "
                f"prereg gain {gain / sigma:+6.2f}s  posthoc {group['posthoc_best_gain'].mean() / sigma:+6.2f}s  "
                f"orientation {correct}/{len(group)}  {'WINS' if gain < 0 else 'loses'}"
            )

        fit = fit_quadratic(levels)
        interval = bootstrap_fit(levels, args.draws, BOOTSTRAP_SEED)
        fits.append({"target": target, **fit, **interval})
        print(f"\n  fitted kappa {fit['kappa']:+.6f}   rho {fit['rho']:+.6f}")
        if fit["has_interior_optimum"]:
            print(
                f"  t* {fit['optimum_tv']:.4f} [{interval['optimum_p05']:.4f}, {interval['optimum_p95']:.4f}]   "
                f"gain at t* {fit['gain_at_optimum']:+.6f} "
                f"({fit['gain_at_optimum'] / sigma:+.2f}s) "
                f"[{interval['gain_p05'] / sigma:+.2f}s, {interval['gain_p95'] / sigma:+.2f}s]"
            )
        else:
            print("  no interior optimum: the fitted response has no minimum at positive tilt")
        print(
            f"  bootstrap: interior optimum in {interval['interior_optimum_share'] * 100:.0f}% of block "
            f"resamples, kappa positive in {interval['kappa_positive_share'] * 100:.0f}%"
        )
        best = levels.groupby("phase_tv")["preregistered_gain"].mean()
        wins = best[best < 0]
        print(f"  levels where the preregistered orientation beats tied: {len(wins)}/{len(best)}")
        paired = stats.ttest_1samp(levels["preregistered_gain"].to_numpy(), 0.0)
        print(f"  pooled preregistered gain {levels['preregistered_gain'].mean():+.6f}  p={paired.pvalue:.3f}\n")

    if records:
        pd.concat(records, ignore_index=True).to_csv(args.output_dir / "level_decomposition.csv", index=False)
    fit_table = pd.DataFrame(fits)
    fit_table.to_csv(args.output_dir / "quadratic_fits.csv", index=False)

    print("=" * 104)
    print("AGAINST THE PREREGISTERED PREDICTION")
    print("=" * 104)
    for _, row in fit_table.iterrows():
        sigma = RUN_SIGMA[row["target"]]
        print(f"\n  {row['target']}")
        print(f"    kappa predicted {implied['kappa_per_tv']:+.6f}  observed {row['kappa']:+.6f}")
        print(f"    rho   predicted {implied['rho_per_tv_squared']:+.6f}  observed {row['rho']:+.6f}")
        if row["has_interior_optimum"]:
            print(f"    t*    predicted {implied['implied_optimum_tv']:.4f}  observed {row['optimum_tv']:.4f}")
            error = row["gain_at_optimum"] - implied["implied_gain_at_optimum_bpb"]
            print(
                f"    gain  predicted {implied['implied_gain_at_optimum_bpb']:+.6f}  "
                f"observed {row['gain_at_optimum']:+.6f}  error {error / sigma:+.2f}s"
            )
        else:
            print("    no interior optimum observed, so the predicted t* is not confirmed")

    provenance = {
        "panel_file": panel_files[0].name,
        "panel_sha256": manifest["panel_sha256"],
        "train_tag": TRAIN_TAG,
        "eval_group": EVAL_GROUP,
        "preregistered_sign": PREREGISTERED_SIGN,
        "implied_response": implied,
        "rows": len(results),
        "bootstrap_draws": int(args.draws),
    }
    (args.output_dir / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"\nwrote {args.output_dir}")


if __name__ == "__main__":
    main()
