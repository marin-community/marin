# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Collect the UniMax-8 trainer-seed repeats at Delphi 3e18 into a repeats summary for the scaling figure.

Seed 0 is the archived ladder run (its values come from the scaling snapshot the figure already uses); seeds 1 and 2
are the 22 September repeats at the same data seed. Uncheatable uses the frozen seven-component weighting of the final
inline evaluation row, the paper's metric, not the new evaluator's byte-pooled parent aggregate; OlmoBaseEval Easy is
the native 51-component macro. Writes ``measured_results.csv``, ``repeats_summary.csv`` (the fairness-summary schema)
and ``receipt.json``.

usage: uv run --offline --no-sync python -m \
    experiments.domain_phase_mix.exploratory.two_phase_many.collect_unimax8_repeats_3e18_20260922 [--allow-partial]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path

import fsspec
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REFERENCE = SCRIPT_DIR / "reference_outputs"
OUTPUT_DIR = REFERENCE / "delphi_unimax8_repeats_3e18_20260922"
SNAPSHOT = REFERENCE / "delphi_scaling_progress_20260625" / "delphi_scaling_completed_wandb.csv"
UNCHEATABLE_FIT = REFERENCE / "delphi_frozen_procedure_validation_3e18_20260908" / "fits" / "fit_uncheatable.json"
TRAINING_ROOT = "gs://marin-us-east5/pinlin_calvin_xu/data_mixture/delphi_baseline_mixtures_issue6607_20260623"
TABLE9_ROOT = "gs://marin-us-east5/evaluation/olmo_base_eval_table9"
DATA_SEED = 660700
REPEATS = {1: "unimax8_3e18_t1", 2: "unimax8_3e18_t2"}
REQUEST_SET_VERSION = "1"
COMPONENTS = 51
METRICS = {"uncheatable": "uncheatable_bpb", "table9": "table9_macro_bpb"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_directory(fs, root: str, prefix: str) -> str:
    """The unique executor output directory whose name is ``prefix-<hash>``."""
    matches = [p for p in fs.ls(root) if p.split("/")[-1].startswith(prefix + "-")]
    if len(matches) != 1:
        raise ValueError(f"{prefix}: expected one directory under {root}, found {matches}")
    return matches[0]


def frozen_weighted_uncheatable(fs, directory: str, weights: dict[str, float]) -> tuple[float, int]:
    rows = [
        json.loads(line) for line in fs.cat(f"{directory}/checkpoints/eval_metrics.jsonl").decode().splitlines() if line
    ]
    final = [row for row in rows if any(key in row for key in weights)][-1]
    if any(key not in final for key in weights):
        raise ValueError(f"{directory}: final evaluation row lacks an Uncheatable component")
    return sum(weight * float(final[key]) for key, weight in weights.items()), int(final["step"])


def table9_macro(fs, directory: str) -> tuple[float, dict]:
    results = json.loads(fs.cat(f"{directory}/olmo_base_eval_table9_results.json").decode())
    components = results["table9_components"]
    if len(components) != COMPONENTS or results["request_set_version"] != REQUEST_SET_VERSION:
        raise ValueError(f"{directory}: unexpected Table 9 result shape")
    provenance = {key: results[key] for key in ("request_set_dir", "request_set_version", "olmo_eval_git_sha")}
    return statistics.fmean(components.values()), provenance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--allow-partial", action="store_true", help="write the summary even if a repeat is unfinished")
    args = parser.parse_args()
    fit = json.loads(UNCHEATABLE_FIT.read_text())
    weights = {task["component"]: float(weight) for task, weight in zip(fit["tasks"], fit["task_weights"], strict=True)}
    snapshot = pd.read_csv(SNAPSHOT)
    anchor = snapshot[snapshot["run_base"].eq("unimax8_3e18")]
    if len(anchor) != 1 or int(anchor["data_seed"].iloc[0]) != DATA_SEED:
        raise ValueError("The ladder's UniMax-8 3e18 run is missing from the snapshot or has another data seed")
    rows = [
        {
            "run": str(anchor["wandb_name"].iloc[0]),
            "trainer_seed": 0,
            "data_seed": DATA_SEED,
            "uncheatable_bpb": float(anchor["eval_uncheatable_eval_bpb"].iloc[0]),
            "table9_macro_bpb": float(anchor["olmo_base_easy_table9_51_component_macro_bpb"].iloc[0]),
            "uncheatable_source": f"{SNAPSHOT.name}:eval_uncheatable_eval_bpb",
            "table9_source": f"{SNAPSHOT.name}:olmo_base_easy_table9_51_component_macro_bpb",
        }
    ]
    fs = fsspec.filesystem("gcs")
    provenance = {}
    for seed, name in REPEATS.items():
        training = run_directory(fs, TRAINING_ROOT, name)
        if fs.cat(f"{training}/.executor_status").decode().strip() != "SUCCESS":
            raise ValueError(f"{training}: training has not succeeded")
        uncheatable, step = frozen_weighted_uncheatable(fs, training, weights)
        row = {
            "run": training.split("/")[-1],
            "trainer_seed": seed,
            "data_seed": DATA_SEED,
            "uncheatable_bpb": uncheatable,
            "table9_macro_bpb": float("nan"),
            "uncheatable_source": f"{training}/checkpoints/eval_metrics.jsonl:step {step}, frozen weighting",
            "table9_source": "",
        }
        try:
            evaluation = run_directory(fs, TABLE9_ROOT, f"t9_{name}")
            macro, provenance[name] = table9_macro(fs, evaluation)
            row["table9_macro_bpb"], row["table9_source"] = macro, f"{evaluation}/olmo_base_eval_table9_results.json"
        except (ValueError, FileNotFoundError) as error:
            if not args.allow_partial:
                raise ValueError(f"{name}: Table 9 result unavailable ({error})") from error
        rows.append(row)
    measured = pd.DataFrame(rows)
    summary = []
    for target, metric in METRICS.items():
        values = measured[metric].dropna()
        seeds = measured.loc[values.index, "trainer_seed"]
        if len(values) != 3 and not args.allow_partial:
            raise ValueError(f"{metric}: expected three seeds, found {len(values)}")
        summary.append(
            {
                "kind": "policy",
                "candidate_id": "unimax8",
                "label": "UniMax-8 (ladder mixture, trainer-seed repeats)",
                "metric": metric,
                "seeds": ";".join(str(int(s)) for s in seeds),
                "values": ";".join(f"{v:.4f}" for v in values),
                "mean": float(values.mean()),
                "sd": float(values.std(ddof=1)) if len(values) > 1 else float("nan"),
                "n": len(values),
                "standard_error": float(values.std(ddof=1) / len(values) ** 0.5) if len(values) > 1 else float("nan"),
                "target": target,
            }
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    measured.to_csv(args.output_dir / "measured_results.csv", index=False)
    pd.DataFrame(summary).to_csv(args.output_dir / "repeats_summary.csv", index=False)
    receipt = {
        "snapshot_sha256": sha256(SNAPSHOT),
        "uncheatable_fit_sha256": sha256(UNCHEATABLE_FIT),
        "table9_provenance": provenance,
        "partial": bool(measured["table9_macro_bpb"].isna().any()),
        "measured_results_sha256": sha256(args.output_dir / "measured_results.csv"),
        "repeats_summary_sha256": sha256(args.output_dir / "repeats_summary.csv"),
    }
    (args.output_dir / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(measured[["run", "trainer_seed", "uncheatable_bpb", "table9_macro_bpb"]].to_string(index=False))
    print(pd.DataFrame(summary)[["metric", "seeds", "mean", "sd", "n"]].to_string(index=False))


if __name__ == "__main__":
    main()
