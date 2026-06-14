"""Export local per-problem analysis tables for downstream-scaling notebooks."""

from __future__ import annotations

import argparse
import gzip
import json
import os
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import fsspec
import numpy as np
import pandas as pd

from marin.execution.executor import Executor

import experiments.evals.delphi.gsm8k as standard_gsm8k
import experiments.downstream_scaling.evals.run_delphi_gsm8k_joint_decode as joint_decode_run
import experiments.downstream_scaling.evals.run_delphi_gsm8k_joint_decode_avg as joint_decode_avg_run
from experiments.downstream_scaling.evals.framework.schema import GRADES_FILENAME, read_grade_rows
from experiments.downstream_scaling.evals.run_delphi_masked_gsm8k_iid import (
    MASK_FRACTIONS,
    TPU_TYPE as MASKED_TPU_TYPE,
    build_steps as build_masked_steps,
)

DEFAULT_OUTPUT_DIR = Path("experiments/downstream_scaling/analysis/data")
TRAIN_ON_TEST_RESULTS_DIR = Path("experiments/train-on-test/results/sweep-wo-ckpts")

EU_WEST_PREFIX = "gs://marin-eu-west4"
US_EAST_PREFIX = "gs://marin-us-east5"

STANDARD_GRADE_PREFIX = "grades/delphi/gsm8k/"
STANDARD_CORRECT_FIELD = "correct_flexible_extract"
STANDARD_N_PROBLEMS = 256

TOPK_PATTERN = re.compile(r"topk_a(?P<top_k_a>\d+)_b(?P<top_k_b>\d+)")
ADVISOR_WEIGHT_PATTERN = re.compile(r"advisor_weight(?P<advisor_weight>\d+)")
TRAIN_ON_TEST_MODEL_SCALE_PATTERN = re.compile(r"delphi-(?P<scale>\d+e\d+)-")


def model_scale(model: str) -> float:
    return float(model.replace("e", "E"))


def train_on_test_model_scale(model: str) -> float:
    match = TRAIN_ON_TEST_MODEL_SCALE_PATTERN.search(model)
    if match is None:
        raise ValueError(f"Could not parse Delphi scale from model name: {model}")
    return float(match.group("scale"))


def train_on_test_model_label(model: str) -> str:
    match = TRAIN_ON_TEST_MODEL_SCALE_PATTERN.search(model)
    if match is None:
        return model
    return match.group("scale")


def path_exists(path: str) -> bool:
    fs, fs_path = fsspec.core.url_to_fs(path)
    return fs.exists(fs_path)


def read_jsonl_gz(path: str) -> list[dict[str, Any]]:
    with fsspec.open(path, "rt", compression="gzip") as fin:
        return [json.loads(line) for line in fin]


def write_table(data: pd.DataFrame, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    data.to_csv(output_dir / f"{stem}.csv", index=False)
    with gzip.open(output_dir / f"{stem}.jsonl.gz", "wt") as fout:
        for row in data.to_dict(orient="records"):
            fout.write(json.dumps(row) + "\n")


def aggregate_per_problem(data: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    return (
        data.groupby(group_columns, as_index=False)
        .agg(
            mean=("problem_accuracy", "mean"),
            n_problems=("problem_id", "nunique"),
            n_completions=("n_completions", "sum"),
        )
        .sort_values(group_columns)
    )


def downstream_grade_rows(records: Iterable[dict[str, Any]], dataset: str) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    missing_paths: list[dict[str, Any]] = []
    for record in records:
        grades_path = record["grades_path"]
        if not path_exists(grades_path):
            missing_paths.append(record)
            continue
        for grade_row in read_grade_rows(grades_path):
            scores = [float(grade["score"]) for grade in grade_row["grades"]]
            rows.append(
                {
                    "dataset": dataset,
                    **{key: value for key, value in record.items() if key != "grades_path"},
                    "problem_id": grade_row["id"],
                    "problem_accuracy": float(np.mean(scores)),
                    "n_completions": len(scores),
                    "source_path": grades_path,
                }
            )
    return pd.DataFrame(rows), missing_paths


def executor_for(prefix: str) -> Executor:
    return Executor(prefix=prefix, executor_info_base_path=f"{prefix}/experiments")


def masked_records() -> list[dict[str, Any]]:
    executor = executor_for(EU_WEST_PREFIX)
    steps = build_masked_steps(MASKED_TPU_TYPE)
    for step in steps:
        executor.compute_version(step, is_pseudo_dep=False)

    records = []
    for step in steps:
        parts = step.name.split("/")
        mask_index = int(parts[-3].removeprefix("mask_"))
        model = parts[-2]
        mask_fraction = MASK_FRACTIONS[mask_index]
        records.append(
            {
                "model": model,
                "model_scale": model_scale(model),
                "mask_fraction": mask_fraction,
                "unmasked_fraction": 1.0 - mask_fraction,
                "grades_path": os.path.join(executor.output_paths[step], GRADES_FILENAME),
            }
        )
    return records


def joint_decode_records() -> list[dict[str, Any]]:
    executor = executor_for(EU_WEST_PREFIX)
    steps = joint_decode_run.build_steps(joint_decode_run.TPU_TYPE)
    for step in steps:
        executor.compute_version(step, is_pseudo_dep=False)

    records = []
    for step in steps:
        parts = step.name.split("/")
        topk_name = parts[-3]
        model = parts[-2]
        match = TOPK_PATTERN.fullmatch(topk_name)
        if match is None:
            raise ValueError(f"Unexpected joint-decode step name: {step.name}")
        records.append(
            {
                "model": model,
                "model_scale": model_scale(model),
                "top_k_a": int(match.group("top_k_a")),
                "top_k_b": int(match.group("top_k_b")),
                "grades_path": os.path.join(executor.output_paths[step], GRADES_FILENAME),
            }
        )
    deduped = pd.DataFrame(records).drop_duplicates(subset=["model", "top_k_a", "top_k_b"], keep="first")
    return deduped.to_dict(orient="records")


def joint_decode_avg_records() -> list[dict[str, Any]]:
    executor = executor_for(US_EAST_PREFIX)
    steps = joint_decode_avg_run.build_steps(joint_decode_avg_run.TPU_TYPE)
    for step in steps:
        executor.compute_version(step, is_pseudo_dep=False)

    records = []
    for step in steps:
        parts = step.name.split("/")
        advisor_weight_name = parts[-3]
        model = parts[-2]
        match = ADVISOR_WEIGHT_PATTERN.fullmatch(advisor_weight_name)
        if match is None:
            raise ValueError(f"Unexpected joint-decode-avg step name: {step.name}")
        records.append(
            {
                "model": model,
                "model_scale": model_scale(model),
                "advisor_weight": int(match.group("advisor_weight")) / 100.0,
                "top_k_a": joint_decode_avg_run.TOP_K_A,
                "top_k_b": joint_decode_avg_run.TOP_K_B,
                "temperature": joint_decode_avg_run.TEMPERATURE,
                "grades_path": os.path.join(executor.output_paths[step], GRADES_FILENAME),
            }
        )
    return records


def standard_gsm8k_per_problem() -> pd.DataFrame:
    executor = executor_for(US_EAST_PREFIX)
    steps = standard_gsm8k.build_steps()
    for step in steps:
        executor.compute_version(step, is_pseudo_dep=False)

    grade_paths = {
        step.name[len(STANDARD_GRADE_PREFIX) :]: executor.output_paths[step]
        for step in steps
        if step.name.startswith(STANDARD_GRADE_PREFIX)
    }

    rows = []
    for model, output_path in grade_paths.items():
        path = f"{output_path}/graded.jsonl.gz"
        if not path_exists(path):
            continue
        graded_rows = read_jsonl_gz(path)
        if len(graded_rows) != STANDARD_N_PROBLEMS:
            raise ValueError(f"{model}: expected {STANDARD_N_PROBLEMS} graded rows, got {len(graded_rows)}")
        for row in graded_rows:
            scores = [float(score) for score in row[STANDARD_CORRECT_FIELD]]
            rows.append(
                {
                    "dataset": "standard_gsm8k",
                    "model": model,
                    "model_scale": model_scale(model),
                    "temperature": standard_gsm8k.TEMPERATURE,
                    "top_p": standard_gsm8k.TOP_P,
                    "top_k": standard_gsm8k.TOP_K,
                    "problem_id": int(row["problem_id"]),
                    "problem_accuracy": float(np.mean(scores)),
                    "n_completions": len(scores),
                    "source_path": path,
                }
            )
    return pd.DataFrame(rows)


def train_on_test_per_problem(results_dir: Path) -> pd.DataFrame:
    rows = []
    for model_dir in sorted(path for path in results_dir.iterdir() if path.is_dir()):
        generations_path = model_dir / "generations.json"
        if not generations_path.exists():
            continue
        payload = json.loads(generations_path.read_text())
        model = payload["model"]
        model_label = train_on_test_model_label(model)
        scale = train_on_test_model_scale(model)
        for alpha_result in payload["results"]:
            alpha = float(alpha_result["alpha"])
            generation_df = pd.DataFrame(alpha_result["generations"])
            problem_scores = (
                generation_df.assign(score=lambda data: data["score"].astype(float))
                .groupby("id", sort=True)
                .agg(
                    problem_accuracy=("score", "mean"),
                    n_completions=("score", "size"),
                )
                .reset_index()
            )
            for row in problem_scores.itertuples(index=False):
                rows.append(
                    {
                        "dataset": "train_on_test_alpha",
                        "model": model,
                        "model_label": model_label,
                        "model_scale": scale,
                        "alpha": alpha,
                        "problem_id": row.id,
                        "problem_accuracy": float(row.problem_accuracy),
                        "n_completions": int(row.n_completions),
                        "source_path": str(generations_path),
                    }
                )
    return pd.DataFrame(rows)


def export_downstream_dataset(
    output_dir: Path,
    dataset: str,
    records: list[dict[str, Any]],
    group_columns: list[str],
) -> dict[str, Any]:
    per_problem, missing_paths = downstream_grade_rows(records, dataset)
    if per_problem.empty:
        raise FileNotFoundError(f"No rows loaded for {dataset}")
    per_problem = per_problem.sort_values([*group_columns, "problem_id"]).reset_index(drop=True)
    aggregate = aggregate_per_problem(per_problem, group_columns)
    write_table(per_problem, output_dir, f"{dataset}_per_problem")
    write_table(aggregate, output_dir, f"{dataset}_aggregate")
    return {
        "dataset": dataset,
        "rows": len(per_problem),
        "aggregate_rows": len(aggregate),
        "missing_paths": len(missing_paths),
    }


def write_readme(output_dir: Path) -> None:
    readme = """# Local Analysis Data

This directory contains local snapshots of per-problem average-grade tables used by the downstream scaling notebooks and `experiments/evals/delphi/gsm8k.ipynb`.

Regenerate everything from the repo root with:

```bash
uv run python experiments/downstream_scaling/analysis/save_analysis_data.py
```

Each dataset is written twice:

- `<dataset>_per_problem.csv` and `<dataset>_per_problem.jsonl.gz`: one row per model / sweep setting / problem.
- `<dataset>_aggregate.csv` and `<dataset>_aggregate.jsonl.gz`: grouped means computed from the per-problem table.

The core columns are:

- `model`: Delphi model slug or Hugging Face model name.
- `model_scale`: numeric FLOP scale parsed from the model name, such as `1e22`.
- `problem_id`: GSM8K problem identifier.
- `problem_accuracy`: average grade for that problem across completions/samples.
- `n_completions`: number of completions/samples averaged for that problem.
- `source_path`: original local path or GCS object used for the export.

Datasets:

- `masked_gsm8k_iid`: from `experiments/downstream_scaling/evals/run_delphi_masked_gsm8k_iid.py`, used by `analysis/masking/delphi_masked_gsm8k_iid_results.ipynb`. Sweep columns: `mask_fraction`, `unmasked_fraction`.
- `joint_decode`: from `experiments/downstream_scaling/evals/run_delphi_gsm8k_joint_decode.py`, used by `analysis/decoding/delphi_gsm8k_joint_decode_results.ipynb`. Sweep columns: `top_k_a`, `top_k_b`.
- `joint_decode_avg`: from `experiments/downstream_scaling/evals/run_delphi_gsm8k_joint_decode_avg.py`, used by `analysis/decoding/delphi_gsm8k_joint_decode_avg_results.ipynb`. Sweep columns: `advisor_weight`, plus fixed `top_k_a`, `top_k_b`, and `temperature`.
- `train_on_test_alpha`: from local `experiments/train-on-test/results/sweep-wo-ckpts/*/generations.json`, used by `experiments/train-on-test/train_on_test_alpha_results.ipynb`. Sweep column: `alpha`.
- `standard_gsm8k`: from `experiments/evals/delphi/gsm8k.py`, used by `experiments/evals/delphi/gsm8k.ipynb` and as the temperature-0.6 baseline in the masking notebook. Sampling columns: `temperature`, `top_p`, `top_k`.
Example:

```python
import pandas as pd

masked = pd.read_csv("experiments/downstream_scaling/analysis/data/masked_gsm8k_iid_per_problem.csv")
masked_aggregate = (
    masked
    .groupby(["model", "model_scale", "mask_fraction", "unmasked_fraction"], as_index=False)
    .agg(mean=("problem_accuracy", "mean"), n_problems=("problem_id", "nunique"))
)
```
"""
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "README.md").write_text(readme)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--train-on-test-results-dir", type=Path, default=TRAIN_ON_TEST_RESULTS_DIR)
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    summaries = [
        export_downstream_dataset(
            output_dir,
            "masked_gsm8k_iid",
            masked_records(),
            ["dataset", "model", "model_scale", "mask_fraction", "unmasked_fraction"],
        ),
        export_downstream_dataset(
            output_dir,
            "joint_decode",
            joint_decode_records(),
            ["dataset", "model", "model_scale", "top_k_a", "top_k_b"],
        ),
        export_downstream_dataset(
            output_dir,
            "joint_decode_avg",
            joint_decode_avg_records(),
            ["dataset", "model", "model_scale", "advisor_weight", "top_k_a", "top_k_b", "temperature"],
        ),
    ]

    train_on_test = train_on_test_per_problem(args.train_on_test_results_dir)
    train_on_test = train_on_test.sort_values(["model_scale", "alpha", "problem_id"]).reset_index(drop=True)
    train_on_test_aggregate = aggregate_per_problem(
        train_on_test,
        ["dataset", "model", "model_label", "model_scale", "alpha"],
    )
    write_table(train_on_test, output_dir, "train_on_test_alpha_per_problem")
    write_table(train_on_test_aggregate, output_dir, "train_on_test_alpha_aggregate")
    summaries.append(
        {
            "dataset": "train_on_test_alpha",
            "rows": len(train_on_test),
            "aggregate_rows": len(train_on_test_aggregate),
            "missing_paths": 0,
        }
    )

    standard = standard_gsm8k_per_problem().sort_values(["model_scale", "problem_id"]).reset_index(drop=True)
    standard_aggregate = aggregate_per_problem(
        standard,
        ["dataset", "model", "model_scale", "temperature", "top_p", "top_k"],
    )
    write_table(standard, output_dir, "standard_gsm8k_per_problem")
    write_table(standard_aggregate, output_dir, "standard_gsm8k_aggregate")
    summaries.append(
        {
            "dataset": "standard_gsm8k",
            "rows": len(standard),
            "aggregate_rows": len(standard_aggregate),
            "missing_paths": 0,
        }
    )

    (output_dir / "manifest.json").write_text(json.dumps(summaries, indent=2) + "\n")
    write_readme(output_dir)
    print(pd.DataFrame(summaries).to_string(index=False))


if __name__ == "__main__":
    main()
