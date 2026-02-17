# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""
Compile and aggregate results from multiple Evalchemy evaluation runs.

This module provides the compile step function for aggregating results across
seeds and logging averaged metrics to wandb. It is used as the `fn` argument
to an `ExecutorStep` created by `compile_evalchemy_results` in `evals.py`.
"""

import json
import logging
import os
import re
import traceback

import fsspec
import pandas as pd

from marin.evaluation.evaluation_config import WANDB_PROJECT

logger = logging.getLogger(__name__)


def _extract_base_model_and_seed(model_name: str) -> tuple[str, int | None]:
    """Extract base model name and seed from model_name like 'model_task_seed42'."""
    match = re.search(r"[-_]seed(\d+)(?=[-_]|$)", model_name)
    if match:
        seed = int(match.group(1))
        base_model = model_name[: match.start()]
        return base_model, seed
    return model_name, None


def _extract_correctness(example: dict) -> float:
    """Extract correctness from an evalchemy example, handling different result schemas.

    Evalchemy benchmarks use different schemas for storing per-example results:
    - Code tasks (CodeForces, LiveCodeBench, CodeElo): boolean "correctness" field
    - JEEBench: per-repetition "score" list (can be fractional for partial credit)
    - Math tasks (AIME24/25, AMC23, HMMT, etc.): "expected_answer"/"answer" + "model_answers"
      string matching
    - HLE, GPQA Diamond: "expected_answer" + "model_answers" string matching
    """
    # 1. Boolean correctness field (code tasks: CodeForces, LiveCodeBench, CodeElo)
    if "correctness" in example:
        return 1 if example["correctness"] else 0

    # 2. Score field (JEEBench) — list of per-repetition scores, possibly fractional
    if "score" in example:
        score = example["score"]
        if isinstance(score, list):
            return sum(score) / len(score) if score else 0
        return float(score)

    # 3. String matching for answer fields (math tasks, HLE, GPQA Diamond)
    # Some tasks (e.g. AIME24) use "expected_answer", some (e.g. AIME25/AMC23) use "answer"
    expected = str(example.get("answer", example.get("expected_answer", ""))).strip()
    model_answers = example.get("model_answers", [])
    model_answer = str(model_answers[0]).strip() if model_answers else ""
    return 1 if (model_answer == expected and expected) else 0


def _load_results_from_input_paths(
    input_paths: list[str],
    fs: fsspec.AbstractFileSystem,
) -> list[dict]:
    """Load per-example results from GCS input paths.

    Raises:
        RuntimeError: If any result files fail to parse as JSON.
        ValueError: If no results are found across all input paths.
    """
    all_results: list[dict] = []
    parse_errors: list[str] = []

    for input_path in input_paths:
        base_dir = input_path
        if base_dir.endswith("results.json"):
            base_dir = base_dir.rsplit("/", 1)[0]

        logger.info(f"Loading evalchemy samples from root {base_dir}")

        if base_dir.startswith("gs://"):
            gcs_root = base_dir
        else:
            gcs_root = "gs://" + base_dir.lstrip("/")

        pattern = gcs_root.rstrip("/") + "/*/*/results_*.json"
        result_files = fs.glob(pattern)

        if not result_files:
            logger.warning(f"No results_*.json files found for input root {base_dir}")
            continue

        for result_file in result_files:
            logger.info(f"Reading results from {result_file}")
            path_parts = result_file.split("/")

            # Infer dataset_name from the task directory
            if len(path_parts) >= 3:
                task_dir = path_parts[-3]
                if "_" in task_dir:
                    dataset_name = task_dir.rsplit("_", 1)[0]
                else:
                    dataset_name = task_dir
            else:
                dataset_name = "unknown_dataset"

            # Infer model_name from directory structure
            if len(path_parts) >= 4:
                model_dir = path_parts[-4]
            elif len(path_parts) >= 2:
                model_dir = path_parts[-2]
            else:
                model_dir = "unknown_model"

            # Strip hash suffix from model_dir
            if "-" in model_dir:
                model_name = model_dir.rsplit("-", 1)[0]
            else:
                model_name = model_dir

            try:
                with fs.open(result_file, "r") as f:
                    data = json.load(f)
            except Exception as e:
                parse_errors.append(f"{result_file}: {e}")
                continue

            for _task_name, task_data in data.get("results", {}).items():
                for example in task_data.get("examples", []):
                    correct = _extract_correctness(example)

                    record = {
                        "id": example.get("id"),
                        "correct": correct,
                        "dataset_name": dataset_name.lower(),
                        "model_name": model_name.lower(),
                    }
                    all_results.append(record)

    if parse_errors:
        raise RuntimeError(f"Failed to parse {len(parse_errors)} result file(s):\n" + "\n".join(parse_errors))

    if not all_results:
        raise ValueError("No results found in any of the provided steps")

    return all_results


def _compute_averaged_results(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]] | None:
    """Compute per-seed accuracy averages grouped by base model and dataset.

    Returns (avg_df, accuracy_cols) or None if grouping columns are missing.
    """
    accuracy_cols = [col for col in df.columns if col in ["exact_match", "acc", "accuracy", "correct"]]

    if not accuracy_cols or "base_model_name" not in df.columns or "dataset_name" not in df.columns:
        logger.warning("Could not compute averaged results: missing accuracy columns or grouping columns")
        return None

    avg_results = []
    for (base_model, dataset), group in df.groupby(["base_model_name", "dataset_name"]):
        per_seed_accuracies = {}
        for col in accuracy_cols:
            if col in group.columns:
                seed_accs = group.groupby("seed")[col].mean()
                per_seed_accuracies[col] = seed_accs

        result = {
            "base_model_name": base_model,
            "dataset_name": dataset,
            "num_seeds": group["seed"].nunique(),
            "seeds": sorted(group["seed"].dropna().unique().tolist()),
        }

        for col in accuracy_cols:
            if col in per_seed_accuracies:
                seed_accs = per_seed_accuracies[col]
                result[f"{col}_mean"] = seed_accs.mean()
                result[f"{col}_std"] = seed_accs.std()
                result[f"{col}_per_seed"] = seed_accs.to_dict()

        avg_results.append(result)

    return pd.DataFrame(avg_results), accuracy_cols


def _log_averaged_results_to_wandb(
    avg_df: pd.DataFrame,
    accuracy_cols: list[str],
    seeds_config: list[int],
    base_eval_run_name: str | None,
    config_model_path: str | None,
    config_task_name: str | None,
) -> None:
    """Log averaged results to wandb, one run per base model."""
    import wandb

    num_seeds = len(seeds_config) if seeds_config else avg_df["num_seeds"].max()
    wandb_entity = os.environ.get("WANDB_ENTITY", "marin-community")

    step_suffix = ""
    if config_model_path:
        step_match = re.search(r"step-(\d+)", config_model_path)
        if step_match:
            step_suffix = f"-step{step_match.group(1)}"

    for base_model in avg_df["base_model_name"].unique():
        model_df = avg_df[avg_df["base_model_name"] == base_model]

        if config_task_name:
            dataset_suffix = config_task_name
        else:
            datasets = model_df["dataset_name"].unique()
            dataset_suffix = "-".join(sorted(datasets))

        if base_eval_run_name:
            wandb_run_name = f"evalchemy-{base_eval_run_name}{step_suffix}" f"-{dataset_suffix}-avg{num_seeds}seeds"
        else:
            if config_model_path:
                model_id = config_model_path.rstrip("/").split("/")[-1]
            else:
                model_id = base_model.lower()
            wandb_run_name = f"evalchemy-{model_id}{step_suffix}" f"-{dataset_suffix}-avg{num_seeds}seeds"

        wandb.init(
            project=WANDB_PROJECT,
            entity=wandb_entity,
            name=wandb_run_name,
            job_type="eval",
            tags=["evalchemy", "averaged-results", base_model.lower()[:64]],
            config={
                "base_model_name": base_model,
                "num_seeds": num_seeds,
                "seeds": seeds_config,
            },
            reinit=True,
        )

        for _, row in model_df.iterrows():
            dataset = row["dataset_name"]
            for col in accuracy_cols:
                mean_col = f"{col}_mean"
                std_col = f"{col}_std"
                if mean_col in row and std_col in row:
                    wandb.log(
                        {
                            f"{dataset}/{col}_mean": row[mean_col],
                            f"{dataset}/{col}_std": row[std_col],
                        }
                    )

        wandb.log({"averaged_results": wandb.Table(dataframe=model_df)})
        wandb.finish()
        logger.info(f"Averaged results for {base_model} logged to wandb as '{wandb_run_name}'")


def compile_evalchemy_results_fn(config: dict) -> None:
    """Top-level function executed by the ExecutorStep to compile evalchemy results.

    Reads individual per-seed evaluation results, aggregates them into compiled
    and averaged DataFrames, saves outputs to GCS, and logs to wandb.

    Config keys:
        input_paths: List of GCS paths to per-seed result directories.
        output_path: Where to write compiled outputs.
        seeds: List of seed values used.
        base_eval_run_name: Custom base name for wandb runs.
        model_path: Model checkpoint path (for wandb naming).
        task_name: Task name (for wandb naming).
    """
    input_paths = config["input_paths"]
    output_path = config["output_path"]
    seeds_config = config.get("seeds", [])
    base_eval_run_name = config.get("base_eval_run_name")
    config_model_path = config.get("model_path")

    logger.info(f"Compiling evalchemy results from {len(input_paths)} input paths")

    if not input_paths:
        raise ValueError("No input paths found!")

    fs = fsspec.filesystem("gcs")

    all_results = _load_results_from_input_paths(input_paths, fs)

    df = pd.DataFrame(all_results)

    df[["base_model_name", "seed"]] = df["model_name"].apply(lambda x: pd.Series(_extract_base_model_and_seed(x)))

    # Save compiled results
    results_file = f"{output_path}/compiled_results.json"
    with fsspec.open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)

    csv_file = f"{output_path}/compiled_results.csv"
    with fsspec.open(csv_file, "w") as f:
        df.to_csv(f, index=False)

    logger.info(f"Compiled results saved to: {results_file}")

    # Compute averaged results across seeds
    averaged = _compute_averaged_results(df)
    if averaged is None:
        return

    avg_df, accuracy_cols = averaged

    # Save averaged results
    avg_results_file = f"{output_path}/averaged_results.json"
    with fsspec.open(avg_results_file, "w") as f:
        json.dump(avg_df.to_dict(orient="records"), f, indent=2)

    avg_csv_file = f"{output_path}/averaged_results.csv"
    with fsspec.open(avg_csv_file, "w") as f:
        avg_df.to_csv(f, index=False)

    logger.info(f"Averaged results saved to: {avg_results_file}")
    logger.info(f"Averaged results:\n{avg_df.to_string()}")

    # Log averaged results to wandb (best-effort)
    try:
        _log_averaged_results_to_wandb(
            avg_df=avg_df,
            accuracy_cols=accuracy_cols,
            seeds_config=seeds_config,
            base_eval_run_name=base_eval_run_name,
            config_model_path=config_model_path,
            config_task_name=config.get("task_name"),
        )
    except Exception:
        logger.warning(f"Failed to log averaged results to wandb:\n{traceback.format_exc()}")
