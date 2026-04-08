# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Download SWE-bench traces for the top finetunes of a given base model from the OT Agent leaderboard.

Downloads parquet trace datasets from HuggingFace to GCS, then converts to jsonl.gz.

Usage:
    uv run lib/marin/src/marin/run/ray_run.py -- python experiments/agent_scaling/download_ot_traces.py
"""

import argparse
import json
import os
import sys
from dataclasses import dataclass

import fsspec
import pandas as pd
import requests

import levanter.utils.fsspec_utils as fsspec_utils
from iris.marin_fs import marin_prefix
from marin.download.huggingface.download_hf import DownloadConfig, download_hf
from experiments.models import ModelConfig, download_model_step
from marin.utils import fsspec_glob, get_directory_friendly_name
from marin.execution.executor import (
    ExecutorStep,
    executor_main,
    output_path_of,
    this_output_path,
    versioned,
)

LEADERBOARD_URL = "https://ot-agent-leaderboard.replit.app/api/leaderboard-pivoted-with-improvement"
BENCHMARK = "swebench-verified-random-100-folders"
BASE_MODEL = "Qwen/Qwen3-8B"
K = 25

LEADERBOARD_CACHE = os.path.join("agent_scaling", "leaderboard_cache.json")


def get_top_k_runs(k: int, leaderboard_url: str, benchmark: str, base_model: str, refresh: bool = False) -> list[dict]:
    full_cache_path = os.path.join(marin_prefix(), LEADERBOARD_CACHE)
    if not refresh and fsspec_utils.exists(full_cache_path):
        with fsspec.open(full_cache_path, "rt") as f:
            data = json.load(f)
    else:
        resp = requests.get(leaderboard_url)
        resp.raise_for_status()
        data = resp.json()

        with fsspec.open(full_cache_path, "wt") as f:
            json.dump(data, f, indent=2)

    results = []
    for entry in data:
        if entry.get("canonicalBaseModelName") != base_model:
            continue
        b = entry.get("benchmarks", {}).get(benchmark)
        if b and b.get("accuracy") is not None:
            traces_url = b.get("hfTracesLink", "")
            dataset_id = traces_url.replace("https://huggingface.co/datasets/", "")
            results.append(
                {
                    "model": entry["canonicalModelName"],
                    "accuracy": b["accuracy"],
                    "dataset_id": dataset_id,
                }
            )

    results.sort(key=lambda x: x["accuracy"], reverse=True)
    return results[:k]


@dataclass(frozen=True)
class ConvertTracesConfig:
    input_path: str
    output_path: str


def convert_parquet_to_jsonl(cfg: ConvertTracesConfig):
    parquet_files = fsspec_glob(os.path.join(cfg.input_path, "**/*.parquet"))
    with fsspec.open(os.path.join(cfg.output_path, "traces.jsonl.gz"), "wt", compression="gzip") as f:
        for pf in parquet_files:
            df = pd.read_parquet(pf)
            for _, row in df.iterrows():
                record = {}
                for col in df.columns:
                    val = row[col]
                    if hasattr(val, "tolist"):
                        val = val.tolist()
                    record[col] = val
                f.write(json.dumps(record) + "\n")


# --- Build steps ---


def build_steps(k: int = K, base_model: str = BASE_MODEL) -> dict[str, dict[str, ExecutorStep]]:
    runs = get_top_k_runs(
        k=k,
        leaderboard_url=LEADERBOARD_URL,
        benchmark=BENCHMARK,
        base_model=base_model,
    )

    steps: dict[str, dict[str, ExecutorStep]] = {}

    for run in runs:
        name = get_directory_friendly_name(run["model"])

        model_step = download_model_step(
            ModelConfig(
                hf_repo_id=run["model"],
                hf_revision="main",
            )
        )

        trace_dl_step = ExecutorStep(
            name=f"raw/ot_traces/{name}",
            fn=download_hf,
            config=DownloadConfig(
                hf_dataset_id=versioned(run["dataset_id"]),
                revision=versioned("main"),
                gcs_output_path=this_output_path(),
                wait_for_completion=True,
            ),
        )

        convert_step = ExecutorStep(
            name=f"ot_traces/{name}",
            fn=convert_parquet_to_jsonl,
            config=ConvertTracesConfig(
                input_path=output_path_of(trace_dl_step),
                output_path=this_output_path(),
            ),
        )

        steps[name] = {"model": model_step, "traces": convert_step}

    return steps


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download OT Agent SWE-bench traces.")
    parser.add_argument("--top-k", type=int, default=K)
    parser.add_argument("--base-model", type=str, default=BASE_MODEL)
    args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0], *remaining]

    steps_dict = build_steps(k=args.top_k, base_model=args.base_model)
    all_steps = [steps_dict[name][step_type] for name in steps_dict for step_type in ["model", "traces"]]
    executor_main(steps=all_steps, description="Download OT Agent SWE-bench traces.")
