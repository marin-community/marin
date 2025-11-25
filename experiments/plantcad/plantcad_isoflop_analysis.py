# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import wandb


def log_run_object(run, run_idx):
    """Log a run object as JSON to show available data."""
    print(f"\n{'=' * 80}")
    print(f"RUN {run_idx + 1}: {run.name}")
    print(f"{'=' * 80}")
    run_dict = {
        "id": run.id,
        "name": run.name,
        "state": run.state,
        "created_at": str(run.created_at),
        "tags": run.tags,
        "config": dict(run.config),
        "summary": dict(run.summary),
    }
    print(json.dumps(run_dict, indent=2, default=str))
    print(f"{'=' * 80}\n")


def fetch_plantcad_runs():
    """Fetch plantcad isoflop runs and extract metrics/tags into a dataframe."""
    api = wandb.Api()
    runs = api.runs(
        "marin",
        filters={"display_name": {"$regex": "^plantcad_isoflop_01"}},
    )

    data = []
    for idx, run in enumerate(runs):
        # Log first 2 runs in detail
        if idx < 2:
            log_run_object(run, idx)

        # Parse tags like "batch_size=32"
        tags_dict = {}
        for tag in run.tags:
            if "=" in tag:
                key, value = tag.split("=", 1)
                try:
                    # Try to convert to appropriate type
                    if "." in value or "e+" in value or "e-" in value:
                        tags_dict[key] = float(value)
                    else:
                        tags_dict[key] = int(value)
                except ValueError:
                    tags_dict[key] = value

        # Calculate execution time
        start_time = pd.to_datetime(run.created_at) if run.created_at else None
        stop_time = pd.to_datetime(run.summary.get("_timestamp"), unit="s") if run.summary.get("_timestamp") else None

        # Handle timezone differences
        if start_time and stop_time:
            if start_time.tzinfo and not stop_time.tzinfo:
                stop_time = stop_time.tz_localize("UTC")
            elif stop_time.tzinfo and not start_time.tzinfo:
                start_time = start_time.tz_localize("UTC")
            duration = (stop_time - start_time).total_seconds()
        else:
            duration = None

        row = {
            "run_name": run.name,
            "state": run.state,
            "start_time": start_time,
            "stop_time": stop_time,
            "duration_sec": duration,
            # Metrics
            "eval_loss": run.summary.get("eval/plantcad_cropped/loss"),
            "train_loss": run.summary.get("train/loss"),
            "total_gflops": run.summary.get("throughput/total_gflops"),
            "total_tokens": run.summary.get("throughput/total_tokens"),
            "run_progress": run.summary.get("run_progress"),
            # Tags
            "architecture": tags_dict.get("architecture"),
            "batch_size": tags_dict.get("batch_size"),
            "flops_budget": tags_dict.get("flops_budget"),
            "hidden_size": tags_dict.get("hidden_size"),
            "num_layers": tags_dict.get("num_layers"),
            "params": tags_dict.get("params"),
            "steps": tags_dict.get("steps"),
            "tokens": tags_dict.get("tokens"),
            "tpu": tags_dict.get("tpu"),
        }
        data.append(row)

    return pd.DataFrame(data)


def save_runs(df, output_path="experiments/plantcad/results/plantcad_isoflops.csv"):
    """Save dataframe to CSV file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved {len(df)} runs to {output_path}")


def summarize_runs(df):
    """Print full dataframe and simplified summary table."""
    print("\n" + "=" * 80)
    print("FULL DATAFRAME")
    print("=" * 80)
    print(df.to_string())

    print("\n" + "=" * 80)
    print("SIMPLIFIED SUMMARY")
    print("=" * 80)
    summary = df[["run_name", "state", "flops_budget", "architecture", "run_progress"]].copy()
    print(summary.to_string())


def visualize_loss_by_token_count(
    df, metric="eval_loss", output_path="experiments/plantcad/results/plantcad_loss_by_tokens.png"
):
    """Plot loss vs tokens, grouped by budget and faceted by architecture."""
    df_clean = df[df["state"] == "finished"].dropna(subset=[metric, "tokens", "architecture", "flops_budget"])

    architectures = sorted(df_clean["architecture"].unique())
    budgets = sorted(df_clean["flops_budget"].unique())

    # Get global y-limits for consistent scales
    y_min, y_max = df_clean[metric].min(), df_clean[metric].max()
    y_padding = (y_max - y_min) * 0.1

    fig, axes = plt.subplots(1, len(architectures), figsize=(5 * len(architectures), 5), squeeze=False)

    for idx, arch in enumerate(architectures):
        ax = axes[0, idx]
        for budget in budgets:
            data = df_clean[(df_clean["architecture"] == arch) & (df_clean["flops_budget"] == budget)].sort_values(
                "tokens"
            )
            ax.plot(data["tokens"], data[metric], alpha=0.5, linewidth=1)
            ax.scatter(data["tokens"], data[metric], label=f"{budget:.1e}", alpha=0.7)
        ax.set_xlabel("Token Count")
        ax.set_ylabel("Validation Loss")
        ax.set_title(f"Architecture = {arch}")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylim(y_min - y_padding, y_max + y_padding)
        ax.grid(alpha=0.3)

    # Single legend to the right of all plots
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, title="FLOPS", loc="center left", bbox_to_anchor=(1, 0.5))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved plot to {output_path}")


def visualize_loss_by_param_count(
    df, metric="eval_loss", output_path="experiments/plantcad/results/plantcad_loss_by_params.png"
):
    """Plot loss vs params, grouped by budget and faceted by architecture."""
    df_clean = df[df["state"] == "finished"].dropna(subset=[metric, "params", "architecture", "flops_budget"])

    architectures = sorted(df_clean["architecture"].unique())
    budgets = sorted(df_clean["flops_budget"].unique())

    # Get global y-limits for consistent scales
    y_min, y_max = df_clean[metric].min(), df_clean[metric].max()
    y_padding = (y_max - y_min) * 0.1

    fig, axes = plt.subplots(1, len(architectures), figsize=(5 * len(architectures), 5), squeeze=False)

    for idx, arch in enumerate(architectures):
        ax = axes[0, idx]
        for budget in budgets:
            data = df_clean[(df_clean["architecture"] == arch) & (df_clean["flops_budget"] == budget)].sort_values(
                "params"
            )
            ax.plot(data["params"], data[metric], alpha=0.5, linewidth=1)
            ax.scatter(data["params"], data[metric], label=f"{budget:.1e}", alpha=0.7)
        ax.set_xlabel("Param Count")
        ax.set_ylabel("Validation Loss")
        ax.set_title(f"Architecture = {arch}")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylim(y_min - y_padding, y_max + y_padding)
        ax.grid(alpha=0.3)

    # Single legend to the right of all plots
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, title="FLOPS", loc="center left", bbox_to_anchor=(1, 0.5))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved plot to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch and analyze plantcad isoflop runs")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force refetch from W&B even if CSV exists",
    )
    parser.add_argument(
        "--output",
        default="experiments/plantcad/results/plantcad_isoflops.csv",
        help="Output CSV path (default: experiments/plantcad/results/plantcad_isoflops.csv)",
    )
    args = parser.parse_args()

    output_path = Path(args.output)

    # Check if CSV exists and load from it unless --force is specified
    if output_path.exists() and not args.force:
        print(f"Loading existing data from {output_path}")
        df = pd.read_csv(output_path)
        print(f"Loaded {len(df)} runs from CSV")
    else:
        print("Fetching runs from W&B...")
        df = fetch_plantcad_runs()
        save_runs(df, output_path)

    summarize_runs(df)
    visualize_loss_by_token_count(df)
    visualize_loss_by_param_count(df)
