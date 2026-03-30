#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

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

"""Pull DPO sweep data from W&B and create grouped plots."""

import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import wandb

OUT_DIR = Path(__file__).parent
DATA_DIR = OUT_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

# Metrics we want to plot, with display names
METRICS = {
    "train/dpo_loss": "DPO Train Loss",
    "eval/bloom_speceval_v2_val/dpo_loss": "DPO Eval Loss",
    "eval/bloom_speceval_v2_val/dpo_accuracy": "DPO Eval Accuracy",
    "train/dpo_accuracy": "DPO Train Accuracy",
    "train/dpo_margin_policy": "DPO Train Policy Margin",
    "eval/bloom_speceval_v2_val/dpo_margin_policy": "DPO Eval Policy Margin",
}

CONFIG_COLORS = {
    "β=0.01, lr=5e-7": "#1f77b4",
    "β=0.01, lr=7.5e-7": "#ff7f0e",
    "β=0.1, lr=5e-7": "#2ca02c",
    "β=0.1, lr=7.5e-7": "#d62728",
}

CONFIG_ORDER = list(CONFIG_COLORS.keys())


def parse_run_config(name: str) -> dict:
    """Extract beta, lr, seed from run name."""
    beta_m = re.search(r"beta([\d.]+)", name)
    seed_m = re.search(r"seed(\d+)", name)
    lr_m = re.search(r"lr([\d.e-]+)", name)
    return {
        "beta": beta_m.group(1) if beta_m else "?",
        "lr": lr_m.group(1) if lr_m else "5e-7",
        "seed": int(seed_m.group(1)) if seed_m else -1,
    }


def config_label(cfg: dict) -> str:
    return f"β={cfg['beta']}, lr={cfg['lr']}"


def pull_data():
    """Pull all history data from W&B and save to JSON."""
    api = wandb.Api()
    runs = api.runs("marin-community/dpo")

    all_data = {}
    for run in runs:
        if "bloom_speceval_v2" not in run.name:
            continue

        cfg = parse_run_config(run.name)
        label = config_label(cfg)
        print(f"Pulling {run.name} ({label}, seed={cfg['seed']})...")

        keys = [*list(METRICS.keys()), "_step"]
        hist = run.history(keys=keys, samples=5000)

        run_data = {
            "name": run.name,
            "config": cfg,
            "label": label,
            "state": run.state,
            "history": {},
        }

        for metric_key in METRICS:
            if metric_key in hist.columns:
                df = hist[["_step", metric_key]].dropna()
                run_data["history"][metric_key] = {
                    "steps": df["_step"].tolist(),
                    "values": df[metric_key].tolist(),
                }

        all_data[run.name] = run_data

    # Save raw data
    out_path = DATA_DIR / "sweep_data.json"
    with open(out_path, "w") as f:
        json.dump(all_data, f, indent=2)
    print(f"\nSaved data for {len(all_data)} runs to {out_path}")
    return all_data


def load_data() -> dict:
    """Load cached data from JSON."""
    with open(DATA_DIR / "sweep_data.json") as f:
        return json.load(f)


def make_plots(all_data: dict):
    """Create one plot per metric, grouping by config with seed lines + mean."""
    # Group runs by config label
    groups: dict[str, list] = defaultdict(list)
    for run_data in all_data.values():
        groups[run_data["label"]].append(run_data)

    for metric_key, metric_name in METRICS.items():
        fig, ax = plt.subplots(figsize=(10, 6))

        for label in CONFIG_ORDER:
            if label not in groups:
                continue

            color = CONFIG_COLORS[label]
            seed_lines = []

            for run_data in sorted(groups[label], key=lambda r: r["config"]["seed"]):
                hist = run_data["history"].get(metric_key)
                if hist is None or len(hist["steps"]) == 0:
                    continue

                steps = np.array(hist["steps"])
                values = np.array(hist["values"])

                # Plot individual seed as thin, semi-transparent
                ax.plot(
                    steps,
                    values,
                    color=color,
                    alpha=0.25,
                    linewidth=0.8,
                )
                seed_lines.append((steps, values))

            # Compute and plot mean across seeds
            if seed_lines:
                # Interpolate all seeds onto a common step grid
                all_steps = sorted(set(s for steps, _ in seed_lines for s in steps))
                common_steps = np.array(all_steps)
                interpolated = []
                for steps, values in seed_lines:
                    interp_vals = np.interp(common_steps, steps, values)
                    interpolated.append(interp_vals)

                mean_vals = np.mean(interpolated, axis=0)
                std_vals = np.std(interpolated, axis=0)

                ax.plot(
                    common_steps,
                    mean_vals,
                    color=color,
                    linewidth=2.0,
                    label=label,
                )
                # Shade +/- 1 std
                ax.fill_between(
                    common_steps,
                    mean_vals - std_vals,
                    mean_vals + std_vals,
                    color=color,
                    alpha=0.12,
                )

        # Log scale for loss metrics
        if "loss" in metric_key.lower():
            ax.set_yscale("log")

        ax.set_xlabel("Training Step", fontsize=12)
        ax.set_ylabel(metric_name, fontsize=12)
        ax.set_title(f"{metric_name} — DPO Sweep (Bloom SpecEval v2)", fontsize=13)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        # Save
        safe_name = metric_name.lower().replace(" ", "_")
        fig.savefig(OUT_DIR / f"{safe_name}.png", dpi=150)
        fig.savefig(OUT_DIR / f"{safe_name}.pdf")
        print(f"Saved {safe_name}.png / .pdf")
        plt.close(fig)

    # Also make a combined 2x3 figure
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes_flat = axes.flatten()

    for idx, (metric_key, metric_name) in enumerate(METRICS.items()):
        ax = axes_flat[idx]
        for label in CONFIG_ORDER:
            if label not in groups:
                continue
            color = CONFIG_COLORS[label]
            seed_lines = []
            for run_data in sorted(groups[label], key=lambda r: r["config"]["seed"]):
                hist = run_data["history"].get(metric_key)
                if hist is None or len(hist["steps"]) == 0:
                    continue
                steps = np.array(hist["steps"])
                values = np.array(hist["values"])
                ax.plot(steps, values, color=color, alpha=0.25, linewidth=0.8)
                seed_lines.append((steps, values))

            if seed_lines:
                all_steps = sorted(set(s for steps, _ in seed_lines for s in steps))
                common_steps = np.array(all_steps)
                interpolated = []
                for steps, values in seed_lines:
                    interp_vals = np.interp(common_steps, steps, values)
                    interpolated.append(interp_vals)
                mean_vals = np.mean(interpolated, axis=0)
                std_vals = np.std(interpolated, axis=0)
                ax.plot(common_steps, mean_vals, color=color, linewidth=2.0, label=label)
                ax.fill_between(
                    common_steps,
                    mean_vals - std_vals,
                    mean_vals + std_vals,
                    color=color,
                    alpha=0.12,
                )

        if "loss" in metric_key.lower():
            ax.set_yscale("log")

        ax.set_xlabel("Step", fontsize=10)
        ax.set_ylabel(metric_name, fontsize=10)
        ax.set_title(metric_name, fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=9)

    # Single legend for the whole figure
    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=11, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("DPO Sweep — Bloom SpecEval v2 (3 seeds per config)", fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "combined_all_metrics.png", dpi=150, bbox_inches="tight")
    fig.savefig(OUT_DIR / "combined_all_metrics.pdf", bbox_inches="tight")
    print("Saved combined_all_metrics.png / .pdf")
    plt.close(fig)


if __name__ == "__main__":
    import sys

    if "--plot-only" in sys.argv:
        data = load_data()
    else:
        data = pull_data()

    make_plots(data)
    print("\nDone!")
