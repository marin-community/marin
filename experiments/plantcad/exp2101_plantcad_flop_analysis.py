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

"""Analyze the relationship between actual FLOPs and model size.

Computes flops_per_token analytically via Levanter's lm_flops_per_token for the
Qwen3 architecture grid used in the isoflop sweep, and compares to the C=6ND
approximation. Also validates against empirical data from the isoflop CSV.
"""

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from levanter.layers.rotary import Llama3RotaryEmbeddingsConfig
from levanter.models.qwen import Qwen3Config
from levanter.utils.flop_utils import lm_flops_per_token

from experiments.llama import compute_num_parameters

EXPORT_DPI = 300

# Architecture constants (matching exp2101_plantcad_multi_isoflop_sweep.py)
MLP_RATIO = 4
SEQ_LEN = 4096
VOCAB_SIZE = 7
BASE_HIDDEN_LAYER_RATIO = 64
HIDDEN_HEAD_RATIO = 128
MIN_HIDDEN_POW = 8
MAX_HIDDEN_POW = 16
STEP_SIZE = 128


def compute_architecture_flops_grid() -> pd.DataFrame:
    """Compute flops_per_token and param count for each hidden size on the architecture grid."""
    results = []

    for hidden in range(2**MIN_HIDDEN_POW, 2**MAX_HIDDEN_POW + 1, STEP_SIZE):
        hs_pow = math.log2(hidden)
        intermediate = hidden * MLP_RATIO
        num_layers = round(hidden / (BASE_HIDDEN_LAYER_RATIO + (hs_pow * 4) - MIN_HIDDEN_POW))
        n_heads = max(1, hidden // HIDDEN_HEAD_RATIO)

        model_cfg = Qwen3Config(
            max_seq_len=SEQ_LEN,
            hidden_dim=hidden,
            intermediate_dim=intermediate,
            num_heads=n_heads,
            num_kv_heads=n_heads,
            num_layers=num_layers,
            rope=Llama3RotaryEmbeddingsConfig(),
        )
        params = compute_num_parameters(model_cfg, VOCAB_SIZE)
        flops = lm_flops_per_token(hidden, intermediate, num_layers, n_heads, n_heads, SEQ_LEN, VOCAB_SIZE, glu=True)
        results.append(
            {
                "hidden_size": hidden,
                "num_layers": num_layers,
                "params": params,
                "flops_per_token": flops,
            }
        )

    df = pd.DataFrame(results)
    # lm_flops_per_token is forward-pass only; multiply by 3 for full training (fwd + bwd)
    df["flops_per_token"] = df["flops_per_token"] * 3
    # k(N) = C / (N * D) = flops_per_token / params (since C = flops_per_token * D)
    df["k"] = df["flops_per_token"] / df["params"]
    df["log10_params"] = np.log10(df["params"])
    df["log10_flops_per_token"] = np.log10(df["flops_per_token"])
    df["log10_k"] = np.log10(df["k"])
    return df


def load_empirical_data(result_version: str) -> pd.DataFrame:
    """Load empirical isoflop data for validation overlay."""
    csv_path = Path(f"experiments/plantcad/results/v{result_version}/plantcad_isoflops.csv")
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    df = df[df["state"] == "finished"].copy()
    df["total_flops"] = df["total_gflops"] * 1e9
    df["empirical_k"] = df["total_flops"] / (df["params"] * df["total_tokens"])
    # Deduplicate: k is the same for all epochs at a given (params, tokens, flops_budget)
    df = df.drop_duplicates(subset=["params", "total_tokens", "flops_budget"])
    df["log10_params"] = np.log10(df["params"])
    df["log10_empirical_k"] = np.log10(df["empirical_k"])
    return df


def _format_param_count(x: float, _pos=None) -> str:
    """Format a log10(params) tick as a human-readable param count."""
    v = 10**x
    if v >= 1e12:
        return f"{v / 1e12:.0f}T"
    if v >= 1e9:
        return f"{v / 1e9:.0f}B"
    if v >= 1e6:
        return f"{v / 1e6:.0f}M"
    if v >= 1e3:
        return f"{v / 1e3:.0f}K"
    return f"{v:.0f}"


def _format_pow10(x: float, _pos=None) -> str:
    """Format a log10 tick as 10^x using LaTeX superscript."""
    exp = int(round(x))
    return f"$10^{{{exp}}}$"


def _set_log10_ticks(ax, axis: str, formatter, values: np.ndarray) -> None:
    """Set integer log10 ticks spanning the data range with a custom formatter."""
    lo = int(np.floor(values.min()))
    hi = int(np.ceil(values.max()))
    ticks = list(range(lo, hi + 1))
    if axis == "x":
        ax.set_xticks(ticks)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(formatter))
    else:
        ax.set_yticks(ticks)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(formatter))


def visualize_flops_per_token_vs_params(
    df_grid: pd.DataFrame,
    df_empirical: pd.DataFrame | None,
    output_dir: Path,
) -> None:
    """Plot k(N) vs params (main) and flops_per_token vs params (secondary)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), gridspec_kw={"width_ratios": [2, 1]})
    ax_k, ax_flops = axes

    # --- Left (2/3 width): k(N) = flops_per_token / N vs params ---

    # Fit k(N) = k_inf + c * N^gamma (3-parameter model with asymptote)
    def _k_model(N, k_inf, c, gamma):
        return k_inf + c * N**gamma

    popt, _ = curve_fit(
        _k_model,
        df_grid["params"].values,
        df_grid["k"].values,
        p0=[6.0, 1e3, -0.5],
        maxfev=10000,
    )
    k_inf, c, gamma = popt
    k_fit = _k_model(df_grid["params"].values, *popt)

    ax_k.plot(
        df_grid["log10_params"],
        df_grid["k"],
        "o-",
        color="seagreen",
        markersize=5,
        linewidth=3,
        label="Analytical (Levanter)",
    )
    ax_k.plot(
        df_grid["log10_params"],
        k_fit,
        "-",
        color="black",
        linewidth=1.5,
        zorder=5,
        label=f"Fit: $k(N) = {k_inf:.2f} + {c:.1f} \\cdot N^{{{gamma:.4f}}}$",
    )
    if df_empirical is not None:
        ax_k.scatter(
            df_empirical["log10_params"],
            df_empirical["empirical_k"],
            color="orange",
            alpha=0.8,
            s=100,
            edgecolors="black",
            linewidths=0.5,
            zorder=6,
            label="Empirical (isoflop runs)",
        )
    ax_k.axhline(6, color="red", linestyle="--", linewidth=1.5, label="k = 6 (C=6ND)")
    ax_k.set_xlabel("Params")
    ax_k.set_ylabel("k = C/(N·D)")
    ax_k.set_title("k(N) = C/(N·D) vs Model Size")
    ax_k.legend(fontsize=10)
    ax_k.grid(True, alpha=0.3)
    _set_log10_ticks(ax_k, "x", _format_param_count, df_grid["log10_params"])

    # --- Right (1/3 width): flops_per_token vs params (limited to 100B) ---
    df_flops = df_grid[df_grid["params"] <= 1e11]
    ax_flops.plot(
        df_flops["log10_params"],
        df_flops["log10_flops_per_token"],
        "o-",
        color="seagreen",
        markersize=5,
        linewidth=1.5,
        label="Analytical (Levanter)",
    )
    ax_flops.plot(
        df_flops["log10_params"],
        np.log10(6) + df_flops["log10_params"],
        "--",
        color="red",
        linewidth=1.5,
        label="6N (C=6ND approx)",
    )
    ax_flops.set_xlabel("Params")
    ax_flops.set_ylabel("FLOPs per Token")
    ax_flops.set_title("FLOPs per Token vs Model Size")
    ax_flops.legend(fontsize=10)
    ax_flops.grid(True, alpha=0.3)
    _set_log10_ticks(ax_flops, "x", _format_param_count, df_flops["log10_params"])
    _set_log10_ticks(ax_flops, "y", _format_pow10, df_flops["log10_flops_per_token"])

    plt.tight_layout()
    base_name = "plantcad_flops_by_params_approx"
    output_path = output_dir / f"{base_name}.png"
    fig.savefig(output_path, dpi=EXPORT_DPI, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), dpi=EXPORT_DPI, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")

    # Export fit coefficients
    json_path = output_dir / f"{base_name}.json"
    fit_data = {
        "model": "k(N) = k_inf + c * N^gamma",
        "description": (
            "FLOPs per token = k(N) * N, where N = param count. "
            "Fitted on analytical FLOPs from 3x lm_flops_per_token (Levanter)."
        ),
        "k_inf": k_inf,
        "c": c,
        "gamma": gamma,
    }
    with open(json_path, "w") as f:
        json.dump(fit_data, f, indent=2)
    print(f"Saved: {json_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze FLOPs vs model size relationship")
    parser.add_argument(
        "--result-version",
        type=str,
        default="1.15",
        help="Result version for empirical CSV overlay (default: 1.15)",
    )
    args = parser.parse_args()

    output_dir = Path(f"experiments/plantcad/results/v{args.result_version}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Compute analytical flops grid
    df_grid = compute_architecture_flops_grid()
    print(
        f"Computed flops for {len(df_grid)} model sizes "
        f"({df_grid['params'].min():.0f} to {df_grid['params'].max():.0f} params)"
    )

    # Load empirical data for overlay
    try:
        df_empirical = load_empirical_data(args.result_version)
        print(f"Loaded {len(df_empirical)} empirical data points")
    except FileNotFoundError:
        print("No empirical CSV found, skipping overlay")
        df_empirical = None

    visualize_flops_per_token_vs_params(df_grid, df_empirical, output_dir)


if __name__ == "__main__":
    main()
