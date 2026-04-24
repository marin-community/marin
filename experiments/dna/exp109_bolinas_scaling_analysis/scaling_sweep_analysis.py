# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compare exp109 scaling sweep lm_eval results against the TraitGym reference (exp55/58/59).

Pulls W&B runs matching `dna-bolinas-scaling-v0.5-*`, takes the max over training steps for
each lm_eval/AUPRC metric, normalizes the metric names to match the reference pivot
convention (single string — e.g. `global`, `missense_variant`), and merges with the
reference data from the gist at
https://gist.github.com/eric-czech/787e7ab1a0e0be87bfecc7bce1fa8e83 (see
bolinas-dna#109: https://github.com/Open-Athena/bolinas-dna/issues/109).

Reference pivot reproduces the exact aggregation from the gist's analysis.md:
`score_type=minus_llr`, per-group final `step`, pivoted to one row per
(experiment, dataset) x subset, then MAX rolled up per experiment and overall.

Composite metrics:
  composite1 = mean(missense, tss_proximal, 5_prime_UTR, 3_prime_UTR, splicing, synonymous)
  composite2 = composite1 without synonymous_variant

Visualization: 1x2 (bar chart of composites, heatmap of composite1 constituents) covering
the reference per-experiment MAX rows, the overall MAX (exp{55,58,59}), and exp109's
4 largest models (by param count).

Usage:
    uv run --with pandas --with pyarrow --with matplotlib --with wandb \
        experiments/dna/exp109_bolinas_scaling_analysis/scaling_sweep_analysis.py [--refresh]

Caches wandb data and the reference parquet locally; pass --refresh to re-fetch.
"""

import json
import re
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

VERSION = "v0.5"
EXP109_NAME = "exp109"
AGGREGATE_EXP_NAME = "exp{55,58,59}"

WANDB_PROJECT = "eric-czech/marin"
WANDB_RUN_PREFIX = f"dna-bolinas-scaling-{VERSION}-"

REFERENCE_PARQUET_URL = (
    "https://gist.githubusercontent.com/eric-czech/787e7ab1a0e0be87bfecc7bce1fa8e83/"
    "raw/a017bc68afd2a1ec78f8bb0074e9eb1acb6b274e/results.parquet"
)

CACHE_DIR = Path("/tmp")
WANDB_CACHE_PATH = CACHE_DIR / f"scaling_sweep_{VERSION}_lm_eval.json"
REFERENCE_CACHE_PATH = CACHE_DIR / "exp55_58_59_results.parquet"

RESULTS_DIR = Path(f"experiments/dna/exp109_bolinas_scaling_analysis/results/scaling/{VERSION}")
MERGED_CSV_PATH = RESULTS_DIR / "merged_lm_eval.csv"
PLOT_PATH = RESULTS_DIR / "scaling_sweep_vs_reference.png"

# Traitgym subsets in the reference pivot (same ordering as the gist, with `global` first).
REFERENCE_SUBSETS = (
    "global",
    "3_prime_UTR_variant",
    "5_prime_UTR_variant",
    "distal",
    "missense_variant",
    "non_coding_transcript_exon_variant",
    "splicing",
    "synonymous_variant",
    "tss_proximal",
)

COMPOSITE1_CONSTITUENTS = (
    "missense_variant",
    "tss_proximal",
    "5_prime_UTR_variant",
    "3_prime_UTR_variant",
    "splicing",
    "synonymous_variant",
)
COMPOSITE2_CONSTITUENTS = tuple(m for m in COMPOSITE1_CONSTITUENTS if m != "synonymous_variant")

# W&B lm_eval keys pulled from scan_history; enumerated here so history iteration is scoped.
WANDB_METRIC_KEYS = (
    "lm_eval/traitgym_mendelian_v2_255/auprc",
    "lm_eval/traitgym_mendelian_v2_255/3_prime_UTR_variant/auprc",
    "lm_eval/traitgym_mendelian_v2_255/5_prime_UTR_variant/auprc",
    "lm_eval/traitgym_mendelian_v2_255/distal/auprc",
    "lm_eval/traitgym_mendelian_v2_255/missense_variant/auprc",
    "lm_eval/traitgym_mendelian_v2_255/non_coding_transcript_exon_variant/auprc",
    "lm_eval/traitgym_mendelian_v2_255/splicing/auprc",
    "lm_eval/traitgym_mendelian_v2_255/synonymous_variant/auprc",
    "lm_eval/traitgym_mendelian_v2_255/tss_proximal/auprc",
    "lm_eval/averages/macro_avg_auprc",
    "lm_eval/averages/micro_avg_auprc",
)

_TRAITGYM_SUBSET_RE = re.compile(r"^lm_eval/traitgym_mendelian_v2_255/(?P<subset>.+)/auprc$")


def _normalize_wandb_metric(key: str) -> str:
    """Map W&B lm_eval key to the flat reference naming convention.

    - `lm_eval/traitgym_mendelian_v2_255/auprc`             -> `global`
    - `lm_eval/traitgym_mendelian_v2_255/<subset>/auprc`    -> `<subset>`
    - `lm_eval/averages/macro_avg_auprc`                    -> `macro_avg_auprc`
    - `lm_eval/averages/micro_avg_auprc`                    -> `micro_avg_auprc`
    """
    if key == "lm_eval/traitgym_mendelian_v2_255/auprc":
        return "global"
    m = _TRAITGYM_SUBSET_RE.match(key)
    if m:
        return m.group("subset")
    if key.startswith("lm_eval/averages/"):
        return key.rsplit("/", 1)[-1]
    return key


_RUN_NAME_RE = re.compile(rf"^{re.escape(WANDB_RUN_PREFIX)}h(?P<hidden>\d+)-p(?P<params>[A-Za-z0-9]+)$")


def _parse_run_name(name: str) -> tuple[int, str]:
    m = _RUN_NAME_RE.match(name)
    if not m:
        raise ValueError(f"Run name does not match expected pattern: {name!r}")
    return int(m.group("hidden")), m.group("params")


# =============================================================================
# W&B fetch (exp109)
# =============================================================================


def fetch_wandb(project: str, run_prefix: str) -> list[dict]:
    import wandb

    api = wandb.Api(timeout=300)
    runs = list(api.runs(project, filters={"display_name": {"$regex": f"^{re.escape(run_prefix)}"}}))
    rows = []
    for r in runs:
        hidden, params = _parse_run_name(r.name)
        max_by_key: dict[str, float] = {}
        for row in r.scan_history(keys=["_step", *WANDB_METRIC_KEYS]):
            for key in WANDB_METRIC_KEYS:
                v = row.get(key)
                if v is None:
                    continue
                try:
                    v = float(v)
                except (TypeError, ValueError):
                    continue
                if not np.isfinite(v):
                    continue
                prev = max_by_key.get(key)
                if prev is None or v > prev:
                    max_by_key[key] = v
        rows.append(
            {
                "name": r.name,
                "state": r.state,
                "hidden_size": hidden,
                "param_label": params,
                "metrics": max_by_key,
            }
        )
    return rows


def load_wandb(refresh: bool = False) -> list[dict]:
    if not refresh and WANDB_CACHE_PATH.exists():
        with open(WANDB_CACHE_PATH) as f:
            rows = json.load(f)
        print(f"Loaded {len(rows)} wandb runs from cache ({WANDB_CACHE_PATH})")
        return rows
    rows = fetch_wandb(WANDB_PROJECT, WANDB_RUN_PREFIX)
    with open(WANDB_CACHE_PATH, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"Fetched {len(rows)} wandb runs, cached to {WANDB_CACHE_PATH}")
    return rows


# =============================================================================
# Reference parquet (exp55/58/59)
# =============================================================================


def _download_reference(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    # Use curl to avoid platform-specific urllib SSL cert issues.
    subprocess.run(["curl", "-sSLf", "-o", str(dest), url], check=True)


def load_reference(refresh: bool = False) -> pd.DataFrame:
    if refresh or not REFERENCE_CACHE_PATH.exists():
        _download_reference(REFERENCE_PARQUET_URL, REFERENCE_CACHE_PATH)
        print(f"Downloaded reference parquet to {REFERENCE_CACHE_PATH}")
    df = pd.read_parquet(REFERENCE_CACHE_PATH)
    print(f"Loaded reference parquet: {len(df)} rows")
    return df


def reference_pivot(df: pd.DataFrame) -> pd.DataFrame:
    """Reproduce analysis.md: filter to minus_llr @ per-group final step, pivot by subset.

    Primary key of input: (experiment, model, step, score_type, subset). We take the
    max step per (experiment, model, score_type, subset), filter to score_type=minus_llr,
    and pivot subsets into columns.
    """
    primary_key = ["experiment", "model", "step", "score_type", "subset"]
    assert not df.duplicated(subset=primary_key).any(), "reference primary key not unique"

    group_key = [c for c in primary_key if c != "step"]
    df = df.copy()
    df["final_step"] = df.groupby(group_key)["step"].transform("max")

    sub = df[(df["step"] == df["final_step"]) & (df["score_type"] == "minus_llr")].rename(columns={"model": "dataset"})
    assert not sub.duplicated(subset=["experiment", "dataset", "subset"]).any()

    pivot = sub.pivot(index=["experiment", "dataset"], columns="subset", values="value")
    cols = ["global"] + [c for c in pivot.columns if c != "global"]
    return pivot[cols]


def reference_long_with_rollups(pivot: pd.DataFrame) -> pd.DataFrame:
    """Return long-form reference rows: raw (experiment, dataset) plus MAX rollups.

    Rollup semantics match analysis.md:
      - per-experiment MAX across datasets  -> dataset="MAX"
      - overall MAX across experiment x dataset -> experiment=exp{55,58,59}, dataset="MAX"
    """
    raw = pivot.reset_index().assign(role="raw")

    by_exp = pivot.groupby(level="experiment").max().reset_index()
    by_exp["dataset"] = "MAX"
    by_exp["role"] = "per_experiment_max"

    overall = pivot.max().to_frame().T
    overall.insert(0, "experiment", AGGREGATE_EXP_NAME)
    overall.insert(1, "dataset", "MAX")
    overall["role"] = "overall_max"

    combined = pd.concat([raw, by_exp, overall], ignore_index=True, sort=False)
    long = combined.melt(
        id_vars=["experiment", "dataset", "role"],
        var_name="metric",
        value_name="value",
    )
    long["source"] = "reference"
    long["param_label"] = pd.NA
    long["hidden_size"] = pd.NA
    return long


# =============================================================================
# Long-form construction & merge
# =============================================================================


def exp109_long(rows: list[dict]) -> pd.DataFrame:
    """Flatten exp109 runs to long-form: one row per (run, metric)."""
    records = []
    for row in rows:
        for key, value in row["metrics"].items():
            records.append(
                {
                    "experiment": EXP109_NAME,
                    "dataset": "animals",
                    "role": "raw",
                    "metric": _normalize_wandb_metric(key),
                    "value": float(value),
                    "source": "wandb",
                    "param_label": row["param_label"],
                    "hidden_size": row["hidden_size"],
                }
            )
    return pd.DataFrame.from_records(records)


def merge_long(ref_long: pd.DataFrame, exp109_long_df: pd.DataFrame) -> pd.DataFrame:
    """Stack reference + exp109 rows; keep every metric (outer-join semantics).

    Metrics that appear on only one side (e.g. `macro_avg_auprc` is wandb-only) are
    preserved — they simply have no counterpart on the other side.
    """
    merged = pd.concat([ref_long, exp109_long_df], ignore_index=True, sort=False)
    ref_metrics = set(ref_long["metric"].unique())
    wandb_metrics = set(exp109_long_df["metric"].unique())
    ref_only = sorted(ref_metrics - wandb_metrics)
    wandb_only = sorted(wandb_metrics - ref_metrics)
    both = sorted(ref_metrics & wandb_metrics)
    print(f"Metrics in both:       {both}")
    print(f"Reference-only:        {ref_only}")
    print(f"W&B-only (unmerged):   {wandb_only}")
    return merged


def add_composites(long: pd.DataFrame) -> pd.DataFrame:
    """Compute composite1/composite2 per (experiment, dataset, role, source).

    A composite is only emitted for a group if every constituent metric is present with a
    finite value — partial aggregates would be misleading.
    """
    group_keys = ["experiment", "dataset", "role", "source", "param_label", "hidden_size"]
    wide = long.pivot_table(index=group_keys, columns="metric", values="value", dropna=False, aggfunc="first")
    wide = wide.reset_index()

    def _composite(cols: tuple[str, ...]) -> pd.Series:
        missing = [c for c in cols if c not in wide.columns]
        if missing:
            return pd.Series(np.nan, index=wide.index)
        sub = wide[list(cols)]
        finite = sub.notna().all(axis=1)
        return np.where(finite, sub.mean(axis=1), np.nan)

    wide["composite1"] = _composite(COMPOSITE1_CONSTITUENTS)
    wide["composite2"] = _composite(COMPOSITE2_CONSTITUENTS)

    value_cols = [c for c in wide.columns if c not in group_keys]
    long_with_composites = wide.melt(id_vars=group_keys, value_vars=value_cols, var_name="metric", value_name="value")
    long_with_composites = long_with_composites.dropna(subset=["value"])
    return long_with_composites


# =============================================================================
# Visualization
# =============================================================================


def _extract_params_numeric(label: str) -> float:
    """Convert a param label like '46M', '1B' into a numeric (tokens) value for sorting."""
    m = re.match(r"^(\d+(?:\.\d+)?)([MB])$", label)
    if not m:
        raise ValueError(f"Cannot parse param label: {label!r}")
    scale = 1e6 if m.group(2) == "M" else 1e9
    return float(m.group(1)) * scale


def select_viz_rows(long: pd.DataFrame) -> pd.DataFrame:
    """Pick the rows shown in the viz: reference per-experiment MAX + overall MAX + 4 largest exp109 models."""
    ref_rows = long[(long["source"] == "reference") & (long["role"].isin(["per_experiment_max", "overall_max"]))].copy()
    ref_rows["display_name"] = ref_rows["experiment"]
    ref_rows["sort_key"] = ref_rows["experiment"].map({"exp55": 0, "exp58": 1, "exp59": 2, AGGREGATE_EXP_NAME: 3})

    exp109_rows = long[long["experiment"] == EXP109_NAME].copy()
    param_labels = exp109_rows["param_label"].dropna().unique().tolist()
    sorted_labels = sorted(param_labels, key=_extract_params_numeric)
    largest_four = sorted_labels[-4:]
    print(f"exp109 param labels: {sorted_labels} -> viz selection (4 largest): {largest_four}")
    exp109_rows = exp109_rows[exp109_rows["param_label"].isin(largest_four)].copy()
    exp109_rows["display_name"] = EXP109_NAME + "-" + exp109_rows["param_label"]
    exp109_rows["sort_key"] = 10 + exp109_rows["param_label"].map({label: i for i, label in enumerate(largest_four)})

    viz = pd.concat([ref_rows, exp109_rows], ignore_index=True, sort=False)
    return viz


def plot_composites(viz: pd.DataFrame) -> None:
    """1x2 figure: bar chart of composites (left), heatmap of constituents (right)."""
    order = viz.sort_values("sort_key")["display_name"].drop_duplicates().tolist()

    comp_wide = (
        viz[viz["metric"].isin(["composite1", "composite2"])]
        .pivot_table(index="display_name", columns="metric", values="value", aggfunc="first")
        .reindex(order)
    )

    heatmap_wide = (
        viz[viz["metric"].isin(COMPOSITE1_CONSTITUENTS)]
        .pivot_table(index="display_name", columns="metric", values="value", aggfunc="first")
        .reindex(order)[list(COMPOSITE1_CONSTITUENTS)]
    )

    fig, (ax_bar, ax_heat) = plt.subplots(1, 2, figsize=(16, 6), gridspec_kw={"width_ratios": [1.1, 1.4]})

    x = np.arange(len(order))
    width = 0.38
    bars1 = ax_bar.bar(
        x - width / 2,
        comp_wide["composite1"].values,
        width,
        label="composite1 (6 subsets)",
        color="#4C72B0",
        edgecolor="k",
        linewidth=0.4,
    )
    bars2 = ax_bar.bar(
        x + width / 2,
        comp_wide["composite2"].values,
        width,
        label="composite2 (no synonymous)",
        color="#DD8452",
        edgecolor="k",
        linewidth=0.4,
    )
    for bars in (bars1, bars2):
        for bar in bars:
            h = bar.get_height()
            if np.isfinite(h):
                ax_bar.text(
                    bar.get_x() + bar.get_width() / 2, h + 0.003, f"{h:.3f}", ha="center", va="bottom", fontsize=7
                )
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(order, rotation=30, ha="right", fontsize=9)
    ax_bar.set_ylabel("AUPRC (mean of constituents)")
    ax_bar.set_title(f"Composite AUPRC — {EXP109_NAME} vs TraitGym reference")
    ax_bar.legend(fontsize=9, loc="upper left")
    ax_bar.grid(axis="y", alpha=0.3, linewidth=0.5)
    ymax = float(np.nanmax(comp_wide.values)) if comp_wide.size else 1.0
    ax_bar.set_ylim(0, ymax * 1.15)

    data = heatmap_wide.values.astype(float)
    im = ax_heat.imshow(data, aspect="auto", cmap="viridis", vmin=np.nanmin(data), vmax=np.nanmax(data))
    ax_heat.set_xticks(np.arange(heatmap_wide.shape[1]))
    ax_heat.set_xticklabels(heatmap_wide.columns, rotation=30, ha="right", fontsize=9)
    ax_heat.set_yticks(np.arange(heatmap_wide.shape[0]))
    ax_heat.set_yticklabels(heatmap_wide.index, fontsize=9)
    ax_heat.set_title("Composite1 constituents (AUPRC)")
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = data[i, j]
            if not np.isfinite(v):
                continue
            # pick readable text color based on cell luminance
            rgba = im.cmap(im.norm(v))
            lum = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
            ax_heat.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=8, color="white" if lum < 0.5 else "black")
    fig.colorbar(im, ax=ax_heat, shrink=0.85)

    fig.suptitle(
        f"Bolinas DNA scaling sweep ({VERSION}) — exp109 vs TraitGym reference (exp55/58/59)\n"
        "Reference rows: MAX across `dataset` per experiment; exp109 rows: 4 largest models",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOT_PATH, dpi=300, bbox_inches="tight")
    fig.savefig(PLOT_PATH.with_suffix(".pdf"), bbox_inches="tight")
    print(f"Saved plot to {PLOT_PATH}")


# =============================================================================
# Entry point
# =============================================================================


def main(refresh: bool = False) -> None:
    ref_raw = load_reference(refresh=refresh)
    pivot = reference_pivot(ref_raw)
    ref_long = reference_long_with_rollups(pivot)

    wandb_rows = load_wandb(refresh=refresh)
    exp109_df = exp109_long(wandb_rows)

    merged = merge_long(ref_long, exp109_df)
    enriched = add_composites(merged)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    enriched.sort_values(["source", "experiment", "dataset", "metric"]).to_csv(MERGED_CSV_PATH, index=False)
    print(f"Saved merged long-form data to {MERGED_CSV_PATH} ({len(enriched)} rows)")

    viz = select_viz_rows(enriched)
    plot_composites(viz)


if __name__ == "__main__":
    refresh = "--refresh" in sys.argv
    main(refresh=refresh)
