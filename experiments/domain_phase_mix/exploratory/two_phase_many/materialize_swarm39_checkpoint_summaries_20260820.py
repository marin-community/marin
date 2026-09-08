# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas", "wandb"]
# ///
"""Materialize strictly pre-boundary checkpoint summaries for Delphi heldouts.

The W&B run history logs parameter and gradient norms every ten steps. This
script summarizes the final phase-0 window while excluding the boundary step,
which may already contain the first phase-1 batch. Output is append-safe and a
rerun skips every row that has already materialized successfully.
"""

from __future__ import annotations

import argparse
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import wandb

SCRIPT_DIR = Path(__file__).resolve().parent
HELDOUT_PATH = SCRIPT_DIR / "reference_outputs" / "delphi_3e18_append_only_heldouts_20260714" / "heldout_current.csv"
OUTPUT_DIR = SCRIPT_DIR / "reference_outputs" / "swarm39_checkpoint_summaries_20260820"
OUTPUT_PATH = OUTPUT_DIR / "checkpoint_summaries.csv"

LAYERS = tuple(range(10))
WINDOW_STEPS = 500
WINDOW_ROWS = 20
EPSILON = 1e-30

ATTENTION_COMPONENTS = (
    "self_attn.k_proj.weight",
    "self_attn.q_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.o_proj.weight",
)
MLP_COMPONENTS = ("mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight")
NORM_COMPONENTS = (
    "input_layernorm.weight",
    "post_attention_layernorm.weight",
    "self_attn.k_norm.weight",
    "self_attn.q_norm.weight",
)

FEATURE_COLUMNS = (
    "state_train_loss_log_mean",
    "state_train_loss_log_slope",
    "state_grad_total_log_mean",
    "state_grad_total_log_sd",
    "state_grad_total_log_slope",
    "state_param_total_log_mean",
    "state_grad_param_total_log_ratio",
    "state_attention_grad_param_log_ratio",
    "state_mlp_grad_param_log_ratio",
    "state_norm_grad_param_log_ratio",
    "state_attention_depth_slope",
    "state_mlp_depth_slope",
    "state_attention_depth_sd",
    "state_mlp_depth_sd",
    "state_embedding_grad_param_log_ratio",
    "state_head_grad_param_log_ratio",
)


def norm_key(kind: str, layer: int, component: str) -> str:
    return f"{kind}/norm/transformer.layers.{layer}.{component}"


def source_keys() -> tuple[str, ...]:
    keys = [
        "_step",
        "train/loss",
        "grad/norm/total",
        "params/norm/total",
        "grad/norm/embeddings.token_embeddings.weight",
        "params/norm/embeddings.token_embeddings.weight",
        "grad/norm/lm_head.weight",
        "params/norm/lm_head.weight",
    ]
    for layer in LAYERS:
        for component in (*ATTENTION_COMPONENTS, *MLP_COMPONENTS, *NORM_COMPONENTS):
            keys.extend((norm_key("grad", layer, component), norm_key("params", layer, component)))
    return tuple(keys)


SOURCE_KEYS = source_keys()


def log_values(values: np.ndarray) -> np.ndarray:
    assert np.isfinite(values).all() and (values > 0.0).all()
    return np.log(np.maximum(values, EPSILON))


def slope(values: np.ndarray) -> float:
    if len(values) < 2:
        return 0.0
    coordinate = np.linspace(-1.0, 1.0, len(values))
    return float(np.polyfit(coordinate, values, 1)[0])


def aggregate_components(frame: pd.DataFrame, kind: str, components: tuple[str, ...]) -> np.ndarray:
    values = np.empty((len(frame), len(LAYERS)))
    for column, layer in enumerate(LAYERS):
        matrix = frame[[norm_key(kind, layer, component) for component in components]].to_numpy(float)
        values[:, column] = np.sqrt(np.square(matrix).sum(axis=1))
    return values


def summarize_history(history: list[dict], boundary: int) -> dict[str, float | int]:
    frame = pd.DataFrame(history).sort_values("_step").tail(WINDOW_ROWS).reset_index(drop=True)
    missing = [key for key in SOURCE_KEYS if key not in frame or frame[key].isna().any()]
    if missing:
        raise ValueError(f"missing {len(missing)} required keys, first={missing[:3]}")
    assert bool((frame["_step"] < boundary).all()), "post-boundary row entered checkpoint summary"

    train_loss = log_values(frame["train/loss"].to_numpy(float))
    grad_total = log_values(frame["grad/norm/total"].to_numpy(float))
    param_total = log_values(frame["params/norm/total"].to_numpy(float))

    group_ratios: dict[str, np.ndarray] = {}
    for name, components in (
        ("attention", ATTENTION_COMPONENTS),
        ("mlp", MLP_COMPONENTS),
        ("norm", NORM_COMPONENTS),
    ):
        grad = aggregate_components(frame, "grad", components)
        param = aggregate_components(frame, "params", components)
        group_ratios[name] = log_values(grad) - log_values(param)

    attention_depth = group_ratios["attention"].mean(axis=0)
    mlp_depth = group_ratios["mlp"].mean(axis=0)
    embedding_ratio = log_values(frame["grad/norm/embeddings.token_embeddings.weight"].to_numpy(float)) - log_values(
        frame["params/norm/embeddings.token_embeddings.weight"].to_numpy(float)
    )
    head_ratio = log_values(frame["grad/norm/lm_head.weight"].to_numpy(float)) - log_values(
        frame["params/norm/lm_head.weight"].to_numpy(float)
    )

    return {
        "summary_step": int(frame["_step"].iloc[-1]),
        "summary_fraction": float(frame["_step"].iloc[-1]) / float(boundary),
        "summary_rows": len(frame),
        "state_train_loss_log_mean": float(train_loss.mean()),
        "state_train_loss_log_slope": slope(train_loss),
        "state_grad_total_log_mean": float(grad_total.mean()),
        "state_grad_total_log_sd": float(grad_total.std(ddof=0)),
        "state_grad_total_log_slope": slope(grad_total),
        "state_param_total_log_mean": float(param_total.mean()),
        "state_grad_param_total_log_ratio": float((grad_total - param_total).mean()),
        "state_attention_grad_param_log_ratio": float(group_ratios["attention"].mean()),
        "state_mlp_grad_param_log_ratio": float(group_ratios["mlp"].mean()),
        "state_norm_grad_param_log_ratio": float(group_ratios["norm"].mean()),
        "state_attention_depth_slope": slope(attention_depth),
        "state_mlp_depth_slope": slope(mlp_depth),
        "state_attention_depth_sd": float(attention_depth.std(ddof=0)),
        "state_mlp_depth_sd": float(mlp_depth.std(ddof=0)),
        "state_embedding_grad_param_log_ratio": float(embedding_ratio.mean()),
        "state_head_grad_param_log_ratio": float(head_ratio.mean()),
    }


def materialize_row(api: wandb.Api, row) -> dict[str, object]:
    base = {
        "heldout_id": row.heldout_id,
        "wandb_run_id": row.wandb_run_id,
        "phase_boundary_step": row.phase_boundary_step,
    }
    try:
        boundary = int(row.phase_boundary_step)
        run = api.run(f"{row.wandb_entity}/{row.wandb_project}/{row.wandb_run_id}")
        history = list(
            run.scan_history(
                keys=list(SOURCE_KEYS),
                min_step=max(0, boundary - WINDOW_STEPS),
                max_step=boundary - 1,
                page_size=100,
            )
        )
        if not history:
            raise ValueError("no complete norm rows strictly before phase boundary")
        return {**base, **summarize_history(history, boundary), "error": None}
    except Exception as error:
        return {**base, "error": f"{type(error).__name__}: {error}"}


def write_rows(existing: pd.DataFrame, rows: list[dict[str, object]], output: Path) -> pd.DataFrame:
    combined = pd.concat([existing, pd.DataFrame(rows)], ignore_index=True)
    combined = combined.drop_duplicates("heldout_id", keep="last").sort_values("heldout_id")
    temporary = output.with_suffix(".tmp.csv")
    combined.to_csv(temporary, index=False)
    os.replace(temporary, output)
    return combined


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--retry-errors", action="store_true")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    panel = pd.read_csv(HELDOUT_PATH)
    panel = panel[panel["fit_panel_overlap"] == "coordinate_disjoint"].copy()
    panel = panel[panel["wandb_run_id"].notna() & panel["phase_boundary_step"].notna()]
    existing = pd.read_csv(OUTPUT_PATH) if OUTPUT_PATH.exists() else pd.DataFrame(columns=["heldout_id", "error"])
    complete = set(existing.loc[existing["error"].isna(), "heldout_id"])
    if args.retry_errors:
        pending = panel[~panel["heldout_id"].isin(complete)]
    else:
        attempted = set(existing["heldout_id"])
        pending = panel[~panel["heldout_id"].isin(attempted)]
    if args.limit:
        pending = pending.head(args.limit)

    print(f"{len(panel)} rows in scope, {len(complete)} complete, {len(pending)} pending", flush=True)
    if pending.empty:
        return

    api = wandb.Api(timeout=90)
    buffer: list[dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(materialize_row, api, row): row.heldout_id for row in pending.itertuples()}
        for completed, future in enumerate(as_completed(futures), start=1):
            buffer.append(future.result())
            if len(buffer) >= 25 or completed == len(futures):
                existing = write_rows(existing, buffer, OUTPUT_PATH)
                buffer.clear()
            if completed % 100 == 0 or completed == len(futures):
                present = int(existing["error"].isna().sum())
                print(f"materialized {completed}/{len(futures)} pending; {present} total complete", flush=True)

    complete_mask = existing["error"].isna()
    print(f"complete {int(complete_mask.sum())}/{len(existing)}; failures {int((~complete_mask).sum())}")
    if not complete_mask.all():
        print(existing.loc[~complete_mask, "error"].value_counts().head(10).to_dict())
    if complete_mask.any():
        values = existing.loc[complete_mask, FEATURE_COLUMNS].to_numpy(float)
        assert np.isfinite(values).all()
        print(
            "summary fraction: "
            f"median={existing.loc[complete_mask, 'summary_fraction'].median():.5f}, "
            f"range=({existing.loc[complete_mask, 'summary_fraction'].min():.5f}, "
            f"{existing.loc[complete_mask, 'summary_fraction'].max():.5f})"
        )


if __name__ == "__main__":
    main()
