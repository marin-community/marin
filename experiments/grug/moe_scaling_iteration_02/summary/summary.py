# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Fit iteration_02 isoflop curves for AdamH and GatedNorm runs from W&B.

This adapts the legacy `experiments/grug/moe_scaling/summary/summary.py`
workflow to the iteration_02 AdamH/GatedNorm families. The run collector is
deliberately tolerant of reruns: it scans the launch groups declared in the
iteration_02 launchers, parses budget/hidden-dim from tags first, then falls
back to irregular run names, and picks one canonical run per
`(variant, budget, hidden_dim)` cell.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import re

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import wandb

from experiments.grug.moe_scaling_iteration_02.launch import (
    _build_model_config,
    _compute_flops_per_token,
    _compute_tokens_and_batch,
)
from experiments.grug.moe_scaling_iteration_02.model import GrugModelConfig

WANDB_PROJECT = "marin-community/dial_moe"
DEFAULT_METRIC_KEY = "eval/paloma/c4_en/bpb"
OUTPUT_DIR = Path(__file__).parent
TRACKED_BUDGETS: tuple[float, ...] = (1e18, 3e18, 1e19)
TRACKED_HIDDEN_DIMS: tuple[int, ...] = (512, 768, 1024, 1536, 2048)
EXTRAPOLATION_BUDGETS: tuple[float, ...] = (1e18, 3e18, 1e19, 1e20, 1e21)
GATED_NORM_RANK = 16

VARIANT_LABELS = {
    "adamh": "AdamH",
    "gatednorm": "GatedNorm",
}
CANONICAL_GROUPS = {
    "adamh": ("isoflop-moe-adamh-r3-gatedschema",),
    "gatednorm": ("isoflop-moe-adamh-gatednorm-r2",),
}

BUDGET_TAG_RE = re.compile(r"^budget=(?P<budget>\d+(?:\.\d+)?e\+?\d+)$")
DIM_TAG_RE = re.compile(r"^d=(?P<hidden_dim>\d+)$")
BUDGET_TEXT_RE = re.compile(r"(?<!\d)(?P<budget>(?:1|3)e\+?18|1e\+?19)(?=-|$)")
DIM_TEXT_RE = re.compile(r"(?:^|-)d(?P<hidden_dim>\d+)(?:-|$)")
GROUP_RE = re.compile(r'^GROUP = "(?P<group>[^"]+)"$', re.MULTILINE)

PALETTE = ["#1877F2", "#F0701A", "#5A24C7", "#E42C97", "#00487C"]
TRAINING_CURVE_METRICS = [
    ("train/cross_entropy_loss", "Cross Entropy Loss"),
    ("train/router/load_balancing_loss", "Load Balancing Loss"),
    ("train/router/router_z_loss", "Router Z Loss"),
    ("grad/norm/total", "Grad Norm"),
    ("moe_bias/layer_1/expert_10", "MoE Bias L1/E10"),
    ("moe_bias/layer_4/expert_10", "MoE Bias L4/E10"),
    ("moe_bias/layer_6/expert_10", "MoE Bias L6/E10"),
]


@dataclass(frozen=True)
class RunCell:
    variant: str
    budget: float
    hidden_dim: int
    name: str
    group: str
    path: str | None
    state: str
    metric: float
    tokens: float
    batch_size: int
    train_steps: int
    params: float
    step: int
    created_ts: float
    tags: tuple[str, ...]


@dataclass(frozen=True)
class IsoFlopFit:
    budget: float
    a: float
    b: float
    c: float
    logT_star: float
    T_star: float
    loss_star: float


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variant",
        choices=("all", "adamh", "gatednorm"),
        default="all",
        help="Which variant family to summarize.",
    )
    parser.add_argument(
        "--metric",
        default=DEFAULT_METRIC_KEY,
        help=f"W&B summary metric to fit. Defaults to {DEFAULT_METRIC_KEY}.",
    )
    parser.add_argument(
        "--skip-training-curves",
        action="store_true",
        help="Skip the training-curve subplot export for the best 1e19 run.",
    )
    return parser.parse_args()


def _launch_root() -> Path:
    return OUTPUT_DIR.parent


def _declared_group_names() -> tuple[str, ...]:
    groups: set[str] = set()
    for path in _launch_root().glob("launch_isoflop_moe_adamh*.py"):
        text = path.read_text()
        groups.update(match.group("group") for match in GROUP_RE.finditer(text))

    groups.update(group for family in CANONICAL_GROUPS.values() for group in family)
    return tuple(sorted(groups))


def _metric_value(summary: Any, metric_key: str) -> float | None:
    value = summary.get(metric_key)
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _extract_budget(tags: list[str], config: dict[str, Any], *texts: str) -> float | None:
    for tag in tags:
        match = BUDGET_TAG_RE.match(tag)
        if match:
            return float(match.group("budget"))

    for key in ("budget",):
        value = config.get(key)
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value)
            except ValueError:
                pass

    for text in texts:
        match = BUDGET_TEXT_RE.search(text)
        if match:
            return float(match.group("budget"))

    return None


def _extract_hidden_dim(tags: list[str], config: dict[str, Any], *texts: str) -> int | None:
    for tag in tags:
        match = DIM_TAG_RE.match(tag)
        if match:
            return int(match.group("hidden_dim"))

    value = config.get("hidden_dim")
    if isinstance(value, int):
        return value
    model_cfg = config.get("model")
    if isinstance(model_cfg, dict):
        hidden_dim = model_cfg.get("hidden_dim")
        if isinstance(hidden_dim, int):
            return hidden_dim

    for text in texts:
        match = DIM_TEXT_RE.search(text)
        if match:
            return int(match.group("hidden_dim"))

    return None


def _extract_step(summary: Any) -> int:
    for key in ("_step", "global_step", "step"):
        value = summary.get(key)
        if isinstance(value, (int, float)):
            return int(value)
    return 0


def _parse_timestamp(raw: Any) -> float:
    if not isinstance(raw, str) or not raw:
        return 0.0

    normalized = raw.replace("Z", "+00:00")
    try:
        return float(datetime.fromisoformat(normalized).timestamp())
    except ValueError:
        return 0.0


def _run_path(run: Any) -> str | None:
    path = getattr(run, "path", None)
    if isinstance(path, list) and path:
        return "/".join(path)
    return None


def _variant_for_run(name: str, group: str, tags: list[str], config: dict[str, Any]) -> str | None:
    text = " ".join([name, group, *tags]).lower()
    if "gatednorm" in text:
        return "gatednorm"

    gated_norm_rank = config.get("gated_norm_rank")
    if isinstance(gated_norm_rank, int) and gated_norm_rank > 0:
        return "gatednorm"

    if "isoflop-moe-adamh" in text or "attn-mlp-lmh-adamh" in tags:
        return "adamh"

    return None


def _iteration_02_run(name: str, group: str, tags: list[str]) -> bool:
    text = " ".join([name, group, *tags]).lower()
    if "isoflop-moe-adamh" in text:
        return True

    if "attn-mlp-lmh-adamh" not in tags or "isoflop" not in tags:
        return False

    return any(marker in text for marker in ("iteration-02", "iteration_02", "it02"))


def _iter_candidate_runs(api: wandb.Api) -> list[Any]:
    seen_paths: set[str] = set()
    runs: list[Any] = []

    for group in _declared_group_names():
        for run in api.runs(WANDB_PROJECT, filters={"group": group}):
            path = _run_path(run) or f"{group}:{getattr(run, 'name', '')}"
            if path in seen_paths:
                continue
            seen_paths.add(path)
            runs.append(run)

    for run in api.runs(WANDB_PROJECT, filters={"tags": {"$in": ["attn-mlp-lmh-adamh"]}}):
        path = _run_path(run) or f"{getattr(run, 'group', '')}:{getattr(run, 'name', '')}"
        if path in seen_paths:
            continue
        seen_paths.add(path)
        runs.append(run)

    return runs


def _group_priority(variant: str, group: str, tags: tuple[str, ...]) -> int:
    if group in CANONICAL_GROUPS[variant]:
        return 2
    if any(tag == f"source={CANONICAL_GROUPS[variant][0]}" for tag in tags):
        return 1
    return 0


def _state_priority(state: str) -> int:
    return {
        "finished": 3,
        "running": 2,
        "failed": 1,
        "crashed": 1,
        "killed": 0,
    }.get(state, 0)


def _variant_model_config(hidden_dim: int, variant: str) -> GrugModelConfig:
    cfg = _build_model_config(hidden_dim)
    if variant == "gatednorm":
        return dataclasses.replace(cfg, gated_norm_rank=GATED_NORM_RANK)
    return cfg


def _estimate_total_params(cfg: GrugModelConfig) -> int:
    d = cfg.hidden_dim
    vocab_size = cfg.vocab_size
    num_layers = cfg.num_layers
    num_dense_layers = cfg.num_dense_layers
    num_moe_layers = num_layers - num_dense_layers
    num_heads = cfg.num_heads
    num_kv_heads = cfg.num_kv_heads
    head_dim = d // num_heads

    embed = vocab_size * d
    attn = num_layers * (d * d + 2 * num_kv_heads * head_dim * d + d * d)
    dense_mlp = num_dense_layers * (2 * d * cfg.dense_intermediate_dim + cfg.dense_intermediate_dim * d)
    expert_mlp = num_moe_layers * cfg.num_experts * (2 * d * cfg.intermediate_dim + cfg.intermediate_dim * d)
    shared_mlp = num_moe_layers * (2 * d * cfg.shared_expert_intermediate_dim + cfg.shared_expert_intermediate_dim * d)
    norms = (2 * num_layers + 1) * d

    total = embed + attn + dense_mlp + expert_mlp + shared_mlp + norms
    if cfg.gated_norm_rank is not None:
        gated_norms = (2 * num_layers + 2) * (2 * d * cfg.gated_norm_rank)
        total += gated_norms
    return total


def _run_to_cell(run: Any, metric_key: str) -> RunCell | None:
    name = getattr(run, "name", "")
    group = getattr(run, "group", "") or ""
    tags = list(getattr(run, "tags", []) or [])
    config = dict(getattr(run, "config", {}) or {})

    if not _iteration_02_run(name, group, tags):
        return None

    variant = _variant_for_run(name, group, tags, config)
    if variant is None:
        return None

    budget = _extract_budget(tags, config, name, group)
    hidden_dim = _extract_hidden_dim(tags, config, name, group)
    if budget is None or hidden_dim is None:
        return None
    if budget not in TRACKED_BUDGETS or hidden_dim not in TRACKED_HIDDEN_DIMS:
        return None

    summary = getattr(run, "summary", {})
    metric = _metric_value(summary, metric_key)
    if metric is None:
        return None
    step = _extract_step(summary)
    if step <= 0:
        return None

    model_cfg = _variant_model_config(hidden_dim, variant)
    flops_per_token = _compute_flops_per_token(model_cfg)
    tokens, batch_size, train_steps = _compute_tokens_and_batch(budget, flops_per_token)
    params = summary.get("parameter_count")
    if not isinstance(params, (int, float)):
        params = _estimate_total_params(model_cfg)

    return RunCell(
        variant=variant,
        budget=budget,
        hidden_dim=hidden_dim,
        name=name,
        group=group,
        path=_run_path(run),
        state=getattr(run, "state", "unknown"),
        metric=metric,
        tokens=tokens,
        batch_size=batch_size,
        train_steps=train_steps,
        params=float(params),
        step=step,
        created_ts=max(
            _parse_timestamp(getattr(run, "updated_at", None)),
            _parse_timestamp(getattr(run, "created_at", None)),
            float(summary.get("_timestamp", 0.0) or 0.0),
        ),
        tags=tuple(tags),
    )


def fetch_run_cells(metric_key: str) -> tuple[list[RunCell], dict[str, list[str]]]:
    api = wandb.Api()
    rows: list[RunCell] = []
    groups_to_names: dict[str, list[str]] = defaultdict(list)

    for run in _iter_candidate_runs(api):
        row = _run_to_cell(run, metric_key)
        if row is None:
            continue
        rows.append(row)
        groups_to_names[row.group].append(row.name)

    return rows, groups_to_names


def select_best_runs(rows: list[RunCell]) -> tuple[list[RunCell], dict[tuple[str, float, int], list[RunCell]]]:
    grouped: dict[tuple[str, float, int], list[RunCell]] = defaultdict(list)
    for row in rows:
        grouped[(row.variant, row.budget, row.hidden_dim)].append(row)

    selected: list[RunCell] = []
    for key, candidates in grouped.items():
        variant = key[0]
        best = max(
            candidates,
            key=lambda row: (
                min(row.step, row.train_steps),
                _state_priority(row.state),
                _group_priority(variant, row.group, row.tags),
                row.created_ts,
            ),
        )
        selected.append(best)

    selected.sort(key=lambda row: (row.variant, row.budget, row.hidden_dim))
    return selected, grouped


def fit_parabola(budget_rows: list[RunCell]) -> IsoFlopFit:
    tokens = np.array([row.tokens for row in budget_rows], dtype=float)
    losses = np.array([row.metric for row in budget_rows], dtype=float)
    log_tokens = np.log10(tokens)

    a, b, c = np.polyfit(log_tokens, losses, 2)
    log_t_star = float(np.clip(-b / (2 * a), log_tokens.min(), log_tokens.max()))
    t_star = 10**log_t_star
    loss_star = a * log_t_star**2 + b * log_t_star + c

    return IsoFlopFit(
        budget=budget_rows[0].budget,
        a=a,
        b=b,
        c=c,
        logT_star=log_t_star,
        T_star=t_star,
        loss_star=float(loss_star),
    )


def fit_frontier(fits: list[IsoFlopFit]) -> tuple[np.ndarray, np.ndarray]:
    log_flops = np.array([np.log10(fit.budget) for fit in fits])
    log_tokens = np.array([fit.logT_star for fit in fits])
    losses = np.array([fit.loss_star for fit in fits])

    token_coeffs = np.polyfit(log_flops, log_tokens, 1)
    loss_coeffs = np.polyfit(log_tokens, losses, 1)
    return loss_coeffs, token_coeffs


def fit_param_frontier(fits: list[IsoFlopFit], variant: str) -> np.ndarray:
    log_flops = np.array([np.log10(fit.budget) for fit in fits])
    log_params = np.array([np.log10(_optimal_params_for_budget(fit.budget, fit.T_star, variant)) for fit in fits])
    return np.polyfit(log_flops, log_params, 1)


def _optimal_params_for_budget(budget: float, optimal_tokens: float, variant: str) -> float:
    flops_per_token = budget / (3 * optimal_tokens)
    flops_grid = []
    params_grid = []
    for hidden_dim in TRACKED_HIDDEN_DIMS:
        cfg = _variant_model_config(hidden_dim, variant)
        flops_grid.append(_compute_flops_per_token(cfg))
        params_grid.append(_estimate_total_params(cfg))
    return float(10 ** np.interp(np.log10(flops_per_token), np.log10(flops_grid), np.log10(params_grid)))


def predict_optimal(
    token_coeffs: np.ndarray,
    loss_coeffs: np.ndarray,
    budget: float,
    param_coeffs: np.ndarray | None = None,
) -> tuple[float, float, float | None]:
    log_budget = np.log10(budget)
    log_t_star = np.polyval(token_coeffs, log_budget)
    loss_star = np.polyval(loss_coeffs, log_t_star)
    params_star = None
    if param_coeffs is not None:
        params_star = float(10 ** np.polyval(param_coeffs, log_budget))
    return 10**log_t_star, float(loss_star), params_star


def fmt_sci(value: float) -> str:
    exponent = math.floor(math.log10(abs(value)))
    coeff = value / 10**exponent
    if coeff == 1.0:
        return f"1e{exponent}"
    return f"{coeff:.0f}e{exponent}"


def _slugify_metric(metric_key: str) -> str:
    return metric_key.replace("/", "_")


def write_figure(fig: go.Figure, stem: str) -> Path:
    image_path = OUTPUT_DIR / f"{stem}.png"
    try:
        fig.write_image(str(image_path), scale=2)
        return image_path
    except ValueError as err:
        if "Kaleido" not in str(err):
            raise

    html_path = OUTPUT_DIR / f"{stem}.html"
    fig.write_html(str(html_path), include_plotlyjs="cdn")
    return html_path


def make_plot(
    rows: list[RunCell],
    fits: list[IsoFlopFit],
    token_coeffs: np.ndarray,
    loss_coeffs: np.ndarray,
    *,
    metric_key: str,
    variant: str,
) -> go.Figure:
    fig = go.Figure()
    budget_colors = {budget: PALETTE[i] for i, budget in enumerate(TRACKED_BUDGETS)}

    for fit in fits:
        budget = fit.budget
        color = budget_colors.get(budget, "#888")
        label = fmt_sci(budget)
        subset = [row for row in rows if row.budget == budget]

        fig.add_trace(
            go.Scatter(
                x=[row.tokens for row in subset],
                y=[row.metric for row in subset],
                mode="markers+text",
                marker=dict(size=8, color=color),
                text=[f"d{row.hidden_dim}" for row in subset],
                textposition="top center",
                textfont=dict(size=9),
                name=f"{label} FLOPs",
                legendgroup=label,
                hovertemplate="d=%{text}<br>tokens=%{x:.3e}<br>metric=%{y:.4f}<extra></extra>",
            )
        )

        log_tokens = np.log10(np.array([row.tokens for row in subset]))
        grid = np.linspace(log_tokens.min(), log_tokens.max(), 200)
        yhat = fit.a * grid**2 + fit.b * grid + fit.c
        fig.add_trace(
            go.Scatter(
                x=10**grid,
                y=yhat,
                mode="lines",
                line=dict(width=2, color=color),
                name=f"{label} fit",
                legendgroup=label,
                showlegend=False,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[fit.T_star],
                y=[fit.loss_star],
                mode="markers",
                marker=dict(symbol="x", size=14, color=color, line=dict(width=2)),
                name=f"{label} optimum",
                legendgroup=label,
                showlegend=False,
                hovertemplate=f"Optimum @ {label}<br>T*=%{{x:.3e}}<br>metric=%{{y:.4f}}<extra></extra>",
            )
        )

    frontier_tokens = []
    frontier_losses = []
    for budget in EXTRAPOLATION_BUDGETS:
        t_star, loss_star, _ = predict_optimal(token_coeffs, loss_coeffs, budget)
        frontier_tokens.append(np.log10(t_star))
        frontier_losses.append(loss_star)

    grid_tokens = np.linspace(min(frontier_tokens), max(frontier_tokens), 200)
    grid_loss = np.polyval(loss_coeffs, grid_tokens)
    fig.add_trace(
        go.Scatter(
            x=10**grid_tokens,
            y=grid_loss,
            mode="lines",
            line=dict(width=2, dash="dash", color="gray"),
            name="Optimal frontier",
        )
    )

    for budget in EXTRAPOLATION_BUDGETS:
        t_star, loss_star, _ = predict_optimal(token_coeffs, loss_coeffs, budget)
        fig.add_trace(
            go.Scatter(
                x=[t_star],
                y=[loss_star],
                mode="markers",
                marker=dict(symbol="diamond", size=10, color="gray", line=dict(width=1, color="black")),
                name=f"Predicted @ {fmt_sci(budget)}",
                hovertemplate=f"Predicted @ {fmt_sci(budget)}<br>T*=%{{x:.3e}}<br>metric=%{{y:.4f}}<extra></extra>",
            )
        )

    fig.update_xaxes(type="log", title_text="Training tokens")
    fig.update_yaxes(title_text=metric_key)
    fig.update_layout(
        template="plotly_white",
        title=f"iteration_02 {VARIANT_LABELS[variant]} ISOFlop curves ({metric_key})",
        width=1000,
        height=600,
    )
    return fig


def make_training_curves(api: wandb.Api, row: RunCell) -> Path:
    runs = api.runs(WANDB_PROJECT, filters={"display_name": row.name})
    run = next(iter(runs), None)
    if run is None:
        raise ValueError(f"Run not found in W&B for {row.name}")

    keys = ["_step"] + [metric for metric, _ in TRAINING_CURVE_METRICS]
    history = list(run.scan_history(keys=keys, page_size=10000))

    ncols = 2
    nrows = (len(TRAINING_CURVE_METRICS) + ncols - 1) // ncols
    fig = make_subplots(
        rows=nrows,
        cols=ncols,
        subplot_titles=[title for _, title in TRAINING_CURVE_METRICS],
        vertical_spacing=0.08,
        horizontal_spacing=0.1,
    )

    for index, (metric_key, title) in enumerate(TRAINING_CURVE_METRICS):
        subplot_row = index // ncols + 1
        subplot_col = index % ncols + 1
        steps = [point["_step"] for point in history if point.get(metric_key) is not None]
        values = [point[metric_key] for point in history if point.get(metric_key) is not None]
        fig.add_trace(
            go.Scatter(x=steps, y=values, mode="lines", line=dict(width=1.5), name=title, showlegend=False),
            row=subplot_row,
            col=subplot_col,
        )
        fig.update_xaxes(title_text="Step", row=subplot_row, col=subplot_col)

    fig.update_layout(
        template="plotly_white",
        title=f"{row.name} Training Curves",
        width=1200,
        height=300 * nrows + 100,
    )

    return write_figure(fig, f"training_curves_{row.name.replace('+', '')}")


def _fit_variant(rows: list[RunCell]) -> list[IsoFlopFit]:
    fits: list[IsoFlopFit] = []
    grouped: dict[float, list[RunCell]] = defaultdict(list)
    for row in rows:
        grouped[row.budget].append(row)

    for budget in sorted(grouped):
        budget_rows = grouped[budget]
        if len(budget_rows) < 3:
            print(f"  Skipping {fmt_sci(budget)}: only {len(budget_rows)} selected runs")
            continue
        fit = fit_parabola(budget_rows)
        fits.append(fit)
        print(f"  {fmt_sci(budget)}: optimal tokens={fit.T_star:.3e}, metric*={fit.loss_star:.4f}")
    return fits


def _best_1e19_run(rows: list[RunCell]) -> RunCell | None:
    candidates = [row for row in rows if row.budget == 1e19]
    if not candidates:
        return None
    return min(candidates, key=lambda row: row.metric)


def _serialize_rows(rows: list[RunCell]) -> list[dict[str, Any]]:
    return [
        {
            "variant": row.variant,
            "budget": row.budget,
            "hidden_dim": row.hidden_dim,
            "name": row.name,
            "group": row.group,
            "path": row.path,
            "state": row.state,
            "metric": row.metric,
            "tokens": row.tokens,
            "batch_size": row.batch_size,
            "train_steps": row.train_steps,
            "params": row.params,
            "step": row.step,
            "tags": list(row.tags),
        }
        for row in rows
    ]


def _missing_cells(rows: list[RunCell], variant: str) -> list[dict[str, Any]]:
    seen = {(row.budget, row.hidden_dim) for row in rows if row.variant == variant}
    missing = []
    for budget in TRACKED_BUDGETS:
        for hidden_dim in TRACKED_HIDDEN_DIMS:
            if (budget, hidden_dim) in seen:
                continue
            missing.append({"budget": budget, "hidden_dim": hidden_dim})
    return missing


def write_variant_summary(
    rows: list[RunCell],
    fits: list[IsoFlopFit],
    *,
    variant: str,
    metric_key: str,
    token_coeffs: np.ndarray,
    loss_coeffs: np.ndarray,
    param_coeffs: np.ndarray,
    best_run: RunCell | None,
) -> Path:
    predictions = {}
    for budget in EXTRAPOLATION_BUDGETS:
        t_star, loss_star, params_star = predict_optimal(token_coeffs, loss_coeffs, budget, param_coeffs)
        predictions[fmt_sci(budget)] = {
            "optimal_tokens": t_star,
            "optimal_model_params": params_star,
            "predicted_metric": loss_star,
        }

    result = {
        "variant": variant,
        "metric": metric_key,
        "selected_runs": _serialize_rows(rows),
        "missing_cells": _missing_cells(rows, variant),
        "predictions": predictions,
        "best_1e19_run": (
            None
            if best_run is None
            else {
                "name": best_run.name,
                "budget": best_run.budget,
                "hidden_dim": best_run.hidden_dim,
                "metric": best_run.metric,
                "tokens": best_run.tokens,
                "batch_size": best_run.batch_size,
                "train_steps": best_run.train_steps,
                "params": best_run.params,
                "group": best_run.group,
                "path": best_run.path,
            }
        ),
    }

    output_path = OUTPUT_DIR / f"summary_{variant}_{_slugify_metric(metric_key)}.json"
    output_path.write_text(json.dumps(result, indent=2) + "\n")
    return output_path


def summarize_variant(
    api: wandb.Api,
    rows: list[RunCell],
    *,
    variant: str,
    metric_key: str,
    skip_training_curves: bool,
) -> None:
    variant_rows = [row for row in rows if row.variant == variant]
    print(f"\n=== {VARIANT_LABELS[variant]} ===")
    print(f"Selected {len(variant_rows)} cells")
    for row in variant_rows:
        print(
            f"  {fmt_sci(row.budget)} d{row.hidden_dim}: {row.metric:.4f} "
            f"from {row.name} [{row.state}, group={row.group}]"
        )

    fits = _fit_variant(variant_rows)
    if len(fits) < 2:
        print("  Not enough fitted budgets to extrapolate frontier")
        return

    loss_coeffs, token_coeffs = fit_frontier(fits)
    param_coeffs = fit_param_frontier(fits, variant)

    fig = make_plot(variant_rows, fits, token_coeffs, loss_coeffs, metric_key=metric_key, variant=variant)
    image_path = write_figure(fig, f"isoflop_curve_{variant}_{_slugify_metric(metric_key)}")
    print(f"  Plot saved to {image_path}")

    best_run = _best_1e19_run(variant_rows)
    if best_run is not None and not skip_training_curves:
        curves_path = make_training_curves(api, best_run)
        print(f"  Training curves saved to {curves_path}")

    summary_path = write_variant_summary(
        variant_rows,
        fits,
        variant=variant,
        metric_key=metric_key,
        token_coeffs=token_coeffs,
        loss_coeffs=loss_coeffs,
        param_coeffs=param_coeffs,
        best_run=best_run,
    )
    print(f"  Summary saved to {summary_path}")


def write_selection_manifest(
    all_rows: list[RunCell],
    grouped_rows: dict[tuple[str, float, int], list[RunCell]],
    groups_to_names: dict[str, list[str]],
    metric_key: str,
) -> Path:
    manifest = {
        "metric": metric_key,
        "declared_groups": {group: sorted(names) for group, names in sorted(groups_to_names.items())},
        "selected_runs": _serialize_rows(all_rows),
        "duplicates": {
            f"{variant}:{fmt_sci(budget)}:d{hidden_dim}": _serialize_rows(rows)
            for (variant, budget, hidden_dim), rows in sorted(grouped_rows.items())
            if len(rows) > 1
        },
    }
    output_path = OUTPUT_DIR / f"selected_runs_{_slugify_metric(metric_key)}.json"
    output_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return output_path


def main() -> None:
    args = _parse_args()
    variants = ("adamh", "gatednorm") if args.variant == "all" else (args.variant,)

    print("Fetching iteration_02 AdamH-family runs from W&B...")
    rows, groups_to_names = fetch_run_cells(args.metric)
    selected_rows, grouped_rows = select_best_runs(rows)
    manifest_path = write_selection_manifest(selected_rows, grouped_rows, groups_to_names, args.metric)
    print(f"Selection manifest saved to {manifest_path}")

    api = wandb.Api()
    for variant in variants:
        summarize_variant(
            api, selected_rows, variant=variant, metric_key=args.metric, skip_training_curves=args.skip_training_curves
        )


if __name__ == "__main__":
    main()
