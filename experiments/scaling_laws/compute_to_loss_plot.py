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

import re
from collections.abc import Iterable

import jax.numpy as jnp
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
import wandb
import time
from jaxopt import ScipyMinimize

# ---------------- Theme ----------------
pio.templates.default = "plotly_white"


# ---------------- Constants ----------------
SEQ_LEN = 4096
REQUIRED_TAGS = {"steps", "B", "FLOPs", "d", "L"}
DEFAULT_METRIC_KEY = "eval/paloma/bpb"
# MMLU_CHOICE_KEY = "lm_eval/mmlu_sl_verb_5_shot/choice_logprob"
PALOMA_BPB_KEY = "eval/paloma/bpb"
PALOMA_C4EN_BPB_KEY = "eval/paloma/c4_en/bpb"
MMLU_ACC_KEY = "lm_eval/mmlu_sl_verb_5_shot/acc_norm"
# Additional sigmoid y-metrics
GSM8K_BPB_KEY = "lm_eval/gsm8k_loss_8shot/bpb"
MATH500_BPB_KEY = "lm_eval/math_500_loss/bpb"
HELLASWAG_ACC_KEY = "lm_eval/hellaswag_10shot/acc_norm"
UNCHEATABLE_EVAL_BPB_KEY = "eval/uncheatable_eval/bpb"


def _metric_slug(key: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_]+", "_", key)


# ---------------- Helpers ----------------
def _is_fit_run_name(name: str) -> bool:
    """Decide whether a run should be used for fitting.

    Include if:
    - contains "isoflop" OR matches "qwen-.*B-Base" (case-insensitive)
    Exclude if:
    - contains "marin" OR "llama" (case-insensitive)
    """
    if not name:
        return False
    s = str(name)
    if re.search(r"marin-community", s, flags=re.IGNORECASE) or re.search(r"llama", s, flags=re.IGNORECASE):
        return False
    if re.search(r"isoflop", s, flags=re.IGNORECASE):
        return True
    if re.search(r"qwen-.*B-Base", s, flags=re.IGNORECASE):
        return True
    return False


def _tags_to_dict(tags: Iterable[str]) -> dict[str, str]:
    return {k: v for k, v in (t.split("=", 1) for t in tags if "=" in t)}


def _parse_flops_from_name(name: str) -> float | None:
    """Parse FLOPs value from run name like 'isoflop-1e+19-...'.

    Returns float on success, None otherwise.
    """
    m = re.search(r"isoflop-([0-9.+\-eE]+)", name)
    if not m:
        return None
    try:
        flops_str = m.group(1).rstrip("-")
        return float(flops_str)
    except ValueError:
        return None


def df_from_sources(source_runs: list[tuple[list, str]], metric_key: str = DEFAULT_METRIC_KEY) -> pd.DataFrame:
    """Build a dataframe from [(runs, fragment), ...].

    Columns: tokens, loss, flops, params, name, label
    """
    records = []
    for runs, fragment in source_runs:
        for run in runs:
            tags = _tags_to_dict(run.tags)

            required_no_flops = REQUIRED_TAGS - {"FLOPs"}
            if not required_no_flops.issubset(tags):
                continue

            steps = float(tags["steps"])  # optimizer steps
            batch = float(tags["B"])  # global batch size
            name = run.displayName

            flops_tag = tags.get("FLOPs")
            if flops_tag is not None:
                flops = float(flops_tag)
            else:
                flops_parsed = _parse_flops_from_name(name)
                if flops_parsed is None:
                    continue
                flops = float(flops_parsed)
            if flops < 1e18:
                # Ignore tiny runs which can be noisy
                continue

            tokens = steps * batch * SEQ_LEN
            loss = run.summaryMetrics.get(metric_key)
            if loss is None:
                continue

            # Some metrics may be reported under different casing/keys on occasion.
            # Keep only finite values.
            try:
                loss = float(loss)
            except Exception:
                continue

            params = run.summaryMetrics.get("parameter_count")

            records.append(
                dict(
                    tokens=tokens,
                    loss=loss,
                    flops=flops,
                    params=params,
                    name=name,
                    label=fragment,
                    fit=_is_fit_run_name(name),
                )
            )
    return pd.DataFrame.from_records(records)


def mmlu_df_from_sources(
    source_runs: list[tuple[list, str]],
    key_x: str = PALOMA_BPB_KEY,
    key_y: str = MMLU_ACC_KEY,
) -> pd.DataFrame:
    """Backward-compatible wrapper for generic XY joiner for sigmoid plotting."""
    return xy_df_from_sources(source_runs, key_x=key_x, key_y=key_y)


def xy_df_from_sources(
    source_runs: list[tuple[list, str]],
    key_x: str,
    key_y: str,
) -> pd.DataFrame:
    """Collect metrics for sigmoid fit by consolidating across runs.

    Some runs contain only x (e.g., loss metrics) and others contain only y (e.g., accuracies).
    We key runs by their configuration (B, d, L, FLOPs, optionally M) and join x and y from different runs.

    Returns columns: x, y, name, label
    - x uses "loss (- choice log-prob)" convention: x = -choice_logprob when applicable
    - y is normalized to fraction if provided in [0,100]
    """
    x_by_cfg = {}
    y_by_cfg = {}
    label_by_cfg = {}

    def _cfg_key_from_run(run) -> tuple[float, float, float, float, str] | None:
        name = getattr(run, "displayName", getattr(run, "name", "")) or ""
        tags_iter = getattr(run, "tags", None)
        tag_dict = _tags_to_dict(tags_iter) if tags_iter else {}

        def _parse_float(value):
            try:
                return float(value)
            except Exception:
                return None

        # Try tags first
        B = _parse_float(tag_dict.get("B"))
        d = _parse_float(tag_dict.get("d"))
        L = _parse_float(tag_dict.get("L"))
        C = _parse_float(tag_dict.get("FLOPs"))
        M = tag_dict.get("M")

        # Fallbacks: parse from name (case-insensitive)
        if B is None:
            m = re.search(r"B(\d+)", name, flags=re.IGNORECASE)
            if m:
                B = _parse_float(m.group(1))
        if d is None:
            m = re.search(r"d(\d+)", name, flags=re.IGNORECASE)
            if m:
                d = _parse_float(m.group(1))
        if L is None:
            m = re.search(r"L(\d+)", name, flags=re.IGNORECASE)
            if m:
                L = _parse_float(m.group(1))
        if C is None:
            parsed = _parse_flops_from_name(name)
            if parsed is not None:
                C = _parse_float(parsed)

        # Model identifier is required to disambiguate configs across models
        if None in (B, d, L, C) and M is None:
            return None

        # Using model identifier
        if M is None:
            return (float(B), float(d), float(L), float(C))
        else:
            return str(M)

    # First pass: build maps for x and y separately
    for runs, fragment in source_runs:
        for run in runs:
            sm = getattr(run, "summaryMetrics", None)
            if not isinstance(sm, dict):
                continue
            cfg = _cfg_key_from_run(run)
            if cfg is None:
                continue

            # Preserve first-seen label for the config
            if cfg not in label_by_cfg:
                label_by_cfg[cfg] = fragment

            # x metric
            x_raw = sm.get(key_x)
            if x_raw is not None and cfg not in x_by_cfg:
                try:
                    x_val = float(x_raw)
                except Exception:
                    x_val = None
                if x_val is not None and jnp.isfinite(x_val):
                    x_loss = -float(x_val)
                    name_x = getattr(run, "displayName", getattr(run, "name", "")) or ""
                    x_by_cfg[cfg] = (x_loss, name_x)

            # y metric
            y_raw = sm.get(key_y)

            # print(run.displayName)
            # if "mmlu" in run.displayName.lower():
            #     print(sm.get(key_y))
            #     print(f"raw value: {y_raw}")

            if y_raw is not None and cfg not in y_by_cfg:
                # print(f"found y: {y_raw} for run: {run.displayName}")
                try:
                    y_val = float(y_raw)
                except Exception:
                    y_val = None
                if y_val is not None and jnp.isfinite(y_val):
                    # Normalize percentages to fraction if clearly in [0,100]
                    # y_norm = float(y_val) / 100.0 if float(y_val) > 1.0 and float(y_val) <= 100.0 else float(y_val)
                    #
                    name_y = getattr(run, "displayName", getattr(run, "name", "")) or ""
                    y_by_cfg[cfg] = (y_val, name_y)

    # Join on intersecting configs
    records = []
    for cfg in set(x_by_cfg.keys()) & set(y_by_cfg.keys()):
        (x_loss, name_x) = x_by_cfg[cfg]
        (y_acc, name_y) = y_by_cfg[cfg]
        label = label_by_cfg.get(cfg, "")
        combined_name = name_x if name_x == name_y else f"{name_x} | {name_y}"
        records.append(
            dict(
                x=float(x_loss),
                y=float(y_acc),
                name=combined_name,
                label=label,
                fit=_is_fit_run_name(combined_name),
            )
        )

    return pd.DataFrame.from_records(records)


def _robust_quad_logx(x: jnp.ndarray, y: jnp.ndarray, delta: float = 1.0) -> jnp.ndarray:
    """Robust quadratic fit: y ~ a*(log10(x))^2 + b*log10(x) + c."""
    L = jnp.log10(x)

    def huber(residual):
        abs_r = jnp.abs(residual)
        quad = 0.5 * residual**2
        linear = delta * (abs_r - 0.5 * delta)
        return jnp.where(abs_r <= delta, quad, linear)

    def objective(params):
        a, b, c = params
        pred = a * L**2 + b * L + c
        residuals = y - pred
        return jnp.sum(huber(residuals))

    opt = ScipyMinimize(fun=objective, method="BFGS", value_and_grad=False)
    init = jnp.array(jnp.polyfit(L, y, 2)) if len(L) >= 3 else jnp.array([0.0, *jnp.polyfit(L, y, 1)])
    return opt.run(init_params=init).params


def _fit_logistic_sigmoid(x: jnp.ndarray, y: jnp.ndarray, delta: float = 0.02):
    """Fit y ≈ bottom + (top - bottom) / (1 + exp(-k * (x - x0))).

    Uses robust Huber loss. Returns params (bottom, top, k, x0).
    """
    # Initial guesses
    bottom0 = float(jnp.percentile(y, 5.0))
    top0 = float(jnp.percentile(y, 95.0))
    x0_0 = float(jnp.median(x))
    k0 = 1.0 / float(jnp.std(x) + 1e-6)

    def huber(residual):
        abs_r = jnp.abs(residual)
        quad = 0.5 * residual**2
        linear = delta * (abs_r - 0.5 * delta)
        return jnp.where(abs_r <= delta, quad, linear)

    def predict(params, x_in):
        bottom, top, k, x0 = params
        return bottom + (top - bottom) / (1.0 + jnp.exp(-k * (x_in - x0)))

    def objective(params):
        y_hat = predict(params, x)
        return jnp.sum(huber(y - y_hat))

    opt = ScipyMinimize(fun=objective, method="BFGS", value_and_grad=False)
    init = jnp.array([bottom0, top0, k0, x0_0])
    result = opt.run(init_params=init).params
    return tuple(map(float, result))


def compute_minima_across_buckets(df: pd.DataFrame) -> list[tuple[float, float, float, str]]:
    """Return per-compute minima across all labels.

    Output list of tuples: (FLOPs, loss_at_min, tokens_at_min, label_source)
    """
    minima = []  # (label, C, N*, loss*) for every (label, C)
    if df is None or df.empty:
        return []

    for (lab, C), sub in df.groupby(["label", "flops"], as_index=False):
        sub = sub.sort_values("tokens")
        if sub.empty or len(sub) < 2:
            continue
        a, b, c = _robust_quad_logx(jnp.array(sub.tokens.values), jnp.array(sub.loss.values))
        if a == 0:
            continue
        L_opt = -b / (2 * a)
        N_star = float(10**L_opt)
        loss_opt = float(a * L_opt**2 + b * L_opt + c)
        minima.append((lab, float(C), N_star, loss_opt))

    # Aggregate by compute bucket: choose lowest loss across labels
    best_by_C: dict[float, tuple[float, float, str]] = {}
    for lab, C, N_star, loss_opt in minima:
        if C not in best_by_C or loss_opt < best_by_C[C][0]:
            best_by_C[C] = (loss_opt, N_star, lab)

    results = [(C, loss, N_star, lab) for C, (loss, N_star, lab) in best_by_C.items()]
    results.sort(key=lambda t: t[0])
    return results


def compute_to_loss_plot(
    fit_min_points: list[tuple[float, float, float, str]],
    others_min_points: list[tuple[float, float, float, str]] | None = None,
    *,
    fit_color: str = "#000000",
    other_color: str = "#A0A0A0",
) -> go.Figure:
    """Linear fit in log10(C) space using only fit_min_points. Optionally overlays other minima."""
    if not fit_min_points and not others_min_points:
        return go.Figure()

    fig = go.Figure()

    if fit_min_points:
        Cs = jnp.array([p[0] for p in fit_min_points])
        losses = jnp.array([p[1] for p in fit_min_points])

        # Fit a simple linear trend in log10(C) space: loss ~ m * log10(C) + b
        m, b = jnp.polyfit(jnp.log10(Cs), losses, 1)
        C_fit = jnp.logspace(jnp.log10(float(Cs.min())) - 0.05, jnp.log10(float(Cs.max())) + 0.05, 400)
        y_fit = m * jnp.log10(C_fit) + b

        resid = losses - (m * jnp.log10(Cs) + b)
        sigma = float(jnp.std(resid))
        upper = y_fit + sigma
        lower = y_fit - sigma

        # Confidence band (shaded)
        fig.add_trace(
            go.Scatter(
                x=list(map(float, C_fit)),
                y=list(map(float, upper)),
                mode="lines",
                line=dict(color="rgba(0,0,0,0)"),
                showlegend=False,
                hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=list(map(float, C_fit)),
                y=list(map(float, lower)),
                mode="lines",
                line=dict(color="rgba(0,0,0,0)"),
                fill="tonexty",
                fillcolor="rgba(0,0,0,0.15)",
                name="fit ±1sigma",
                hoverinfo="skip",
            )
        )

        # Fitted trend line
        fig.add_trace(
            go.Scatter(
                x=list(map(float, C_fit)),
                y=list(map(float, y_fit)),
                mode="lines",
                line=dict(color=fit_color, dash="dash", width=2),
                name="fit",
            )
        )

        # Data points at per-compute minima (fit set)
        fig.add_trace(
            go.Scatter(
                x=list(map(float, Cs)),
                y=list(map(float, losses)),
                mode="markers",
                marker=dict(symbol="x", size=12, color=fit_color),
                name="fit minima",
                hovertemplate=("C=%{x:.2e} FLOPs<br>loss=%{y:.4f}<extra></extra>"),
            )
        )

    # Overlay other minima
    if others_min_points:
        Cs_o = [float(p[0]) for p in others_min_points]
        losses_o = [float(p[1]) for p in others_min_points]
        fig.add_trace(
            go.Scatter(
                x=Cs_o,
                y=losses_o,
                mode="markers",
                marker=dict(symbol="circle", size=9, color=other_color),
                name="other minima",
            )
        )

    fig.update_layout(
        template="plotly_white",
        xaxis_type="log",
        xaxis_title="Compute FLOPs (C)",
        yaxis_title="Loss at N* (↓ better)",
        title="Compute Optimal Scaling Law in Loss",
        width=900,
        height=600,
    )

    return fig


def mmlu_sigmoid_plot(mmlu_df: pd.DataFrame) -> go.Figure:
    """Plot accuracy vs loss(-choice log-prob) with logistic regression fit.

    Fits only on rows where df.fit is True; other points are shown separately.
    """
    return sigmoid_plot_generic(
        mmlu_df,
        x_label="Paloma/c4_en/bpb",
        y_label="Accuracy (fraction)",
        color="#F0701A",
    )


def sigmoid_plot_generic(df: pd.DataFrame, x_label: str, y_label: str, color: str = "#F0701A") -> go.Figure:
    """Generic sigmoid plot with robust fit for arbitrary y metric.

    Fits only on df.fit==True if present; plots non-fit points in grey.
    """
    fig = go.Figure()
    if df is None or df.empty:
        return fig

    fit_mask = df["fit"].values if "fit" in df.columns else jnp.ones(len(df), dtype=bool)
    df_fit = df[fit_mask]
    df_other = df[~fit_mask]

    # Scatter points: fit set by label colors
    if not df_fit.empty:
        for lab, sub in df_fit.groupby("label"):
            fig.add_trace(
                go.Scatter(
                    x=list(map(float, sub["x"].values)),
                    y=list(map(float, sub["y"].values)),
                    mode="markers",
                    name=str(lab),
                    marker=dict(size=8),
                    hovertemplate=("loss(-clp)=%{x:.3f}<br>acc=%{y:.3%}<br>%{text}<extra></extra>"),
                    text=sub["name"].values,
                )
            )

    # Scatter points: other set in grey
    if not df_other.empty:
        fig.add_trace(
            go.Scatter(
                x=list(map(float, df_other["x"].values)),
                y=list(map(float, df_other["y"].values)),
                mode="markers",
                name="other",
                marker=dict(size=8, color="#A0A0A0"),
                hovertemplate=("loss(-clp)=%{x:.3f}<br>acc=%{y:.3%}<br>%{text}<extra></extra>"),
                text=df_other["name"].values,
            )
        )

    # Fit only on the fit set
    if not df_fit.empty:
        x = jnp.array(df_fit["x"].values)
        y = jnp.array(df_fit["y"].values)
        bottom, top, k, x0 = _fit_logistic_sigmoid(x, y)

        x_fit = jnp.linspace(float(x.min()) - 0.1, float(x.max()) + 0.1, 400)
        y_fit = bottom + (top - bottom) / (1.0 + jnp.exp(-k * (x_fit - x0)))

        fig.add_trace(
            go.Scatter(
                x=list(map(float, x_fit)),
                y=list(map(float, y_fit)),
                mode="lines",
                line=dict(color=color, dash="dash", width=3),
                name=f"sigmoid fit (k={float(k):.3g})",
            )
        )

    fig.update_layout(
        template="plotly_white",
        xaxis_title=x_label,
        yaxis_title=y_label,
        title="Loss Predicts Metric Sigmoidally",
        width=900,
        height=600,
        yaxis=dict(tickformat=",.0%") if (df["y"].max() <= 1.0 + 1e-6) else dict(),
    )
    return fig


def linear_plot_generic(df: pd.DataFrame, x_label: str, y_label: str, color: str = "#1f77b4") -> go.Figure:
    """Generic linear plot with y ≈ m*x + b fit for arbitrary metrics.

    Fits only on df.fit==True if present; plots non-fit points in grey.
    Intended for BPB vs BPB or other linear relationships.
    """
    fig = go.Figure()
    if df is None or df.empty:
        return fig

    fit_mask = df["fit"].values if "fit" in df.columns else jnp.ones(len(df), dtype=bool)
    df_fit = df[fit_mask]
    df_other = df[~fit_mask]

    # Scatter points: fit set by label colors
    if not df_fit.empty:
        for lab, sub in df_fit.groupby("label"):
            fig.add_trace(
                go.Scatter(
                    x=list(map(float, sub["x"].values)),
                    y=list(map(float, sub["y"].values)),
                    mode="markers",
                    name=str(lab),
                    marker=dict(size=8),
                    hovertemplate=("x=%{x:.4f}<br>y=%{y:.4f}<br>%{text}<extra></extra>"),
                    text=sub["name"].values,
                )
            )

    # Scatter points: other set in grey
    if not df_other.empty:
        fig.add_trace(
            go.Scatter(
                x=list(map(float, df_other["x"].values)),
                y=list(map(float, df_other["y"].values)),
                mode="markers",
                name="other",
                marker=dict(size=8, color="#A0A0A0"),
                hovertemplate=("x=%{x:.4f}<br>y=%{y:.4f}<br>%{text}<extra></extra>"),
                text=df_other["name"].values,
            )
        )

    # Fit only on the fit set
    if not df_fit.empty:
        x = jnp.array(df_fit["x"].values)
        y = jnp.array(df_fit["y"].values)
        # Simple linear fit y ~ m*x + b
        m, b = jnp.polyfit(x, y, 1)

        x_fit = jnp.linspace(float(x.min()), float(x.max()), 400)
        y_fit = m * x_fit + b

        fig.add_trace(
            go.Scatter(
                x=list(map(float, x_fit)),
                y=list(map(float, y_fit)),
                mode="lines",
                line=dict(color=color, dash="dash", width=3),
                name=f"linear fit (m={float(m):.3g})",
            )
        )

    fig.update_layout(
        template="plotly_white",
        xaxis_title=x_label,
        yaxis_title=y_label,
        title="Linear Relationship",
        width=900,
        height=600,
    )
    return fig


def compose_side_by_side(fig_left: go.Figure, fig_right: go.Figure, titles: tuple[str, str]) -> go.Figure:
    fig = make_subplots(rows=1, cols=2, subplot_titles=list(titles))
    for tr in fig_left.data:
        fig.add_trace(tr, row=1, col=1)
    for tr in fig_right.data:
        fig.add_trace(tr, row=1, col=2)

    # Axis configs per panel
    fig.update_xaxes(type="log", title_text="Compute FLOPs (C)", row=1, col=1)
    fig.update_yaxes(title_text="Loss at N* (↓)", row=1, col=1)
    fig.update_xaxes(title_text="Loss (- choice log-prob)", row=1, col=2)
    fig.update_yaxes(title_text="Accuracy (fraction)", tickformat=",.0%", row=1, col=2)

    fig.update_layout(template="plotly_white", width=1500, height=650)
    return fig


# ---------------- Main ----------------
def main(
    sources: list[tuple[str, str]],
    metric_key: str = DEFAULT_METRIC_KEY,
    x_keys: list[str] | None = None,
):
    """Entry point: builds and logs a compute→loss plot.

    sources: list of (ENTITY/PROJECT, REGEX_FRAGMENT). We query with
      r'isoflop.*(<fragment>)(?!-mt).*' to exclude obvious mid-train followups.
    """
    wandb.login()
    run = wandb.init(
        entity="marin-community",
        project="marin-analysis",
        job_type="compute-to-loss",
        resume="never",
        name="compute-to-loss",
    )

    api = wandb.Api()
    source_runs = []

    def _collect_runs_iteratively(entity_project: str, filters: dict, max_retries: int = 6, base_delay: float = 1.0):
        """Iterate W&B runs with pagination and retry on failures, deduping as needed."""
        seen_ids: set[str] = set()
        collected = []
        delay = base_delay
        for attempt in range(max_retries):
            try:
                # per_page keeps requests smaller; iteration paginates under the hood
                for run in api.runs(entity_project, filters=filters, per_page=200):
                    print("success")
                    uid = getattr(run, "id", getattr(run, "name", None))
                    if uid is None:
                        uid = str(id(run))
                    if uid in seen_ids:
                        continue
                    seen_ids.add(uid)
                    collected.append(run)
                break
            except Exception as e:
                if attempt == max_retries - 1:
                    raise e
                time.sleep(delay * (1.0 + 0.25 * (attempt + 1)))
                delay *= 2.0
                # retry the whole iteration; duplicates are filtered via seen_ids
                continue
        return collected

    for entity_project, fragment in sources:
        if "/" not in entity_project:
            raise ValueError(f"Bad ENTITY/PROJECT: {entity_project}")
        if not fragment:
            raise ValueError("Empty regex fragment")

        # Build a list of queries to union: isoflop-fragment and simple family names
        queries = [
            rf"isoflop.*({fragment}).*",
            r"^marin-us-central1.*",
            r"paloma-uncheatable-eval-logprobs-v2$",
        ]
        # regex = rf"isoflop.*({fragment}).*"
        # filters = {"displayName": {"$regex": regex}, "state": "finished"}
        # runs = api.runs(entity_project.strip(), filters=filters)
        combined_runs = []
        for regex in queries:
            filters = {"displayName": {"$regex": regex}, "state": "finished"}
            # runs = _collect_runs_iteratively(entity_project.strip(), filters)
            runs = api.runs(entity_project.strip(), filters=filters)
            combined_runs.extend(runs)

        # Deduplicate overlapping results from the two queries
        unique = {}
        for r in combined_runs:
            uid = getattr(r, "id", getattr(r, "name", None))
            if uid is None:
                uid = id(r)
            unique[uid] = r
        source_runs.append((list(unique.values()), fragment.strip()))

    df = df_from_sources(source_runs, metric_key=metric_key)
    minima = compute_minima_across_buckets(df)
    # Split minima into fit vs other using name-based heuristic on source DF
    fit_Cs = set(
        float(C)
        for C, sub in df.groupby("flops").groups.items()
        if any(_is_fit_run_name(n) for n in df.iloc[sub]["name"].values)
    )
    fit_min = [p for p in minima if p[0] in fit_Cs]
    other_min = [p for p in minima if p[0] not in fit_Cs]
    fig_left = compute_to_loss_plot(fit_min, others_min_points=other_min)

    # Build MMLU sigmoid plot from same sources
    mmlu_df = mmlu_df_from_sources(source_runs, key_x=PALOMA_C4EN_BPB_KEY, key_y=MMLU_ACC_KEY)
    fig_right = mmlu_sigmoid_plot(mmlu_df)

    # Build additional sigmoid/linear plots for requested metrics
    gsm8k_df = xy_df_from_sources(source_runs, key_x=PALOMA_C4EN_BPB_KEY, key_y=GSM8K_BPB_KEY)
    math500_df = xy_df_from_sources(source_runs, key_x=PALOMA_C4EN_BPB_KEY, key_y=MATH500_BPB_KEY)
    hellaswag_df = xy_df_from_sources(source_runs, key_x=PALOMA_C4EN_BPB_KEY, key_y=HELLASWAG_ACC_KEY)

    fig_gsm8k = linear_plot_generic(gsm8k_df, x_label="Paloma/c4_en/bpb", y_label="GSM8K 8-shot BPB", color="#1f77b4")
    fig_math500 = linear_plot_generic(math500_df, x_label="Paloma/c4_en/bpb", y_label="MATH500 BPB", color="#2ca02c")
    fig_hellaswag = sigmoid_plot_generic(
        hellaswag_df, x_label="Paloma/c4_en/bpb", y_label="HellaSwag 10-shot Accuracy", color="#d62728"
    )

    # Side-by-side composition
    combined = compose_side_by_side(
        fig_left, fig_right, ("Compute Optimal Scaling Law in Loss", "Loss Predicts Accuracy Sigmoidally")
    )

    log_payload = {
        "compute_to_loss": wandb.Plotly(fig_left),
        "mmlu/base": wandb.Plotly(fig_right),
        "gsm8k/base": wandb.Plotly(fig_gsm8k),
        "math500/base": wandb.Plotly(fig_math500),
        "hellaswag/base": wandb.Plotly(fig_hellaswag),
        "compute_and_mmlu_side_by_side": wandb.Plotly(combined),
    }

    # Optionally, iterate multiple x_keys for each y metric and log separate figures
    if x_keys:
        color_cycle = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
        for idx, xk in enumerate(x_keys):
            color = color_cycle[idx % len(color_cycle)]

            # MMLU (sigmoid)
            mmlu_df_x = xy_df_from_sources(source_runs, key_x=xk, key_y=MMLU_ACC_KEY)
            fig_mmlu_x = sigmoid_plot_generic(mmlu_df_x, x_label=xk, y_label="Accuracy (fraction)", color=color)
            log_payload[f"mmlu/{_metric_slug(xk)}"] = wandb.Plotly(fig_mmlu_x)

            # GSM8K (linear)
            gsm8k_df_x = xy_df_from_sources(source_runs, key_x=xk, key_y=GSM8K_BPB_KEY)
            fig_gsm8k_x = linear_plot_generic(gsm8k_df_x, x_label=xk, y_label="GSM8K 8-shot BPB", color=color)
            log_payload[f"gsm8k/{_metric_slug(xk)}"] = wandb.Plotly(fig_gsm8k_x)

            # MATH500 (linear)
            math500_df_x = xy_df_from_sources(source_runs, key_x=xk, key_y=MATH500_BPB_KEY)
            fig_math500_x = linear_plot_generic(math500_df_x, x_label=xk, y_label="MATH500 BPB", color=color)
            log_payload[f"math500/{_metric_slug(xk)}"] = wandb.Plotly(fig_math500_x)

            # HellaSwag (sigmoid)
            hellaswag_df_x = xy_df_from_sources(source_runs, key_x=xk, key_y=HELLASWAG_ACC_KEY)
            fig_hellaswag_x = sigmoid_plot_generic(
                hellaswag_df_x, x_label=xk, y_label="HellaSwag 10-shot Accuracy", color=color
            )
            log_payload[f"hellaswag/{_metric_slug(xk)}"] = wandb.Plotly(fig_hellaswag_x)

    wandb.log(log_payload)
    run.finish()


if __name__ == "__main__":
    SOURCES = [
        ("marin-community/marin", "nemo-wider-depth-adapt"),
        # Add other sources as needed
    ]
    main(SOURCES, metric_key=DEFAULT_METRIC_KEY, x_keys=[PALOMA_BPB_KEY, PALOMA_C4EN_BPB_KEY, UNCHEATABLE_EVAL_BPB_KEY])
