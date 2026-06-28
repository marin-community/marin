# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo",
#     "matplotlib",
#     "numpy",
#     "pandas",
#     "plotly",
#     "scikit-learn",
#     "scipy",
# ]
# ///
"""Marimo notebook for Pareto-aware issue #5416 DSP optimization."""

import marimo

__generated_with = "0.23.5"
app = marimo.App(width="full")


@app.cell
def _():
    import json
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from dataclasses import asdict
    from pathlib import Path

    import marimo as mo
    import matplotlib
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go
    from scipy.optimize import minimize
    from scipy.special import logsumexp
    from scipy.stats import pearsonr, spearmanr

    matplotlib.use("Agg")
    matplotlib.rcParams["text.usetex"] = False

    from experiments.domain_phase_mix.exploratory.two_phase_many.fit_dsp_canonical_variants_300m import (
        VARIANTS as DSP_VARIANTS,
    )
    from experiments.domain_phase_mix.exploratory.two_phase_many.fit_dsp_canonical_variants_300m import (
        FittedDSPModel,
    )
    from experiments.domain_phase_mix.exploratory.two_phase_many.fit_dsp_canonical_variants_300m import (
        _fit_variant as fit_dsp_variant,
    )
    from experiments.domain_phase_mix.exploratory.two_phase_many.fit_dsp_canonical_variants_300m import (
        _oof_predictions as dsp_oof_predictions,
    )

    return (
        DSP_VARIANTS,
        FittedDSPModel,
        Path,
        ThreadPoolExecutor,
        as_completed,
        asdict,
        dsp_oof_predictions,
        fit_dsp_variant,
        go,
        json,
        logsumexp,
        minimize,
        mo,
        np,
        pd,
        pearsonr,
        plt,
        px,
        spearmanr,
    )


@app.cell
def _():
    from experiments.domain_phase_mix.exploratory.two_phase_many.fit_dsp_canonical_variants_300m import (
        _packet_from_frame as packet_from_frame,
    )
    from experiments.domain_phase_mix.exploratory.two_phase_many.fit_dsp_canonical_variants_300m import (
        _predict as predict_dsp_loss,
    )
    from experiments.domain_phase_mix.exploratory.two_phase_many.fit_dsp_canonical_variants_300m import (
        _value_grad_logits as dsp_value_grad_logits,
    )
    from experiments.domain_phase_mix.exploratory.two_phase_many.metric_registry.issue5416_aggregate import (
        fit_issue5416_projection,
        score_issue5416_aggregate,
        write_issue5416_projection,
    )
    from experiments.domain_phase_mix.exploratory.two_phase_many.surrogate_search import (
        plot_grp_vs_proportional as reference_plot,
    )
    from experiments.domain_phase_mix.static_batch_selection import average_phase_tv_distance

    return (
        average_phase_tv_distance,
        dsp_value_grad_logits,
        fit_issue5416_projection,
        packet_from_frame,
        predict_dsp_loss,
        reference_plot,
        score_issue5416_aggregate,
        write_issue5416_projection,
    )


@app.cell
def _(Path):
    TWO_PHASE_ROOT = Path(__file__).resolve().parent
    MATRIX_DIR = TWO_PHASE_ROOT / "metric_registry" / "raw_metric_matrix_300m"
    SIGNAL_CSV = MATRIX_DIR / "raw_metric_matrix_300m.csv"
    VARIABLE_NOISE_CSV = MATRIX_DIR / "noise_baseline_run00097_variable_subset_300m.csv"
    OUTPUT_DIR = TWO_PHASE_ROOT / "reference_outputs" / "pareto_aware_effective_exposure_issue5416_20260511"
    MODEL_CACHE_DIR = OUTPUT_DIR / "model_cache"
    IMG_DIR = OUTPUT_DIR / "img"
    EXISTING_AGGREGATE_MODEL_JSON = (
        TWO_PHASE_ROOT
        / "reference_outputs"
        / "dsp_issue5416_aggregate_300m_20260510"
        / "dsp_effective_exposure_penalty_nnls"
        / "params.json"
    )
    VARIANT_NAME = "dsp_effective_exposure_penalty_nnls"
    PROPORTIONAL_RUN = "baseline_proportional"
    RNG_SEED = 5416
    DEPLOYABLE_MAX_NEAREST_TV = 0.35
    DEPLOYABLE_MAX_GROUP_REGRESSION_NOISE = -1.0
    return (
        DEPLOYABLE_MAX_GROUP_REGRESSION_NOISE,
        DEPLOYABLE_MAX_NEAREST_TV,
        EXISTING_AGGREGATE_MODEL_JSON,
        IMG_DIR,
        MODEL_CACHE_DIR,
        OUTPUT_DIR,
        PROPORTIONAL_RUN,
        RNG_SEED,
        SIGNAL_CSV,
        VARIABLE_NOISE_CSV,
        VARIANT_NAME,
    )


@app.cell
def _(mo):
    mo.md(
        r"""
    # Pareto-aware effective-exposure DSP optimization for issue #5416

    This notebook uses the `dsp_effective_exposure_penalty_nnls` form as a
    local optimization surrogate for the current issue #5416 task aggregate.
    The aggregate-only optimum is useful but can hide per-task regressions.
    The Pareto-aware objectives below add item- and group-level safeguards
    around the same surrogate.

    ## Canonical DSP form

    We now treat `dsp_effective_exposure_penalty_nnls` as the canonical DSP
    form. For domain $i$ and two training phases, define the exposure-normalized
    phase masses $e_{0i}=c_{0i}w_{0i}$ and $e_{1i}=c_{1i}w_{1i}$, then use one
    global phase-1 exposure multiplier:

    $$
    z_i(w)=e_{0i}+\gamma e_{1i}.
    $$

    The fitted scalar target is:

    $$
    \hat y(w)=\beta_0
    -\sum_{i=1}^{M} a_i\left(1-\exp(-\rho_i z_i(w))\right)
    +\sum_{i=1}^{M} p_i\,\operatorname{softplus}\!\left(\log(1+z_i(w))-\tau_i\right)^2.
    $$

    The NNLS head enforces $a_i\ge0$ and $p_i\ge0$; the nonlinear fit enforces
    $\rho_i>0$ and $\gamma>0$. The per-domain parameters are
    $(\rho_i,\tau_i,a_i,p_i)$, plus global $(\beta_0,\gamma)$, so the 39-domain
    model has $4M+2=158$ parameters. For BPB/loss targets, lower $\hat y$ is
    better; this notebook orients issue #5416 scores so reported candidate
    deltas are positive when helpful.

    Sign convention:

    | Quantity | Meaning | Helpful Direction |
    | :--- | :--- | :--- |
    | issue #5416 aggregate | fixed factor score from selected task proxies | positive |
    | item delta | predicted item score minus proportional item score, in item z-units | positive |
    | noise-standardized delta | item or group delta divided by variable-subset noise std | positive |
    | nearest observed TV | average phase-TV distance from candidate to nearest signal-row mixture | lower |
    """
    )
    return


@app.cell
def _(
    DSP_VARIANTS,
    EXISTING_AGGREGATE_MODEL_JSON,
    FittedDSPModel,
    IMG_DIR,
    MODEL_CACHE_DIR,
    OUTPUT_DIR,
    PROPORTIONAL_RUN,
    SIGNAL_CSV,
    ThreadPoolExecutor,
    VARIABLE_NOISE_CSV,
    VARIANT_NAME,
    as_completed,
    asdict,
    dsp_oof_predictions,
    fit_dsp_variant,
    fit_issue5416_projection,
    json,
    np,
    packet_from_frame,
    pd,
    pearsonr,
    score_issue5416_aggregate,
    spearmanr,
    write_issue5416_projection,
):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    IMG_DIR.mkdir(parents=True, exist_ok=True)

    variant_by_name = {variant.name: variant for variant in DSP_VARIANTS}
    if VARIANT_NAME not in variant_by_name:
        raise ValueError(f"Unknown DSP variant: {VARIANT_NAME}")
    effective_variant = variant_by_name[VARIANT_NAME]

    signal = pd.read_csv(SIGNAL_CSV, low_memory=False)
    variable_noise = pd.read_csv(VARIABLE_NOISE_CSV, low_memory=False)
    if len(signal) != 242:
        raise ValueError(f"Expected 242 signal rows, found {len(signal)}")
    if len(variable_noise) != 10:
        raise ValueError(f"Expected 10 variable-subset noise rows, found {len(variable_noise)}")
    proportional_rows = signal.loc[signal["run_name"].eq(PROPORTIONAL_RUN)]
    if len(proportional_rows) != 1:
        raise ValueError(f"Expected exactly one {PROPORTIONAL_RUN} row, found {len(proportional_rows)}")
    proportional_idx = int(proportional_rows.index[0])

    projection = fit_issue5416_projection(signal_frame=signal, noise_frame=variable_noise)
    write_issue5416_projection(projection, OUTPUT_DIR / "issue5416_projection.json")
    issue5416_scores = score_issue5416_aggregate(signal, projection, fail_missing=True)
    if issue5416_scores.isna().any():
        raise ValueError("Issue #5416 aggregate has missing signal scores")

    task_columns = tuple(projection.task_columns)
    task_signs = np.asarray(projection.task_signs, dtype=float)
    task_means = np.asarray(projection.means, dtype=float)
    task_stds = np.asarray(projection.stds, dtype=float)
    raw_task_values = signal.loc[:, list(task_columns)].to_numpy(dtype=float)
    item_z = (raw_task_values * task_signs[None, :] - task_means[None, :]) / task_stds[None, :]
    if not np.isfinite(item_z).all():
        raise ValueError("Selected issue #5416 item z-scores are not finite on signal rows")

    noise_values = variable_noise.loc[:, list(task_columns)].to_numpy(dtype=float)
    noise_z = (noise_values * task_signs[None, :] - task_means[None, :]) / task_stds[None, :]
    item_noise_std = noise_z.std(axis=0, ddof=1)
    signal_item_std = item_z.std(axis=0, ddof=1)
    fallback_mask = ~np.isfinite(item_noise_std) | (item_noise_std <= 1e-12)
    item_noise_std = np.where(fallback_mask, signal_item_std, item_noise_std)
    fallback_items = [column for column, used in zip(task_columns, fallback_mask, strict=True) if bool(used)]

    def build_packet_for_loss(loss_values: np.ndarray, name: str):
        frame = signal.copy()
        frame["objective_metric"] = np.asarray(loss_values, dtype=float)
        return packet_from_frame(frame, name=name).base

    aggregate_packet = build_packet_for_loss(-issue5416_scores.to_numpy(dtype=float), "issue5416_aggregate_loss")

    def cache_name(target_id: str) -> str:
        cleaned = target_id.replace("/", "__").replace(" ", "_").replace(":", "_")
        return f"{cleaned}.json"

    def model_from_payload(path):
        payload = json.loads(path.read_text())
        params = {}
        for key, value in payload["params"].items():
            if isinstance(value, list):
                params[key] = np.asarray(value, dtype=float)
            else:
                params[key] = float(value)
        return FittedDSPModel(
            variant=effective_variant,
            params=params,
            intercept=float(payload["intercept"]),
            benefit_coef=np.asarray(payload["benefit_coef"], dtype=float),
            penalty_coef=np.asarray(payload["penalty_coef"], dtype=float),
        )

    def write_model_payload(model, path, target_id: str, trace: pd.DataFrame | None):
        payload = {
            "target_id": target_id,
            "variant": asdict(model.variant),
            "params": {
                key: value.tolist() if isinstance(value, np.ndarray) else float(value)
                for key, value in model.params.items()
            },
            "intercept": float(model.intercept),
            "benefit_coef": model.benefit_coef.tolist(),
            "penalty_coef": model.penalty_coef.tolist(),
        }
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        if trace is not None:
            trace.to_csv(path.with_suffix(".fit_trace.csv"), index=False)

    def fit_or_load_model(target_id: str, loss_values: np.ndarray):
        cache_path = MODEL_CACHE_DIR / cache_name(target_id)
        if cache_path.exists():
            model = model_from_payload(cache_path)
            cache_status = "loaded"
        elif target_id == "issue5416_aggregate" and EXISTING_AGGREGATE_MODEL_JSON.exists():
            model = model_from_payload(EXISTING_AGGREGATE_MODEL_JSON)
            write_model_payload(model, cache_path, target_id, trace=None)
            cache_status = "copied_existing"
        else:
            packet = build_packet_for_loss(loss_values, f"pareto_{target_id}")
            model, trace = fit_dsp_variant(packet, effective_variant)
            write_model_payload(model, cache_path, target_id, trace=trace)
            cache_status = "fit"
        packet = build_packet_for_loss(loss_values, f"pareto_{target_id}")
        oof_loss = dsp_oof_predictions(packet, model)
        score_actual = -loss_values
        score_oof = -oof_loss
        residual = score_oof - score_actual
        metrics = {
            "target_id": target_id,
            "cache_status": cache_status,
            "score_oof_rmse": float(np.sqrt(np.mean(residual**2))),
            "score_oof_mae": float(np.mean(np.abs(residual))),
            "score_oof_spearman": float(spearmanr(score_actual, score_oof).statistic),
            "score_oof_pearson": float(pearsonr(score_actual, score_oof).statistic),
            "target_score_std": float(np.std(score_actual, ddof=1)),
            "low_confidence_item_surrogate": bool(
                target_id != "issue5416_aggregate"
                and (
                    spearmanr(score_actual, score_oof).statistic < 0.25
                    or np.sqrt(np.mean(residual**2)) > np.std(score_actual, ddof=1)
                )
            ),
        }
        return model, packet, metrics

    aggregate_model, aggregate_packet, aggregate_model_metrics = fit_or_load_model(
        "issue5416_aggregate", -issue5416_scores.to_numpy(dtype=float)
    )

    def fit_item_model_task(task: tuple[int, str]):
        task_item_idx, task_column = task
        task_loss = -item_z[:, task_item_idx]
        task_model, task_packet, task_metrics = fit_or_load_model(f"item__{task_column}", task_loss)
        task_metrics["task_column"] = task_column
        task_metrics["item_noise_std_z"] = float(item_noise_std[task_item_idx])
        task_metrics["used_noise_fallback"] = bool(fallback_mask[task_item_idx])
        return task_item_idx, task_model, task_packet, task_metrics

    item_models = [None] * len(task_columns)
    item_packets = [None] * len(task_columns)
    model_metric_rows = [aggregate_model_metrics]
    with ThreadPoolExecutor(max_workers=6) as executor:
        _item_futures = [
            executor.submit(fit_item_model_task, (_fit_item_idx, _fit_column))
            for _fit_item_idx, _fit_column in enumerate(task_columns)
        ]
        for _item_future in as_completed(_item_futures):
            _fit_item_idx, _item_model, _item_packet, _item_metrics = _item_future.result()
            item_models[_fit_item_idx] = _item_model
            item_packets[_fit_item_idx] = _item_packet
            model_metric_rows.append(_item_metrics)
    if any(_item_model is None for _item_model in item_models) or any(
        _item_packet is None for _item_packet in item_packets
    ):
        raise RuntimeError("At least one item DSP model failed to populate")

    model_metrics = pd.DataFrame.from_records(model_metric_rows)
    model_metrics.to_csv(OUTPUT_DIR / "model_fit_metrics.csv", index=False)
    target_scores = signal.loc[:, ["run_name", "is_qsplit240_core"]].copy()
    target_scores["issue5416_aggregate"] = issue5416_scores.to_numpy(dtype=float)
    for _score_item_idx, _score_column in enumerate(task_columns):
        target_scores[f"item_z/{_score_column}"] = item_z[:, _score_item_idx]
    target_scores.to_csv(OUTPUT_DIR / "target_scores.csv", index=False)

    mo_fallback_note = (
        "No selected item needed signal-std fallback for noise scaling."
        if not fallback_items
        else f"Noise fallback used for {len(fallback_items)} selected items: {', '.join(fallback_items)}"
    )
    return (
        aggregate_model,
        aggregate_packet,
        effective_variant,
        item_models,
        item_noise_std,
        item_packets,
        mo_fallback_note,
        model_metrics,
        projection,
        proportional_idx,
        signal,
        task_columns,
        variable_noise,
    )


@app.cell
def _(mo, mo_fallback_note, model_metrics, projection, signal, variable_noise):
    metric_view = model_metrics.loc[
        :,
        [
            "target_id",
            "cache_status",
            "score_oof_rmse",
            "score_oof_spearman",
            "score_oof_pearson",
            "low_confidence_item_surrogate",
        ],
    ].sort_values("score_oof_spearman")
    mo.vstack(
        [
            mo.md(
                f"""
                ## Data and model coverage

                | Check | Value |
                | :--- | ---: |
                | signal rows | {len(signal)} |
                | variable-subset noise rows | {len(variable_noise)} |
                | selected issue #5416 items | {len(projection.task_columns)} |
                | Horn-selected factors | {projection.factor_count} |

                {mo_fallback_note}
                """
            ),
            mo.ui.table(metric_view, page_size=12),
        ]
    )
    return


@app.cell
def _(np, pd, task_columns):
    def item_group(column: str) -> str:
        if column.startswith("eval/uncheatable_eval/"):
            return "uncheatable"
        if "mmlu_sl_verb" in column:
            return "mmlu_sl"
        if any(token in column for token in ("arc_", "openbookqa", "sciq", "medmcqa")):
            return "arc_openbook_sciq"
        if "hellaswag" in column or "swag" in column:
            return "hellaswag_swag"
        if "truthfulqa" in column:
            return "truthfulqa"
        if "gsm8k" in column:
            return "gsm8k_smooth"
        if "humaneval" in column:
            return "humaneval_smooth"
        if "agentic_coding" in column:
            return "agentic_coding"
        if any(token in column for token in ("boolq", "copa", "csqa", "piqa", "winogrande", "wsc273")):
            return "commonsense"
        return "other_task"

    item_groups = pd.Series([item_group(column) for column in task_columns], index=task_columns, name="group")
    group_order = [
        "uncheatable",
        "mmlu_sl",
        "arc_openbook_sciq",
        "hellaswag_swag",
        "commonsense",
        "truthfulqa",
        "gsm8k_smooth",
        "humaneval_smooth",
        "agentic_coding",
        "other_task",
    ]
    present_groups = [group for group in group_order if group in set(item_groups)]
    group_indices = {group: np.flatnonzero(item_groups.to_numpy() == group) for group in present_groups}
    return group_indices, item_groups


@app.cell
def _(
    OUTPUT_DIR,
    RNG_SEED,
    aggregate_model,
    aggregate_packet,
    dsp_value_grad_logits,
    group_indices,
    item_models,
    item_noise_std,
    item_packets,
    logsumexp,
    minimize,
    np,
    pd,
    predict_dsp_loss,
    proportional_idx,
):
    def weights_from_logits(logits: np.ndarray) -> np.ndarray:
        num_domains = aggregate_packet.m
        p0_logits = logits[:num_domains]
        p1_logits = logits[num_domains:]
        p0 = np.exp(p0_logits - np.max(p0_logits))
        p0 /= p0.sum()
        p1 = np.exp(p1_logits - np.max(p1_logits))
        p1 /= p1.sum()
        return np.stack([p0, p1], axis=0)

    def logits_from_weights(weights: np.ndarray) -> np.ndarray:
        clipped = np.clip(np.asarray(weights, dtype=float), 1e-12, None)
        return np.concatenate([np.log(clipped[0]), np.log(clipped[1])])

    def model_score_grad(model, packet, logits: np.ndarray) -> tuple[float, np.ndarray]:
        loss, loss_grad = dsp_value_grad_logits(model, packet, logits)
        return -float(loss), -np.asarray(loss_grad, dtype=float)

    proportional_weights = aggregate_packet.w[proportional_idx]
    proportional_logits = logits_from_weights(proportional_weights)
    proportional_aggregate_score, _ = model_score_grad(aggregate_model, aggregate_packet, proportional_logits)
    proportional_item_scores = []
    for _baseline_model, _baseline_packet in zip(item_models, item_packets, strict=True):
        _baseline_score, _baseline_grad = model_score_grad(_baseline_model, _baseline_packet, proportional_logits)
        proportional_item_scores.append(_baseline_score)
    proportional_item_scores = np.asarray(proportional_item_scores, dtype=float)
    item_noise_scale = np.maximum(item_noise_std, 1e-6)

    observed_scores = -predict_dsp_loss(aggregate_model, aggregate_packet.w, aggregate_packet)
    observed_ranked = np.argsort(-observed_scores)
    rng = np.random.default_rng(RNG_SEED)
    starts = [proportional_logits]
    starts.extend(logits_from_weights(aggregate_packet.w[int(idx)]) for idx in observed_ranked[:5])
    for scale in (0.10, 0.25):
        for _ in range(4):
            starts.append(proportional_logits + rng.normal(scale=scale, size=2 * aggregate_packet.m))

    def component_values(logits: np.ndarray):
        agg_score, agg_grad = model_score_grad(aggregate_model, aggregate_packet, logits)
        item_scores = np.empty(len(item_models), dtype=float)
        item_grads = np.empty((len(item_models), len(logits)), dtype=float)
        for _component_idx, (_component_model, _component_packet) in enumerate(
            zip(item_models, item_packets, strict=True)
        ):
            item_scores[_component_idx], item_grads[_component_idx] = model_score_grad(
                _component_model, _component_packet, logits
            )
        item_delta = item_scores - proportional_item_scores
        item_delta_noise = item_delta / item_noise_scale
        item_delta_noise_grads = item_grads / item_noise_scale[:, None]
        group_delta_noise = np.asarray([item_delta_noise[indices].mean() for indices in group_indices.values()])
        group_delta_noise_grads = np.asarray(
            [item_delta_noise_grads[indices].mean(axis=0) for indices in group_indices.values()]
        )
        return {
            "aggregate_score": agg_score,
            "aggregate_gain": agg_score - proportional_aggregate_score,
            "aggregate_grad": agg_grad,
            "item_delta": item_delta,
            "item_delta_noise": item_delta_noise,
            "item_delta_noise_grads": item_delta_noise_grads,
            "group_delta_noise": group_delta_noise,
            "group_delta_noise_grads": group_delta_noise_grads,
        }

    def softmin(values: np.ndarray, grads: np.ndarray, temperature: float) -> tuple[float, np.ndarray]:
        weights = np.exp(-values / temperature - logsumexp(-values / temperature))
        value = -temperature * logsumexp(-values / temperature)
        grad = np.sum(weights[:, None] * grads, axis=0)
        return float(value), grad

    def lower_tail_mean(values: np.ndarray, grads: np.ndarray, frac: float) -> tuple[float, np.ndarray]:
        count = max(1, int(np.ceil(frac * len(values))))
        indices = np.argsort(values)[:count]
        return float(values[indices].mean()), grads[indices].mean(axis=0)

    def objective_value_grad(name: str, logits: np.ndarray) -> tuple[float, np.ndarray]:
        components = component_values(logits)
        aggregate_score = float(components["aggregate_score"])
        aggregate_grad = np.asarray(components["aggregate_grad"], dtype=float)
        item_delta_noise = np.asarray(components["item_delta_noise"], dtype=float)
        item_delta_noise_grads = np.asarray(components["item_delta_noise_grads"], dtype=float)
        group_delta_noise = np.asarray(components["group_delta_noise"], dtype=float)
        group_delta_noise_grads = np.asarray(components["group_delta_noise_grads"], dtype=float)
        if name == "aggregate_only":
            return aggregate_score, aggregate_grad
        if name == "hard_item_guardrail":
            violations = np.minimum(item_delta_noise + 0.25, 0.0)
            value = aggregate_score - 0.20 * float(np.sum(violations**2))
            grad = aggregate_grad - 0.40 * np.sum((violations[:, None] * item_delta_noise_grads), axis=0)
            return float(value), grad
        if name == "hard_group_guardrail":
            violations = np.minimum(group_delta_noise + 0.25, 0.0)
            value = aggregate_score - 0.35 * float(np.sum(violations**2))
            grad = aggregate_grad - 0.70 * np.sum((violations[:, None] * group_delta_noise_grads), axis=0)
            return float(value), grad
        if name == "group_dro":
            tail_value, tail_grad = softmin(group_delta_noise, group_delta_noise_grads, temperature=0.25)
            return float(tail_value + 0.05 * components["aggregate_gain"]), tail_grad + 0.05 * aggregate_grad
        if name == "item_cvar25":
            tail_value, tail_grad = lower_tail_mean(item_delta_noise, item_delta_noise_grads, frac=0.25)
            return float(components["aggregate_gain"] + 0.35 * tail_value), aggregate_grad + 0.35 * tail_grad
        if name == "item_maximin":
            tail_value, tail_grad = softmin(item_delta_noise, item_delta_noise_grads, temperature=0.20)
            return float(tail_value + 0.02 * components["aggregate_gain"]), tail_grad + 0.02 * aggregate_grad
        if name == "mean_plus_tail_penalty":
            negative = np.minimum(item_delta_noise, 0.0)
            value = aggregate_score - 0.15 * float(np.mean(negative**2))
            grad = aggregate_grad - (0.30 / len(negative)) * np.sum(negative[:, None] * item_delta_noise_grads, axis=0)
            return float(value), grad
        raise ValueError(f"Unknown objective: {name}")

    objective_names = [
        "aggregate_only",
        "hard_item_guardrail",
        "hard_group_guardrail",
        "group_dro",
        "item_cvar25",
        "item_maximin",
        "mean_plus_tail_penalty",
    ]
    optimization_rows = []
    candidate_logits = {}
    for _objective_name in objective_names:
        best_result = None
        for start_id, start in enumerate(starts):
            result = minimize(
                lambda logits, name=_objective_name: -objective_value_grad(name, np.asarray(logits, dtype=float))[0],
                start,
                jac=lambda logits, name=_objective_name: -objective_value_grad(name, np.asarray(logits, dtype=float))[1],
                method="L-BFGS-B",
                options={"maxiter": 300, "ftol": 1e-8, "maxls": 20},
            )
            optimization_rows.append(
                {
                    "objective": _objective_name,
                    "start_id": start_id,
                    "success": bool(result.success),
                    "fun": float(result.fun),
                    "message": str(result.message),
                }
            )
            if best_result is None or float(result.fun) < float(best_result.fun):
                best_result = result
        if best_result is None:
            raise RuntimeError(f"No optimization result for {_objective_name}")
        candidate_logits[_objective_name] = np.asarray(best_result.x, dtype=float)

    optimization_trace = pd.DataFrame.from_records(optimization_rows)
    optimization_trace.to_csv(OUTPUT_DIR / "optimization_trace.csv", index=False)
    return (
        candidate_logits,
        component_values,
        objective_names,
        proportional_weights,
        weights_from_logits,
    )


@app.cell
def _(
    DEPLOYABLE_MAX_GROUP_REGRESSION_NOISE,
    DEPLOYABLE_MAX_NEAREST_TV,
    OUTPUT_DIR,
    aggregate_packet,
    average_phase_tv_distance,
    candidate_logits,
    component_values,
    group_indices,
    item_groups,
    item_noise_std,
    np,
    objective_names,
    pd,
    proportional_weights,
    signal,
    task_columns,
    weights_from_logits,
):
    def phase_entropy(weights: np.ndarray) -> float:
        positive = weights[weights > 0.0]
        return float(-np.sum(positive * np.log(positive)))

    candidate_summary_rows = []
    candidate_weight_rows = []
    item_delta_rows = []
    group_delta_rows = []
    for _candidate_name in objective_names:
        weights = weights_from_logits(candidate_logits[_candidate_name])
        components = component_values(candidate_logits[_candidate_name])
        distances = average_phase_tv_distance(aggregate_packet.w, weights[None, :, :])
        nearest_idx = int(np.argmin(distances))
        item_delta_noise = components["item_delta_noise"]
        group_delta_noise = components["group_delta_noise"]
        improved_items = int(np.sum(item_delta_noise > 0.0))
        improved_groups = int(np.sum(group_delta_noise > 0.0))
        phase0_support = int(np.sum(weights[0] > 1e-3))
        phase1_support = int(np.sum(weights[1] > 1e-3))
        deployable = (
            components["aggregate_gain"] > 0.0
            and float(np.min(group_delta_noise)) >= DEPLOYABLE_MAX_GROUP_REGRESSION_NOISE
            and float(distances[nearest_idx]) <= DEPLOYABLE_MAX_NEAREST_TV
            and phase0_support >= 6
            and phase1_support >= 6
            and max(float(weights[0].max()), float(weights[1].max())) <= 0.70
        )
        candidate_summary_rows.append(
            {
                "candidate": _candidate_name,
                "predicted_aggregate_score": float(components["aggregate_score"]),
                "predicted_aggregate_gain_vs_proportional": float(components["aggregate_gain"]),
                "min_item_delta_noise": float(np.min(item_delta_noise)),
                "min_group_delta_noise": float(np.min(group_delta_noise)),
                "num_improved_items": improved_items,
                "num_improved_groups": improved_groups,
                "worst_regressed_item": task_columns[int(np.argmin(item_delta_noise))],
                "worst_regressed_item_delta_noise": float(np.min(item_delta_noise)),
                "worst_regressed_group": list(group_indices.keys())[int(np.argmin(group_delta_noise))],
                "worst_regressed_group_delta_noise": float(np.min(group_delta_noise)),
                "tv_from_proportional": float(0.5 * np.abs(weights - proportional_weights).sum(axis=1).mean()),
                "nearest_observed_run": str(signal.iloc[nearest_idx]["run_name"]),
                "nearest_observed_tv": float(distances[nearest_idx]),
                "phase0_support_gt_1e3": phase0_support,
                "phase1_support_gt_1e3": phase1_support,
                "phase0_entropy": phase_entropy(weights[0]),
                "phase1_entropy": phase_entropy(weights[1]),
                "phase0_max_weight": float(weights[0].max()),
                "phase1_max_weight": float(weights[1].max()),
                "deployability_flag": bool(deployable),
            }
        )
        for phase_idx, phase_name in enumerate(("phase_0", "phase_1")):
            for domain_name, weight, proportional_weight, multiplier in zip(
                aggregate_packet.domain_names,
                weights[phase_idx],
                proportional_weights[phase_idx],
                aggregate_packet.c0 if phase_idx == 0 else aggregate_packet.c1,
                strict=True,
            ):
                candidate_weight_rows.append(
                    {
                        "candidate": _candidate_name,
                        "domain": domain_name,
                        "phase": phase_name,
                        "weight": float(weight),
                        "proportional_weight": float(proportional_weight),
                        "weight_delta": float(weight - proportional_weight),
                        "effective_epochs": float(weight * multiplier),
                    }
                )
        for _delta_item_idx, _delta_column in enumerate(task_columns):
            item_delta_rows.append(
                {
                    "candidate": _candidate_name,
                    "task_column": _delta_column,
                    "group": str(item_groups.loc[_delta_column]),
                    "item_delta_z": float(components["item_delta"][_delta_item_idx]),
                    "item_noise_std_z": float(item_noise_std[_delta_item_idx]),
                    "item_delta_noise": float(item_delta_noise[_delta_item_idx]),
                }
            )
        for group_name, group_delta in zip(group_indices.keys(), group_delta_noise, strict=True):
            group_delta_rows.append(
                {
                    "candidate": _candidate_name,
                    "group": group_name,
                    "group_delta_noise": float(group_delta),
                }
            )

    candidate_summary = pd.DataFrame.from_records(candidate_summary_rows)
    candidate_weights_long = pd.DataFrame.from_records(candidate_weight_rows)
    candidate_item_deltas = pd.DataFrame.from_records(item_delta_rows)
    candidate_group_deltas = pd.DataFrame.from_records(group_delta_rows)
    candidate_summary.to_csv(OUTPUT_DIR / "candidate_summary.csv", index=False)
    candidate_weights_long.to_csv(OUTPUT_DIR / "candidate_weights_long.csv", index=False)
    candidate_item_deltas.to_csv(OUTPUT_DIR / "candidate_item_deltas.csv", index=False)
    candidate_group_deltas.to_csv(OUTPUT_DIR / "candidate_group_deltas.csv", index=False)
    return (
        candidate_group_deltas,
        candidate_item_deltas,
        candidate_summary,
        candidate_weights_long,
    )


@app.cell
def _(candidate_summary, mo):
    display_cols = [
        "candidate",
        "predicted_aggregate_gain_vs_proportional",
        "min_item_delta_noise",
        "min_group_delta_noise",
        "num_improved_items",
        "num_improved_groups",
        "worst_regressed_group",
        "nearest_observed_tv",
        "phase0_max_weight",
        "phase1_max_weight",
        "deployability_flag",
    ]
    mo.vstack(
        [
            mo.md("## Candidate summary"),
            mo.ui.table(
                candidate_summary.loc[:, display_cols].sort_values(
                    ["deployability_flag", "predicted_aggregate_gain_vs_proportional"],
                    ascending=[False, False],
                ),
                page_size=10,
            ),
        ]
    )
    return


@app.cell
def _(IMG_DIR, candidate_summary, go, mo):
    frontier = go.Figure()
    frontier.add_trace(
        go.Scatter(
            x=candidate_summary["min_group_delta_noise"],
            y=candidate_summary["predicted_aggregate_gain_vs_proportional"],
            mode="markers+text",
            text=candidate_summary["candidate"],
            textposition="top center",
            marker={
                "size": 14,
                "color": candidate_summary["nearest_observed_tv"],
                "colorscale": "RdYlGn_r",
                "colorbar": {"title": "Nearest observed TV"},
                "line": {"color": "#0f172a", "width": 0.8},
            },
            hovertemplate=(
                "<b>%{text}</b><br>"
                "min group delta: %{x:.3f} noise std<br>"
                "aggregate gain: %{y:.4f}<br>"
                "nearest TV: %{marker.color:.3f}<extra></extra>"
            ),
        )
    )
    frontier.add_vline(x=0.0, line_dash="dash", line_color="#64748b")
    frontier.add_vline(x=-1.0, line_dash="dot", line_color="#dc2626")
    frontier.add_hline(y=0.0, line_dash="dash", line_color="#64748b")
    frontier.update_layout(
        template="plotly_white",
        title="Predicted Pareto frontier: aggregate gain vs worst group delta",
        xaxis_title="Worst group delta vs proportional (variable-noise std units)",
        yaxis_title="Predicted issue #5416 aggregate gain vs proportional",
        height=620,
    )
    frontier.write_html(IMG_DIR / "objective_frontier.html", include_plotlyjs="cdn")
    mo.ui.plotly(frontier)
    return


@app.cell
def _(IMG_DIR, candidate_item_deltas, mo, px):
    item_heat = candidate_item_deltas.copy()
    item_heat["short_item"] = item_heat["task_column"].str.replace("eval/uncheatable_eval/", "uncheatable/", regex=False)
    heatmap = px.imshow(
        item_heat.pivot(index="short_item", columns="candidate", values="item_delta_noise"),
        color_continuous_scale="RdYlGn",
        zmin=-3,
        zmax=3,
        aspect="auto",
        title="Predicted item deltas vs proportional (variable-noise std units)",
        labels={"color": "delta / noise std"},
    )
    heatmap.update_layout(template="plotly_white", height=780, xaxis_title="Candidate", yaxis_title="Issue #5416 item")
    heatmap.write_html(IMG_DIR / "item_delta_heatmap.html", include_plotlyjs="cdn")
    mo.ui.plotly(heatmap)
    return


@app.cell
def _(IMG_DIR, candidate_group_deltas, mo, px):
    group_bars = px.bar(
        candidate_group_deltas,
        x="group",
        y="group_delta_noise",
        color="group_delta_noise",
        facet_col="candidate",
        facet_col_wrap=2,
        color_continuous_scale="RdYlGn",
        title="Predicted group deltas vs proportional",
        labels={"group_delta_noise": "delta / noise std"},
    )
    group_bars.add_hline(y=0.0, line_dash="dash", line_color="#64748b")
    group_bars.add_hline(y=-1.0, line_dash="dot", line_color="#dc2626")
    group_bars.update_layout(template="plotly_white", height=980, showlegend=False)
    group_bars.write_html(IMG_DIR / "group_delta_bars.html", include_plotlyjs="cdn")
    mo.ui.plotly(group_bars)
    return


@app.cell
def _(IMG_DIR, candidate_weights_long, mo, px):
    weight_plot_frame = candidate_weights_long.copy()
    weight_plot_frame["domain_delta_label"] = weight_plot_frame["weight_delta"].map(lambda value: f"{value:+.3f}")
    mixture_delta = px.bar(
        weight_plot_frame,
        x="domain",
        y="weight_delta",
        color="weight_delta",
        facet_col="candidate",
        facet_row="phase",
        color_continuous_scale="RdYlGn",
        title="Candidate mixture deltas from proportional",
        labels={"weight_delta": "weight delta", "domain": "domain"},
    )
    mixture_delta.add_hline(y=0.0, line_color="#64748b", line_dash="dash")
    mixture_delta.update_layout(template="plotly_white", height=980, showlegend=False)
    mixture_delta.update_xaxes(tickangle=55, tickfont={"size": 8})
    mixture_delta.write_html(IMG_DIR / "candidate_mixture_delta_facets.html", include_plotlyjs="cdn")
    mo.ui.plotly(mixture_delta)
    return


@app.cell
def _(
    IMG_DIR,
    aggregate_packet,
    candidate_logits,
    candidate_summary,
    effective_variant,
    np,
    plt,
    proportional_weights,
    reference_plot,
    weights_from_logits,
):
    def save_candidate_mixture_png(candidate_name: str, weights: np.ndarray) -> None:
        color_candidate = plt.get_cmap("RdYlGn_r")(0.15)
        color_proportional = "#334155"
        schedules = [
            ("proportional", proportional_weights, color_proportional),
            (candidate_name, weights, color_candidate),
        ]
        non_cc_indices, cc_indices = reference_plot._grp_domain_order(aggregate_packet.domain_names, weights)
        fig, axes = plt.subplots(
            2,
            2,
            figsize=(26, 22),
            gridspec_kw={"width_ratios": [1.0, 1.62], "hspace": 0.22, "wspace": 0.31},
            facecolor="white",
        )
        reference_plot._plot_non_cc_block(
            ax=axes[0, 0],
            indices=non_cc_indices,
            labels=[reference_plot._display_non_cc_label(aggregate_packet.domain_names[idx]) for idx in non_cc_indices],
            schedules=schedules,
            phase_idx=0,
            multipliers=aggregate_packet.c0,
            title="Phase 0: Non-CC Domains",
            show_legend=True,
        )
        reference_plot._plot_cc_block(
            ax=axes[0, 1],
            domain_names=aggregate_packet.domain_names,
            indices=cc_indices,
            schedules=schedules,
            phase_idx=0,
            multipliers=aggregate_packet.c0,
            title="Phase 0: CC Domains",
        )
        reference_plot._plot_non_cc_block(
            ax=axes[1, 0],
            indices=non_cc_indices,
            labels=[reference_plot._display_non_cc_label(aggregate_packet.domain_names[idx]) for idx in non_cc_indices],
            schedules=schedules,
            phase_idx=1,
            multipliers=aggregate_packet.c1,
            title="Phase 1: Non-CC Domains",
            show_legend=False,
        )
        reference_plot._plot_cc_block(
            ax=axes[1, 1],
            domain_names=aggregate_packet.domain_names,
            indices=cc_indices,
            schedules=schedules,
            phase_idx=1,
            multipliers=aggregate_packet.c1,
            title="Phase 1: CC Domains",
        )
        fig.suptitle(
            f"{candidate_name} vs proportional ({effective_variant.name})",
            fontsize=28,
            y=0.996,
            fontweight="bold",
        )
        fig.subplots_adjust(top=0.93, left=0.14, right=0.985, bottom=0.08, hspace=0.24, wspace=0.31)
        fig.savefig(IMG_DIR / f"mixture_{candidate_name}.png", dpi=180, bbox_inches="tight")
        plt.close(fig)

    for candidate_name in candidate_summary["candidate"]:
        save_candidate_mixture_png(candidate_name, weights_from_logits(candidate_logits[candidate_name]))
    return


@app.cell
def _(
    IMG_DIR,
    candidate_group_deltas,
    candidate_item_deltas,
    candidate_summary,
    np,
    plt,
):
    fig, ax = plt.subplots(figsize=(9, 6), facecolor="white")
    scatter = ax.scatter(
        candidate_summary["min_group_delta_noise"],
        candidate_summary["predicted_aggregate_gain_vs_proportional"],
        c=candidate_summary["nearest_observed_tv"],
        cmap="RdYlGn_r",
        s=90,
        edgecolor="#0f172a",
        linewidth=0.8,
    )
    for row in candidate_summary.itertuples(index=False):
        ax.annotate(
            row.candidate,
            (row.min_group_delta_noise, row.predicted_aggregate_gain_vs_proportional),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )
    ax.axvline(0.0, color="#64748b", linestyle="--", linewidth=1.0)
    ax.axvline(-1.0, color="#dc2626", linestyle=":", linewidth=1.0)
    ax.axhline(0.0, color="#64748b", linestyle="--", linewidth=1.0)
    ax.set_xlabel("Worst group delta vs proportional (variable-noise std units)")
    ax.set_ylabel("Predicted issue #5416 aggregate gain vs proportional")
    ax.set_title("Predicted Pareto frontier")
    fig.colorbar(scatter, ax=ax, label="Nearest observed TV")
    fig.tight_layout()
    fig.savefig(IMG_DIR / "objective_frontier.png", dpi=180)
    plt.close(fig)

    heat = candidate_item_deltas.pivot(index="task_column", columns="candidate", values="item_delta_noise")
    fig, ax = plt.subplots(figsize=(12, 11), facecolor="white")
    image = ax.imshow(np.clip(heat.to_numpy(dtype=float), -3, 3), aspect="auto", cmap="RdYlGn", vmin=-3, vmax=3)
    ax.set_xticks(np.arange(len(heat.columns)))
    ax.set_xticklabels(heat.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(np.arange(len(heat.index)))
    ax.set_yticklabels([label.replace("eval/uncheatable_eval/", "uncheatable/") for label in heat.index], fontsize=7)
    ax.set_title("Predicted item deltas vs proportional")
    fig.colorbar(image, ax=ax, label="delta / noise std")
    fig.tight_layout()
    fig.savefig(IMG_DIR / "item_delta_heatmap.png", dpi=180)
    plt.close(fig)

    groups = list(candidate_group_deltas["group"].drop_duplicates())
    candidates = list(candidate_group_deltas["candidate"].drop_duplicates())
    width = 0.8 / max(1, len(candidates))
    x = np.arange(len(groups), dtype=float)
    fig, ax = plt.subplots(figsize=(13, 6.5), facecolor="white")
    for candidate_idx, candidate in enumerate(candidates):
        values = (
            candidate_group_deltas.loc[candidate_group_deltas["candidate"].eq(candidate)]
            .set_index("group")
            .reindex(groups)["group_delta_noise"]
            .to_numpy(dtype=float)
        )
        ax.bar(x + (candidate_idx - (len(candidates) - 1) / 2) * width, values, width=width, label=candidate)
    ax.axhline(0.0, color="#64748b", linestyle="--", linewidth=1.0)
    ax.axhline(-1.0, color="#dc2626", linestyle=":", linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(groups, rotation=35, ha="right")
    ax.set_ylabel("group delta / noise std")
    ax.set_title("Predicted group deltas vs proportional")
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(IMG_DIR / "group_delta_bars.png", dpi=180)
    plt.close(fig)
    return


@app.cell
def _(IMG_DIR, mo):
    mo.md(
        f"""
        ## Exported plots

        Interactive HTML plots and static mixture PNGs were written under:

        `{IMG_DIR}`
        """
    )
    return


if __name__ == "__main__":
    app.run()
