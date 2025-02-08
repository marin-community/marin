"""
Functions for fitting scaling laws and plotting the results.

The functions in this file implement the techniques in https://arxiv.org/pdf/2412.04403.

Our objective is to predict the accuracy of a larger target model on a specific benchmark.
This prediction is done through a two-step modeling process using (N, D) data from various smaller models:
- we first fit a power-law model to predict the task loss from the number of parameters and tokens.
- then, we fit a sigmoidal model to predict the task accuracy from the task loss.

For further details see the corresponding GitHub issue: https://github.com/stanford-crfm/marin/issues/646.

To use this code, call fit_scaling_laws() with appropriate arguments.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit, minimize
from scipy.special import huber

from experiments.llama import llama3_tokenizer_vocab_size

try:
    import pandas as pd
except ImportError:
    pd: Any = None

OPTIMIZATION_TOLERANCE = 1e-10

####################################################################################################
# Power law helpers


def power_law_model(params: Sequence[float], N: np.ndarray, D: np.ndarray, use_log_space: bool = True) -> np.ndarray:
    """
    Power-law equation: A / N^alpha + B / D^beta + E

    Args:
        params: List of parameters [A, B, alpha, beta, E]
        N: Number of parameters
        D: Number of tokens
        use_log_space: Whether to use log space for A and B
    """
    if use_log_space:
        log_A, log_B, alpha, beta, E = params
        A, B = np.exp(log_A), np.exp(log_B)
    else:
        A, B, alpha, beta, E = params
    return A / (N**alpha) + B / (D**beta) + E


def power_law_loss(
    params: Sequence[float],
    N: np.ndarray,
    D: np.ndarray,
    y: np.ndarray,
    use_log_space: bool,
    delta: float,
    reduction: Callable[[np.ndarray], float] | None = np.sum,
) -> float:
    """
    Huber loss for the power-law model.
    Args:
        params: List of parameters [A, B, alpha, beta, E]
        N: Number of parameters
        D: Number of tokens
        y: Actual loss
        use_log_space: if true, residual is set to difference of logs of actual and predicted values
        delta: huber loss delta, indicating the quadratic vs. linear loss changepoint.
        reduction: Optional argument to change the reduction used on the Huber loss, defaults to sum based on https://arxiv.org/pdf/2404.10102v2
    """
    predictions = power_law_model(params, N, D, use_log_space)
    if use_log_space:
        residuals = np.log(y) - np.log(predictions)
    else:
        residuals = y - predictions
    return reduction(huber(delta, residuals))


def fit_power_law(
    N: np.ndarray,
    D: np.ndarray,
    y: np.ndarray,
    use_log_space: bool = False,
    initial_guess: Sequence[float] | None = None,
    delta: float = 1e-3,
) -> np.ndarray | tuple[float, float, float, float, float]:
    """
    Fit a power law model to the data ((N, D), y).

    Args:
        N: Number of parameters
        D: Number of tokens
        y: Actual loss or metric we want to learn to predict
        use_log_space: if true, A and B are in log space *AND* Huber loss is computed in log space.
        initial_guess: Initial guess for the parameters
        delta: huber loss delta, indicating the quadratic vs. linear loss changepoint.
    """
    if use_log_space:
        if initial_guess is None:
            initial_guess = [0.0, 0.0, 1.0, 1.0, 0.0]  # [log_A, log_B, alpha, beta, E]
        bounds = [
            (None, None),  # log_A unbounded
            (None, None),  # log_B unbounded
            (0, None),  # alpha >= 0
            (0, None),  # beta >= 0
            (0, None),  # E >= 0
        ]
    else:
        if initial_guess is None:
            initial_guess = [1.0, 1.0, 1.0, 1.0, 0.0]  # [A, B, alpha, beta, E]
        bounds = [
            (0, None),  # A >= 0
            (0, None),  # B >= 0
            (0, None),  # alpha >= 0
            (0, None),  # beta >= 0
            (0, None),  # E >= 0
        ]

    def objective(params):
        return power_law_loss(params, N, D, y, use_log_space, delta)

    result = minimize(
        objective,
        initial_guess,
        method="L-BFGS-B",
        bounds=bounds,
        options={"ftol": OPTIMIZATION_TOLERANCE, "gtol": OPTIMIZATION_TOLERANCE, "maxiter": 1000},
    )

    if not result.success:
        raise RuntimeError(f"Optimization failed: {result.message}")

    # return the fitted parameters, converting log_A and log_B back to A and B if needed
    if use_log_space:
        log_A, log_B, alpha, beta, E = result.x
        A, B = np.exp(log_A), np.exp(log_B)
        return A, B, alpha, beta, E
    else:
        return result.x


def predict_power_law(params: Sequence[float], N: np.ndarray, D: np.ndarray) -> np.ndarray:
    A, B, alpha, beta, E = params
    return A / (N**alpha) + B / (D**beta) + E


####################################################################################################
# Sigmoidal fit helpers


def fit_sigmoidal(L: np.ndarray, y: np.ndarray, initial_guess: Sequence[float] | None = None) -> np.ndarray:
    """
    Fit a sigmoidal model to the data (L, y).

    Equation: a / (1 + exp(-k * (L - L_0))) + b

    Args:
        L: Task loss
        y: Ground-truth task accuracy
        initial_guess: Initial guess for the parameters
    """

    if initial_guess is None:
        initial_guess = [1.0, 0.0, 1.0, 0.0]  # [a, b, k, L_0]

    lower_bounds = [-np.inf, -np.inf, -np.inf, -np.inf]
    upper_bounds = [np.inf, np.inf, np.inf, np.inf]

    bounds = (lower_bounds, upper_bounds)

    def objective(L, a, b, k, L_0):
        return predict_sigmoidal([a, b, k, L_0], L)

    # use scipy.optimize.curve_fit
    popt, _ = curve_fit(objective, L, y, p0=initial_guess, bounds=bounds, maxfev=5000, method="trf", ftol=OPTIMIZATION_TOLERANCE)

    return popt


def predict_sigmoidal(params: Sequence[float], task_loss: np.ndarray) -> np.ndarray:
    a, b, k, L_0 = params
    return a / (1 + np.exp(-k * (task_loss - L_0))) + b


####################################################################################################
# WandB and data processing helpers


def pull_metrics_from_wandb(
    runs: Sequence[str],
    metrics: Sequence[str],
    entity: str,
    project: str,
    summary_fields: Sequence[str] = ("parameter_count",),
) -> pd.DataFrame:
    """
    Pulls the metrics from the given runs and returns a DataFrame.

    Args:
        runs: List of run IDs
        metrics: List of metrics to pull from the runs; these differ depending on the step (unlike summary_fields)
        entity: WandB entity
        project: WandB project
        summary_fields: List of summary fields to pull from the runs

    Returns:
        Pandas dataFrame with the metrics
    """

    import wandb

    api = wandb.Api()

    data = []
    for run_id in runs:
        run = api.run(f"{entity}/{project}/{run_id}")
        run_data = {"run": run.name}

        # Get model configuration
        model_dict = run.config.get("model", {})

        model_dict = run.config.get("model", {})
        run_data["hidden_dim"] = model_dict.get("hidden_dim", 0)

        # get the summary fields
        for field in summary_fields:
            run_data[field] = run.summary.get(field, None)

        # get the per-step metrics
        history = run.history(keys=metrics)

        for i in range(len(history)):
            step_data = {m: history[m][i] for m in metrics}
            step_data.update(run_data)
            step_data["step"] = i
            data.append(step_data)

    return pd.DataFrame(data)


def filter_zero_d(df: pd.DataFrame, d_key: str = "throughput/total_tokens") -> pd.DataFrame:
    """
    Returns a new DataFrame that excludes any rows where the specified
    'd_key' column is zero.
    """
    return df[df[d_key] != 0].copy()


def extract_scaling_data(
    df: pd.DataFrame,
    param_count_col: str = "parameter_count",
    tokens_col: str = "throughput/total_tokens",
    loss_col: str | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extracts N, D, and y from the given DataFrame.

    Args:
        df: DataFrame
        param_count_col: Column name for the parameter count
        tokens_col: Column name for the tokens
        loss_col: Column name for the loss

    Returns:
        Tuple of numpy arrays: (N, D, y) where
            N = Number of parameters (excluding embedding parameters)
            D = Number of tokens
            y = Loss
    """

    N = df[param_count_col].values
    D = df[tokens_col].values
    y = df[loss_col].values if loss_col is not None else None

    # Apply non_embedding_params element-wise
    N = np.array([non_embedding_params(n, h) for n, h in zip(N, df["hidden_dim"].values, strict=False)])

    return N, D, y


def aggregate_steps(
    df: pd.DataFrame,
    step_mode: str = "all",
    step_range: tuple[int, int] = (1, 5),
    group_col: str = "run",
) -> pd.DataFrame:
    """
    Aggregates the steps for each run.

    Args:
        df: DataFrame
        step_mode: how to aggregate the steps
        step_range: range of steps to aggregate
        group_col: column to group by
    
    step_mode can be:
      - "average": average step_range across each run
      - "last": pick the max step within step_range
      - "all": keep every step (no grouping)
    """

    if step_mode == "average":
        grouped = df.groupby(group_col, as_index=False).mean(numeric_only=True)
        return grouped
    elif step_mode == "last":
        # pick the largest step in the range for each run
        def pick_last(g):
            last_step_idx = g["step"].idxmax()
            return g.loc[last_step_idx]

        grouped = df.groupby(group_col, as_index=False).apply(pick_last)
        return grouped.reset_index(drop=True)
    elif step_mode == "all":
        # no aggregation
        return df.copy()
    else:
        raise ValueError(f"Unknown step_mode: {step_mode}")


def non_embedding_params(total_param_count: int, hidden_dim: int, vocab_size: int = llama3_tokenizer_vocab_size):
    return total_param_count - 2 * hidden_dim * vocab_size


####################################################################################################
# Projection-specific helpers


@dataclass
class ProjectionPoint:
    """Represents a point to project performance to"""

    num_params: int
    num_tokens: int


def get_non_emb_params_for_size(size: int) -> int:
    """
    Get the number of non-embedding parameters for a given model size.
    Used for projection points, since we do not have a hidden dimension associated with a given (N, D) point.
    """

    # this is just to get the number of non-emb params for a given size
    size_to_hidden_dim = {
        250e6: 512,
        500e6: 768,
        750e6: 1024,
        1.4e9: 1536,
        2.4e9: 2048,
        8e9: 4096,
        13e9: 5120,
        22e9: 6144,
        56e9: 8192,
        70e9: 8192,
    }

    # get the closest size
    closest_size = min(size_to_hidden_dim.keys(), key=lambda x: abs(x - size))

    hidden_dim = size_to_hidden_dim[closest_size]

    non_emb_params = non_embedding_params(size, hidden_dim)

    return non_emb_params


def get_default_projection_points() -> list[ProjectionPoint]:
    """Default set of model sizes to project to"""
    sizes = [1.4e9, 8e9, 13e9, 22e9, 70e9]
    chinchilla_multipliers = [0.5, 1, 5, 10, 20, 30, 50]

    return [ProjectionPoint(int(size), int(size * m)) for size in sizes for m in chinchilla_multipliers]


####################################################################################################
# Plotting helpers


def plot_fit(actual: np.ndarray, predicted: np.ndarray, title="Power Law Fit") -> None:
    """
    Plot predicted vs actual values.
    """
    plt.figure()
    plt.scatter(actual, predicted, alpha=0.7)
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title(title)

    # disable offset notation for the y-axis
    plt.ticklabel_format(useOffset=False)

    plt.grid(True)

    return plt


def plot_actual_vs_predicted(
    y_actual: np.ndarray,
    y_predicted: np.ndarray,
    title: str = "Actual vs Predicted",
    task_metric: str = "eval/paloma/c4_en/bpb",
) -> None:
    """
    Plot actual vs predicted values. task_metric is the name of the metric we are predicting.
    """

    plt.figure(figsize=(10, 6))

    # plot actual and predicted values
    plt.plot(y_actual, label="Actual", marker="o", linestyle="-", linewidth=2)
    plt.plot(y_predicted, label="Predicted", marker="x", linestyle="--", linewidth=2)

    # add labels, legend, and title
    plt.xlabel("Step")
    plt.ylabel("Metric: " + task_metric)
    plt.title(title)
    plt.legend()
    plt.grid(True)

    return plt


def plot_scaling_projections(predicted: np.ndarray, points: list[ProjectionPoint]):
    """
    Plot scaling law predictions vs tokens for specified model sizes.

    Args:
        predicted: Array of predicted values
        points: List of ProjectionPoint objects containing parameter and token counts

    Returns:
        matplotlib.pyplot figure object
    """
    plt.figure(figsize=(12, 6))
    unique_params = np.unique([p.num_params for p in points])

    for param in unique_params:
        mask = np.array([p.num_params == param for p in points])
        tokens = np.array([p.num_tokens for p in points])[mask]
        preds = predicted[mask]
        plt.plot(tokens, preds, "o-", linewidth=2, label=f"{param/1e9:.1f}B params")

        # add annotations for each point
        for t, pred in zip(tokens, preds, strict=False):
            token_str = f"{t/1e9:.1f}B" if t < 1e11 else f"{t/1e12:.1f}T"
            plt.annotate(f"{token_str}, {pred:.3f}", (t, pred), ha="center", va="bottom", fontsize=6)

    plt.xscale("log")
    plt.xlabel("Number of Tokens")
    plt.ylabel("Predicted Loss")
    plt.grid(True)
    plt.legend()
    return plt


####################################################################################################
# Functions for fitting scaling laws


def fit_scaling_laws(
    runs: list[str],
    loss_metrics: Sequence[str],
    accuracy_metrics: Sequence[str],
    entity: str,
    project: str,
    pred_run: str | None = None,
    projection_points: list[ProjectionPoint] | None = None,
    aggregation: str = "all",
    tokens_col: str = "throughput/total_tokens",
    param_col: str = "parameter_count",
    use_log_for_ND: bool = False,
    normalize_ND: bool = False,
) -> tuple[dict[str, np.ndarray], dict[str, tuple[np.ndarray, np.ndarray]] | None, list[ProjectionPoint] | None]:
    """Fit scaling laws for both projection and prediction

    Args:
        runs: list of run IDs to fit scaling laws for
        loss_metrics: list of loss metrics to fit scaling laws for
        accuracy_metrics: list of accuracy metrics to fit scaling laws for
        entity: WandB entity
        project: WandB project
        pred_run: run ID to predict scaling laws for- if None, no prediction is done
        projection_points: list of ProjectionPoint objects to project to
        aggregation: how to aggregate steps within each run (all/last/average)
        tokens_col: column name for the number of tokens
        param_col: column name for the number of parameters
        use_log_for_ND: whether to use log space for N and D
        normalize_ND: whether to normalize N and D
    """

    # First pull for losses - only essential metrics
    metrics = [*list(loss_metrics), tokens_col]
    loss_df = pull_metrics_from_wandb(
        runs=runs,
        metrics=metrics,
        entity=entity,
        project=project,
        summary_fields=(param_col,),
    )

    # Process loss data- remove 0-token runs, apply aggregation to the ladder runs' checkpoints (if specified)      
    loss_df_filtered = filter_zero_d(loss_df, tokens_col)
    loss_df_agg = aggregate_steps(loss_df_filtered, step_mode=aggregation)

    # Get N, D
    N, D, _ = extract_scaling_data(loss_df_agg, param_col, tokens_col)
    if use_log_for_ND:
        N = np.log(N)
        D = np.log(D)
    if normalize_ND:
        N_scale = np.mean(N)
        D_scale = np.mean(D)
        N = N / N_scale
        D = D / D_scale

    # Handle projections
    projections = {}
    points = None
    if projection_points:
        points = projection_points or get_default_projection_points()
        N_proj = np.array([get_non_emb_params_for_size(p.num_params) for p in points])
        D_proj = np.array([p.num_tokens for p in points])

        if use_log_for_ND:
            N_proj, D_proj = np.log(N_proj), np.log(D_proj)
        if normalize_ND:
            N_proj, D_proj = N_proj / N_scale, D_proj / D_scale

        for loss_metric in loss_metrics:
            y = loss_df_agg[loss_metric].values
            params = fit_power_law(N, D, y, use_log_space=True)
            projections[loss_metric] = predict_power_law(params, N_proj, D_proj)

    predictions = None
    if pred_run:
        loss_pred_df = pull_metrics_from_wandb(
            runs=[pred_run],
            metrics=[*list(loss_metrics), tokens_col],
            entity=entity,
            project=project,
            summary_fields=(param_col,),
        )

       
        loss_pred_filtered = filter_zero_d(loss_pred_df, tokens_col)
        loss_pred_agg = aggregate_steps(loss_pred_filtered, step_mode=aggregation)

        N_pred, D_pred, _ = extract_scaling_data(loss_pred_agg, param_col, tokens_col)
        if use_log_for_ND:
            N_pred = np.log(N_pred)
            D_pred = np.log(D_pred)
        if normalize_ND:
            N_pred = N_pred / N_scale
            D_pred = D_pred / D_scale

        # Fit losses
        loss_results = {}
        for loss_metric in loss_metrics:
            y = loss_df_agg[loss_metric].values
            params = fit_power_law(N, D, y, use_log_space=True)
            actual_loss = loss_pred_agg[loss_metric].values
            predicted_loss = predict_power_law(params, N_pred, D_pred)
            loss_results[loss_metric] = (actual_loss, predicted_loss)

        # Second pull for accuracies
        if accuracy_metrics:
            acc_df = pull_metrics_from_wandb(
                runs=runs,
                metrics=[*list(accuracy_metrics), tokens_col],
                entity=entity,
                project=project,
                summary_fields=(param_col,),
            )
            acc_pred_df = pull_metrics_from_wandb(
                runs=[pred_run],
                metrics=[*list(accuracy_metrics), tokens_col],
                entity=entity,
                project=project,
                summary_fields=(param_col,),
            )

            acc_df_filtered = filter_zero_d(acc_df, tokens_col)
            acc_df_agg = aggregate_steps(acc_df_filtered, step_mode=aggregation)
            acc_pred_filtered = filter_zero_d(acc_pred_df, tokens_col)
            acc_pred_agg = aggregate_steps(acc_pred_filtered, step_mode=aggregation)

            # Fit accuracies
            accuracy_results = {}
            loss_metric, (_, predicted_loss) = next(iter(loss_results.items()))  # use first loss
            task_losses = loss_df_agg[loss_metric].values
            for acc_metric in accuracy_metrics:
                acc = acc_df_agg[acc_metric].values
                params = fit_sigmoidal(task_losses, acc)
                acc_pred_actual = acc_pred_agg[acc_metric].values
                pred_task_losses = predicted_loss[-len(acc_pred_actual) :]
                acc_preds = predict_sigmoidal(params, pred_task_losses)
                accuracy_results[f"{acc_metric}_from_{loss_metric}"] = (acc_pred_actual, acc_preds)

        predictions = (loss_results, accuracy_results)

    return projections, predictions, points
