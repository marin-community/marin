"""
Functions for fitting scaling laws and plotting the results.

The functions in this file implement the techniques in https://arxiv.org/pdf/2412.04403.

Our objective is to predict the accuracy of a larger target model on a specific benchmark.
This prediction is done through a two-step modeling process using (N, D) data from various smaller models:
- we first fit a power-law model to predict the task loss from the number of parameters and tokens.
- then, we fit a sigmoidal model to predict the task accuracy from the task loss.

For further details see the corresponding GitHub issue: https://github.com/stanford-crfm/marin/issues/646.

To use this code, call fit_task_loss_from_ladder_models() and fit_accuracy_from_task_loss() with appropriate arguments.

Example usage is in marin/scaling_laws/scaling_laws_analysis.ipynb.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit, minimize
from scipy.special import huber

from experiments.llama import LlamaConfig, compute_num_parameters, llama3_tokenizer_vocab_size

try:
    import pandas as pd
except ImportError:
    pd: Any = None

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
            #initial_guess = [0.0, 0.0, 1.0, 1.0, 0.0]  # [log_A, log_B, alpha, beta, E]
            initial_guess = [-1.0, -1.0, 2.0, 1.0, 0.1]  # Based on typical power law values
        # bounds = [
        #     (None, None),  # log_A unbounded
        #     (None, None),  # log_B unbounded
        #     (0, None),  # alpha >= 0
        #     (0, None),  # beta >= 0
        #     (0, None),  # E >= 0
        # ]
        bounds = [
            (-10, 10),     # log_A bounded
            (-10, 10),     # log_B bounded
            (0.1, 10.0),   # alpha range
            (0.1, 10.0),   # beta range
            (0, 1.0),      # E bounded
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

    #result = minimize(objective, initial_guess, method="L-BFGS-B", bounds=bounds)
    result = minimize(objective, initial_guess, method="L-BFGS-B", bounds=bounds, 
                options={'ftol': 1e-10, 'gtol': 1e-10, 'maxiter': 1000, 'disp': True})
    print(f"Success: {result.success}, Message: {result.message}, Nfev: {result.nfev}")
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
    popt, _ = curve_fit(objective, L, y, p0=initial_guess, bounds=bounds, maxfev=5000, method="trf", ftol=1e-8)

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

        # this is precautionary; compute the number of parameters ourselves to avoid discrepancies
        run_data["computed_params"] = compute_num_params_from_run(run, vocab_size=llama3_tokenizer_vocab_size)

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
    loss_col: str = None,
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
    import re

    N = df[param_count_col].values
    D = df[tokens_col].values
    y = df[loss_col].values if loss_col is not None else None

    # get hidden dim from run i.e tootsie-scaling-512-81c36c should result in (512)
    #hidden_dim = df["run"].str.extract(r"(\d+)")[0].astype(int)
    # hidden_dim = (df["run"].str.extract(r"(\d+)")[0].astype(int) 
    #         if df["run"].str.contains("scaling").any() 
    #         else 8192)



    hidden_dims = []
    for run in df["run"]:
        if "scaling" in run:
            match = re.search(r"(\d+)", run)
            hidden_dims.append(int(match.group(1)) if match else 8192)
        else:
            hidden_dims.append(8192)
    
    # Apply non_embedding_params element-wise
    N = np.array([non_embedding_params(n, h) for n, h in zip(N, hidden_dims)])


    # we want non-embedding params
    # N = non_embedding_params(N, hidden_dim=hidden_dim)

    return N, D, y


def aggregate_steps(
    df: pd.DataFrame,
    step_mode: str = "all",
    step_range: tuple[int, int] = (1, 5),
    group_col: str = "run",
) -> pd.DataFrame:
    """
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


def compute_num_params_from_run(run, vocab_size: int = llama3_tokenizer_vocab_size):
    """
    Computes the number of parameters in a run using the model config.

    Args:
        run: WandB run object
        vocab_size: Tokenizer vocab size
    """

    model_dict = run.config.get("model", {})
    llama_config = LlamaConfig(
        hidden_dim=model_dict.get("hidden_dim"),
        num_heads=model_dict.get("num_heads"),
        num_kv_heads=model_dict.get("num_kv_heads"),
        intermediate_dim=model_dict.get("intermediate_dim"),
        #num_layers=model_dict.get("num_layers"),
        num_layers=80
    )

    num_parameters = compute_num_parameters(llama_config, vocab_size=vocab_size)

    return num_parameters


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
    plt.show()

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
    plt.show()

    return plt


####################################################################################################
# Projection helpers


@dataclass
class ProjectionPoint:
    """Represents a point to project performance to"""

    num_params: int
    num_tokens: int


def get_non_emb_params_for_size(size: int) -> int:

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
    #sizes = [1.4e9, 8e9, 13e9, 22e9, 70e9]
    sizes = [13e9, 22e9, 56e9, 70e9]
    chinchilla_multipliers = [0.5, 1, 2, 5, 10, 20]


    return [ProjectionPoint(int(size), int(size * m)) for size in sizes for m in chinchilla_multipliers]


    #tokens = [4.1943040e+10, 1.2582912e+11, 2.0971520e+11, 2.9360128e+11, 3.7748736e+11]

    # for each token count, we want to project to the sizes in sizes
    return [ProjectionPoint(int(size), int(token)) for size in sizes for token in tokens]


def create_projection_df(points: list[ProjectionPoint]) -> pd.DataFrame:
    """Convert projection points to a dataframe format matching scaling law requirements"""
    return pd.DataFrame(
        {
            "parameter_count": [p.num_params for p in points],
            "throughput/total_tokens": [p.num_tokens for p in points],
            "run": [f"projection_{i}" for i in range(len(points))],
        }
    )


####################################################################################################
# Functions for fitting scaling laws


def fit_task_loss_from_ladder_models(
    runs: list[str],
    entity: str,
    project: str,
    metrics: list[str],
    pred_run: str = "llama-8b-tootsie-0.001-19ad63",
    task_loss: str = "eval/paloma/c4_en/bpb",
    aggregation: str = "all",
    tokens_col: str = "throughput/total_tokens",
    param_col: str = "parameter_count",
    param_col_to_use: str = "computed_params",
    use_log_for_ND: bool = False,
    normalize_ND: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Fits a power law model to the given ladder models and predicts the task loss for the given run.

    Args:
        runs: List of runs to pull the data from
        metrics: List of metrics to pull from the runs
        task_loss: Task loss metric to use (eg c4 en bpb, hellaswag bpb, etc.)
        aggregation: Aggregation mode to use for the steps. Can be "average", "last", or "all"
        tokens_col: Column name for the tokens
        param_col: Column name for the parameter count
        param_col_to_use: Column name to use for the parameter count. This is to specify
                    if we want to use JAX's computed params or ours.

    Returns:
        The actual and predicted task losses for the given run as (actual, predicted)
    """

    # get the data
    ladder_df = pull_metrics_from_wandb(
        runs=runs, metrics=metrics, entity=entity, project=project, summary_fields=(param_col,)
    )

    # filter out rows with zero tokens, aggregate the steps
    ladder_df_filtered = filter_zero_d(ladder_df, tokens_col)
    ladder_df_agg = aggregate_steps(ladder_df_filtered, step_mode=aggregation)

    # prepare data (N, D, y) for fitting the power law model
    N, D, y = extract_scaling_data(
        ladder_df_agg, param_col_to_use, tokens_col, task_loss
    )  # this also removes embedding param count

    if use_log_for_ND:
        N = np.log(N)
        D = np.log(D)
    if normalize_ND:
        N_scale = np.mean(N)  # Save the mean of N
        D_scale = np.mean(D)  # Save the mean of D
        N = N / N_scale
        D = D / D_scale

    # fit the power law model and make a prediction on the "training" data for sanity-checking
    params = fit_power_law(N, D, y, use_log_space=True)

    pred_df = pull_metrics_from_wandb(
        runs=[pred_run], metrics=[task_loss, tokens_col], entity=entity, project=project, summary_fields=[param_col]
    )
    pred_df_filtered = filter_zero_d(pred_df, d_key=tokens_col)

    pred_df_agg = aggregate_steps(pred_df_filtered, step_mode=aggregation)
    N_pred, D_pred, y_pred_actual = extract_scaling_data(pred_df_agg, param_col_to_use, tokens_col, task_loss)

    if use_log_for_ND:
        N_pred = np.log(N_pred)
        D_pred = np.log(D_pred)

    if normalize_ND:
        N_pred = N_pred / N_scale
        D_pred = D_pred / D_scale

    preds_big = predict_power_law(params, N_pred, D_pred)

    return y_pred_actual, preds_big


def fit_accuracy_from_task_loss(
    pred_task_losses: np.ndarray,
    runs: list[str],
    entity: str,
    project: str,
    tokens_col: str = "throughput/total_tokens",
    param_col: str = "parameter_count",
    pred_run: str = "llama-8b-tootsie-0.001-19ad63",
    aggregation: str = "all",
    task_loss_col: str = "eval/paloma/c4_en/bpb",
    accuracy_col: str = "lm_eval/hellaswag_0shot/acc",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Fit a sigmoidal function to predict the accuracy from the task loss.
    Ref: https://arxiv.org/pdf/2412.04403 sec 3.2

    Acc(L) = a / (1 + exp(-k * (L - L_0))) + b

    where:
    - L is the task loss
    - a, b, k, L_0 are parameters to fit

    Returns:
        The predicted accuracy
    """

    # get the data
    ladder_df = pull_metrics_from_wandb(
        runs=runs,
        metrics=[task_loss_col, accuracy_col, tokens_col],
        entity=entity,
        project=project,
        summary_fields=(param_col,),
    )

    # filter out rows with zero tokens, aggregate the steps
    ladder_df_filtered = filter_zero_d(ladder_df, tokens_col)
    ladder_df_agg = aggregate_steps(ladder_df_filtered, step_mode=aggregation)

    # get the data for the run we want to predict on
    pred_df = pull_metrics_from_wandb(
        runs=[pred_run],
        metrics=[accuracy_col, tokens_col],
        entity=entity,
        project=project,
        summary_fields=(param_col,),
    )

    pred_df_filtered = filter_zero_d(pred_df, d_key=tokens_col)
    pred_df_agg = aggregate_steps(pred_df_filtered, step_mode=aggregation)

    # get task losses on "training" data-points (meaning runs we use for fitting the sigmoidal model)
    N, D, task_losses = extract_scaling_data(ladder_df_agg, param_col, tokens_col, task_loss_col)

    # get accuracies on "training" data-points
    acc = ladder_df_agg[accuracy_col].values

    # fit the sigmoidal model
    params = fit_sigmoidal(task_losses, acc)

    # filter out pred_task_losses to use only the last n rows.
    # this number is determined by the number of rows in pred_df_agg
    rows_in_pred_df_agg = len(pred_df_agg)
    pred_task_losses = pred_task_losses[-rows_in_pred_df_agg:]

    # predict the accuracy
    acc_preds = predict_sigmoidal(params, pred_task_losses)

    acc_pred_actual = pred_df_agg[accuracy_col].values

    return acc_pred_actual, acc_preds


def fit_multiple_metrics_scaling_laws(
    runs: list[str],
    accuracy_metrics: Sequence[str],
    entity: str,
    project: str,
    pred_run: str = "llama-8b-tootsie-0.001-19ad63",
    task_loss: str = "eval/paloma/c4_en/bpb",
    aggregation: str = "all",
    tokens_col: str = "throughput/total_tokens",
    param_col: str = "parameter_count",
    param_col_to_use: str = "computed_params",
    use_log_for_ND: bool = False,
    normalize_ND: bool = False,
) -> tuple[tuple[np.ndarray, np.ndarray], dict[str, tuple[np.ndarray, np.ndarray]]]:
    """
    Fits scaling laws for task loss and multiple accuracy metrics.

    Args:
        runs: List of runs to pull the data from
        accuracy_metrics: List of accuracy metrics to predict
        [other args same as original functions]

    Returns:
        Tuple of:
            - (actual_loss, predicted_loss): Task loss predictions
            - Dict mapping each accuracy metric to (actual_acc, predicted_acc)
    """
    # first get task loss predictions
    actual_loss, predicted_loss = fit_task_loss_from_ladder_models(
        runs=runs,
        entity=entity,
        project=project,
        metrics=[task_loss, tokens_col],
        pred_run=pred_run,
        task_loss=task_loss,
        aggregation=aggregation,
        tokens_col=tokens_col,
        param_col=param_col,
        param_col_to_use=param_col_to_use,
        use_log_for_ND=use_log_for_ND,
        normalize_ND=normalize_ND,
    )

    # Get all data once for efficiency
    ladder_df = pull_metrics_from_wandb(
        runs=list(runs),
        metrics=[task_loss, *accuracy_metrics, tokens_col],
        entity=entity,
        project=project,
        summary_fields=(param_col,),
    )

    pred_df = pull_metrics_from_wandb(
        runs=[pred_run],
        metrics=[*accuracy_metrics, tokens_col],
        entity=entity,
        project=project,
        summary_fields=(param_col,),
    )

    # Filter and aggregate once
    ladder_df_filtered = filter_zero_d(ladder_df, tokens_col)
    ladder_df_agg = aggregate_steps(ladder_df_filtered, step_mode=aggregation)

    pred_df_filtered = filter_zero_d(pred_df, d_key=tokens_col)
    pred_df_agg = aggregate_steps(pred_df_filtered, step_mode=aggregation)

    # Get task losses for fitting once
    N, D, task_losses = extract_scaling_data(ladder_df_agg, param_col, tokens_col, task_loss)

    # fit each accuracy metric
    accuracy_results = {}
    for acc_metric in accuracy_metrics:
        acc = ladder_df_agg[acc_metric].values
        params = fit_sigmoidal(task_losses, acc)

        # get actual values for this metric
        acc_pred_actual = pred_df_agg[acc_metric].values

        # number of predictions should match actual values
        pred_task_losses = predicted_loss[-len(acc_pred_actual) :]

        # predict accuracies
        acc_preds = predict_sigmoidal(params, pred_task_losses)
        accuracy_results[acc_metric] = (acc_pred_actual, acc_preds)

    return (actual_loss, predicted_loss), accuracy_results


def fit_scaling_laws(
    runs: list[str],
    loss_metrics: Sequence[str],
    accuracy_metrics: Sequence[str],
    entity: str,
    project: str,
    pred_run: str = "llama-8b-tootsie-0.001-19ad63",
    projection_points: list[ProjectionPoint] | None = None,
    aggregation: str = "all",
    tokens_col: str = "throughput/total_tokens",
    param_col: str = "parameter_count",
    param_col_to_use: str = "computed_params",
    use_log_for_ND: bool = False,
    normalize_ND: bool = False,
) -> tuple[dict[str, np.ndarray], dict[str, tuple[np.ndarray, np.ndarray]] | None, list[ProjectionPoint] | None]:
    """Fit scaling laws for both projection and prediction"""

    # First pull for losses - only essential metrics
    metrics = list(loss_metrics) + [tokens_col]
    loss_df = pull_metrics_from_wandb(
        runs=runs,
        metrics=metrics,
        entity=entity,
        project=project,
        summary_fields=(param_col,),
    )

    # Process loss data
    loss_df_filtered = filter_zero_d(loss_df, tokens_col)
    loss_df_agg = aggregate_steps(loss_df_filtered, step_mode=aggregation)

    # Get N, D
    N, D, _ = extract_scaling_data(loss_df_agg, param_col_to_use, tokens_col)
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
            metrics=list(loss_metrics) + [tokens_col],
            entity=entity,
            project=project,
            summary_fields=(param_col,),
        )

        loss_pred_filtered = filter_zero_d(loss_pred_df, tokens_col)
        loss_pred_agg = aggregate_steps(loss_pred_filtered, step_mode=aggregation)

        N_pred, D_pred, _ = extract_scaling_data(loss_pred_agg, param_col_to_use, tokens_col)
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
                metrics=list(accuracy_metrics) + [tokens_col],
                entity=entity,
                project=project,
                summary_fields=(param_col,),
            )
            acc_pred_df = pull_metrics_from_wandb(
                runs=[pred_run],
                metrics=list(accuracy_metrics) + [tokens_col],
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
            loss_metric, (_, predicted_loss) = list(loss_results.items())[0]  # use first loss
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
