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

from collections.abc import Sequence
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
    params: Sequence[float], N: np.ndarray, D: np.ndarray, y: np.ndarray, use_log_space: bool, delta: float
) -> float:
    """
    Mean Huber loss for the power-law model.

    Args:
        params: List of parameters [A, B, alpha, beta, E]
        N: Number of parameters
        D: Number of tokens
        y: Actual loss
        use_log_space: if true, residual is set to difference of logs of actual and predicted values
        delta: huber loss delta, indicating the quadratic vs. linear loss changepoint.
    """
    predictions = power_law_model(params, N, D, use_log_space)
    if use_log_space:
        residuals = np.log(y) - np.log(predictions)
    else:
        residuals = y - predictions
    return np.mean(huber(delta, residuals))


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

    result = minimize(objective, initial_guess, method="L-BFGS-B", bounds=bounds)
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
    popt, _ = curve_fit(objective, L, y, p0=initial_guess, bounds=bounds)

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
    x_axis: str = "throughput/total_gflops",
    summary_fields: Sequence[str] = ("parameter_count",),
) -> pd.DataFrame:
    """
    Pulls the metrics from the given runs and returns a DataFrame.

    Args:
        runs: List of run IDs
        metrics: List of metrics to pull from the runs; these differ depending on the step (unlike summary_fields)
        entity: WandB entity
        project: WandB project
        x_axis: Column to use for the x-axis (eg. throughput/total_gflops)
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
        history = run.history(keys=metrics, x_axis=x_axis)
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
    loss_col: str = "eval/paloma/c4_en/bpb",
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
    y = df[loss_col].values

    # get hidden dim from run i.e tootsie-scaling-512-81c36c should result in (512)
    hidden_dim = df["run"].str.extract(r"(\d+)")[0].astype(int)

    # we want non-embedding params
    N = non_embedding_params(N, hidden_dim=hidden_dim)

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
        num_layers=model_dict.get("num_layers"),
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
    plt.xlabel("Actual Loss")
    plt.ylabel("Predicted Loss")
    plt.title(title)

    # disable offset notation for the y-axis
    plt.ticklabel_format(useOffset=False)

    plt.grid(True)
    plt.show()


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


####################################################################################################
# Functions for fitting scaling laws


def fit_task_loss_from_ladder_models(
    runs: list[str],
    entity: str,
    project: str,
    metrics: list[str],
    x_axis: str = "throughput/total_gflops",
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
        runs=runs, metrics=metrics, entity=entity, project=project, x_axis=x_axis, summary_fields=(param_col,)
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

    return y_pred_actual, preds_big.to_numpy()


def fit_accuracy_from_task_loss(
    pred_task_losses: np.ndarray,
    runs: list[str],
    entity: str,
    project: str,
    x_axis: str = "throughput/total_gflops",
    tokens_col: str = "throughput/total_tokens",
    param_col: str = "parameter_count",
    pred_run: str = "llama-8b-tootsie-0.001-19ad63",
    aggregation: str = "all",
    task_loss_col: str = "eval/paloma/c4_en/bpb",
    accuracy_col: str = "lm_eval/hellaswag_0shot/acc",
) -> np.ndarray:
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
        x_axis=x_axis,
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
        x_axis=x_axis,
        summary_fields=(param_col,),
    )

    pred_df_filtered = filter_zero_d(pred_df, d_key=tokens_col)
    pred_df_agg = aggregate_steps(pred_df_filtered, step_mode=aggregation)

    # get task losses on "training" data-points (meaning runs we use for fitting the sigmoidal model)
    N, D, task_losses = extract_scaling_data(ladder_df_agg, param_col, tokens_col, task_loss_col)

    # get accuracies on "training" data-points
    acc = ladder_df_agg[accuracy_col].values

    # TODO:
    # in the paper they mention "To smoothen the noise, we apply a moving average
    # on the task loss and task accuracy over all checkpoints of each training run,
    # with a window size of 5. We also discard the checkpoints
    # from the first 10% of each training run as these are quite noisy,
    # and add an extra data point (L = 0.0, Acc = 1.0)" and other tweaks."

    # fit the sigmoidal model
    params = fit_sigmoidal(task_losses, acc)

    # filter out pred_task_losses to use only the last number of rows.
    # this number is determined by the number of rows in pred_df_agg
    rows_in_pred_df_agg = len(pred_df_agg)
    pred_task_losses = pred_task_losses[-rows_in_pred_df_agg:]

    # predict the accuracy
    acc_preds = predict_sigmoidal(params, pred_task_losses)

    acc_pred_actual = pred_df_agg[accuracy_col].values

    return acc_pred_actual, acc_preds
