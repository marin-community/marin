from collections.abc import Sequence
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit, minimize
from scipy.special import huber

from experiments.llama import LlamaConfig, compute_num_parameters

try:
    import pandas as pd
except ImportError:
    pd: Any = None

# Llama-3 tokenizer vocab size
LLAMA3_TOKENIZER_VOCAB_SIZE = 128_256


def power_law_model(params, N, D, use_log_space=True):
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


def sigmoidal_model(params, L):
    a, b, k, L_0 = params
    return a / (1 + np.exp(-k * (L - L_0))) + b


def power_law_loss(params, N, D, y, use_log_space, delta):
    predictions = power_law_model(params, N, D, use_log_space)
    if use_log_space:
        residuals = np.log(y) - np.log(predictions)
    else:
        residuals = y - predictions
    return np.mean(huber(delta, residuals))


def sigmoidal_loss(params, L, y, delta):
    # mean of L2 loss between actual and predicted values
    predictions = sigmoidal_model(params, L)
    residuals = y - predictions

    # L2 loss
    return np.mean(residuals**2)


def fit_power_law(N, D, y, use_log_space=False, initial_guess=None, delta=1e-3):
    if use_log_space:
        # Optimize log_A and log_B to ensure A, B > 0
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
        # Directly optimize A, B with constraints A, B >= 0
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

    if use_log_space:
        log_A, log_B, alpha, beta, E = result.x
        A, B = np.exp(log_A), np.exp(log_B)
        return A, B, alpha, beta, E
    else:
        return result.x


def fit_sigmoidal(L, y, initial_guess=None, delta=1e-3):

    if initial_guess is None:
        initial_guess = [1.0, 0.0, 1.0, 0.0]  # [a, b, k, L_0]

    lower_bounds = [0, -np.inf, 0, -np.inf]  # Lower bounds for [a, b, k, L_0]
    upper_bounds = [np.inf, np.inf, np.inf, np.inf]  # Upper bounds for [a, b, k, L_0]
    bounds = (lower_bounds, upper_bounds)

    def objective(L, a, b, k, L_0):
        params = [a, b, k, L_0]
        return sigmoidal_loss(params, L, y, delta)

    # use scipy.optimize.curve_fit
    popt, _ = curve_fit(objective, L, y, p0=initial_guess, bounds=bounds)

    print("Fitted sigmoidal params:", popt)

    return popt


def pull_metrics_from_wandb(
    runs: Sequence[str],
    metrics: Sequence[str],
    entity: str,
    project: str,
    x_axis: str = "throughput/total_gflops",
    summary_fields: Sequence[str] = ("parameter_count",),
) -> pd.DataFrame:
    import wandb

    data = []
    api = wandb.Api()
    for run_id in runs:
        run = api.run(f"{entity}/{project}/{run_id}")
        run_data = {"run": run.name}

        run_data["computed_params"] = compute_num_params_from_run(run, vocab_size=LLAMA3_TOKENIZER_VOCAB_SIZE)

        for field in summary_fields:
            run_data[field] = run.summary.get(field, None)
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


def predict_power_law(params, N, D):
    A, B, alpha, beta, E = params
    return A / (N**alpha) + B / (D**beta) + E


def predict_sigmoidal(params, task_loss):
    return sigmoidal_model(params, task_loss)


def aggregate_steps(
    df: pd.DataFrame,
    step_mode: str = "average",
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


def plot_fit(actual, predicted, title="Power Law Fit"):
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


def extract_ndy(
    df: pd.DataFrame,
    param_count_col: str = "parameter_count",
    tokens_col: str = "throughput/total_tokens",
    loss_col: str = "eval/paloma/c4_en/bpb",
):
    N = df[param_count_col].values
    D = df[tokens_col].values
    y = df[loss_col].values

    # get hidden dim from run i.e tootsie-scaling-512-81c36c should result in (512)
    hidden_dim = df["run"].str.extract(r"(\d+)")[0].astype(int)

    # we want non-embedding params
    N = non_embedding_params(N, hidden_dim=hidden_dim)

    return N, D, y


def plot_actual_vs_predicted(
    y_actual, y_predicted, title="Actual vs Predicted", task_loss: str = "eval/paloma/c4_en/bpb"
):
    plt.figure(figsize=(10, 6))

    # plot actual and predicted values
    plt.plot(y_actual, label="Actual", marker="o", linestyle="-", linewidth=2)
    plt.plot(y_predicted, label="Predicted", marker="x", linestyle="--", linewidth=2)

    # add labels, legend, and title
    plt.xlabel("Step")
    plt.ylabel("Task loss: " + task_loss)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()


def non_embedding_params(total_param_count, hidden_dim):
    return total_param_count - 2 * hidden_dim * LLAMA3_TOKENIZER_VOCAB_SIZE


def compute_num_params_from_run(run, vocab_size: int = LLAMA3_TOKENIZER_VOCAB_SIZE):

    model_dict = run.config.get("model", {})
    llama_config = LlamaConfig(
        hidden_dim=model_dict.get("hidden_dim"),
        num_heads=model_dict.get("num_heads"),
        num_kv_heads=model_dict.get("num_kv_heads"),
        intermediate_dim=model_dict.get("intermediate_dim"),
        num_layers=model_dict.get("num_layers"),
    )

    num_parameters = compute_num_parameters(llama_config, vocab_size=vocab_size)

    print("Computed params:", num_parameters)

    return num_parameters


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
        param_col_to_use: Column name to use for the parameter count. This is to specify if we want to use JAX's computed params or ours.

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
    N, D, y = extract_ndy(
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
    print("Fitted params (small runs):", params)
    preds = predict_power_law(params, N, D)

    pred_df = pull_metrics_from_wandb(
        runs=[pred_run], metrics=[task_loss, tokens_col], entity=entity, project=project, summary_fields=[param_col]
    )
    pred_df_filtered = filter_zero_d(pred_df, d_key=tokens_col)
    # pred_df_filtered = pred_df

    pred_df_agg = aggregate_steps(pred_df_filtered, step_mode=aggregation)
    N_pred, D_pred, y_pred_actual = extract_ndy(pred_df_agg, param_col_to_use, tokens_col, task_loss)

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
    ground_truth_task_losses: np.ndarray,
    runs: list[str],
    entity: str,
    project: str,
    metrics: list[str],
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
        runs=runs, metrics=metrics, entity=entity, project=project, x_axis=x_axis, summary_fields=(param_col,)
    )

    # filter out rows with zero tokens, aggregate the steps
    ladder_df_filtered = filter_zero_d(ladder_df, tokens_col)
    ladder_df_agg = aggregate_steps(ladder_df_filtered, step_mode=aggregation)

    # get task losses on "training" data-points (meaning runs we use for fitting the sigmoidal model)
    N, D, task_losses = extract_ndy(ladder_df_agg, param_col, tokens_col, task_loss_col)

    # get accuracies on "training" data-points
    acc = ladder_df_agg[accuracy_col].values

    # TODO:
    # in the paper they mention "To smoothen the noise, we apply a moving average
    # on the task loss and task accuracy over all checkpoints of each training run,
    # with a window size of 5. We also discard the checkpoints
    # from the first 10% of each training run as these are quite noisy,
    # and add an extra data point (L = 0.0, Acc = 1.0)" and other tweaks."

    print("Task losses:", task_losses)
    print("Accuracies:", acc)
    print("Task losses shape:", task_losses.shape)
    print("Accuracies shape:", acc.shape)
    print("Going to fit sigmoidal model")

    # fit the sigmoidal model
    params = fit_sigmoidal(task_losses, acc, delta=1e-3)
    print("Fitted sigmoidal params:", params)

    # predict the accuracy
    acc_preds = predict_sigmoidal(params, pred_task_losses)

    pred_df = pull_metrics_from_wandb(
        runs=[pred_run],
        metrics=metrics,
        entity=entity,
        project=project,
        summary_fields=[
            param_col,
        ],
    )
    pred_df_filtered = filter_zero_d(pred_df, d_key=tokens_col)

    pred_df_agg = aggregate_steps(pred_df_filtered, step_mode=aggregation)

    print("Predicting on:", pred_df_agg.head())
    print("Number of rows:", len(pred_df_agg))

    acc_pred_actual = pred_df_agg[accuracy_col].values

    return acc_pred_actual, acc_preds
