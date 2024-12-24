import numpy as np
from scipy.optimize import minimize
from scipy.special import huber
from collections.abc import Sequence
from typing import Any

import matplotlib.pyplot as plt

try:
    import pandas as pd
except ImportError:
    pd: Any = None

def power_law_model(params, N, D, use_log_space):
    if use_log_space:
        log_A, log_B, alpha, beta, E = params
        A, B = np.exp(log_A), np.exp(log_B)
    else:
        A, B, alpha, beta, E = params
    return A / (N ** alpha) + B / (D ** beta) + E

def power_law_loss(params, N, D, y, use_log_space, delta):
    predictions = power_law_model(params, N, D, use_log_space)
    if use_log_space:
        residuals = np.log(y) - np.log(predictions)
    else:
        residuals = y - predictions
    return np.mean(huber(delta, residuals))

def fit_power_law(N, D, y, use_log_space=False, initial_guess=None, delta=1e-3):
    if use_log_space:
        # Optimize log_A and log_B to ensure A, B > 0
        if initial_guess is None:
            initial_guess = [0.0, 0.0, 1.0, 1.0, 0.0]  # [log_A, log_B, alpha, beta, E]
        bounds = [
            (None, None),  # log_A unbounded
            (None, None),  # log_B unbounded
            (0, None),      # alpha >= 0
            (0, None),      # beta >= 0
            (0, None)       # E >= 0
        ]
    else:
        # Directly optimize A, B with constraints A, B >= 0
        if initial_guess is None:
            initial_guess = [1.0, 1.0, 1.0, 1.0, 0.0]  # [A, B, alpha, beta, E]
        bounds = [
            (0, None),      # A >= 0
            (0, None),      # B >= 0
            (0, None),      # alpha >= 0
            (0, None),      # beta >= 0
            (0, None)       # E >= 0
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

def pull_metrics_from_wandb(
    runs: Sequence[str],
    metrics: Sequence[str],
    entity: str,
    project: str,
    x_axis: str = "throughput/total_gflops",
    summary_fields: Sequence[str] = ("parameter_count",)
) -> pd.DataFrame:
    import wandb
    data = []
    api = wandb.Api()
    for run_id in runs:
        run = api.run(f"{entity}/{project}/{run_id}")
        run_data = {"run": run.name}
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
    return A / (N ** alpha) + B / (D ** beta) + E

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
    plt.figure()
    plt.scatter(actual, predicted, alpha=0.7)
    plt.xlabel("Actual Loss")
    plt.ylabel("Predicted Loss")
    plt.title(title)
    # disable offset
    plt.ticklabel_format(useOffset=False)  # Disable offset notation
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

    # get hidden dim from run i.e tootsie-scaling-512-81c36c (512)
    hidden_dim = df["run"].str.extract(r"(\d+)")[0].astype(int)

    # we want non-embedding params
    N = non_embedding_params(N, hidden_dim=hidden_dim)

    return N, D, y


def plot_actual_vs_predicted(y_actual, y_predicted, title="Actual vs Predicted"):
    plt.figure(figsize=(10, 6))
    
    # plot actual and predicted values
    plt.plot(y_actual, label="Actual", marker='o', linestyle='-', linewidth=2)    
    plt.plot(y_predicted, label="Predicted", marker='x', linestyle='--', linewidth=2)
    
    # add labels, legend, and title
    plt.xlabel("Step")
    plt.ylabel("Task loss")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()


def non_embedding_params(total_param_count, hidden_dim):
    # Llama-3 tokenizer vocab size
    LLAMA3_TOKENIZER_VOCAB_SIZE = 128_256
    return total_param_count - 2 * hidden_dim * LLAMA3_TOKENIZER_VOCAB_SIZE
