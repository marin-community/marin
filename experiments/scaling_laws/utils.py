import numpy as np
from scipy.optimize import minimize
from scipy.special import huber
from collections.abc import Sequence
from typing import Any

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
    return np.sum(huber(delta, residuals))

def fit_power_law(N, D, y, use_log_space=False, initial_guess=None, delta=1e-3):
    if initial_guess is None:
        if use_log_space:
            initial_guess = [0.0, 0.0, 1.0, 1.0, 0.0]
        else:
            initial_guess = [1.0, 1.0, 1.0, 1.0, 0.0]

    bounds = (
        [(None, None), (None, None), (0, None), (0, None), (0, None)]
        if use_log_space
        else [(0, None), (0, None), (0, None), (0, None), (0, None)]
    )

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

def filter_zero_d(N, D, losses):
    idx = (D != 0)
    return N[idx], D[idx], losses[idx]

def predict_power_law(params, N, D):
    A, B, alpha, beta, E = params
    return A / (N ** alpha) + B / (D ** beta) + E