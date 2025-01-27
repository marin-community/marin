"""
This file contains functions for setting scaling law configurations, wrapper functions to call the relevant
regressions/predictions, and creating a WandB report with the results. The code here implements the function
(see `run_scaling_law_analysis`, that will be called by an ExecutorStep in the scaling laws analysis pipeline.

Our objective is to predict the accuracy of a larger target model on a specific benchmark.
This prediction is done through a two-step modeling process using (N, D) data from various smaller models:
- we first fit a power-law model to predict the task loss from the number of parameters and tokens.
- then, we fit a sigmoidal model to predict task accuracy from the task loss.

Reference:
    Establishing Task Scaling Laws via Compute-Efficient Model Ladders
    Bhagia et. al 2024
    https://arxiv.org/pdf/2412.04403.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import wandb
import numpy as np
import pandas as pd

from marin.execution.executor import ExecutorStep


@dataclass(frozen=True)
class ScalingLawConfig:

    ladder_model_steps: list[ExecutorStep | str]
    """list of (smaller model) steps or wandb run ids to be used as input for scaling laws"""

    pred_model_step: ExecutorStep | str
    """executor step or wandb run id for the larger model to make predictions for"""

    intermediate_task_loss: str = "eval/paloma/c4_en/bpb"
    """intermediate task loss to be used for scaling laws (eg. c4en bpb)"""

    task_accuracies: Sequence[str] = "lm_eval/hellaswag_10shot/acc"
    """task accuracy to predict for the larger model (eg. hellaswag accuracy)"""


def get_wandb_run_id_from_step(step: ExecutorStep) -> str:
    """
    Get the wandb run id from a given ExecutorStep.
    """
    return step.config.trainer.tracker.id


def run_scaling_law_analysis(config: ScalingLawConfig) -> None:
    """
    Analyze scaling laws for a given task loss and multiple accuracy metrics.
    """
    from marin.scaling_laws.utils import fit_multiple_metrics_scaling_laws

    input_run_ids = [
        get_wandb_run_id_from_step(step) if isinstance(step, ExecutorStep) else step
        for step in config.ladder_model_steps
    ]

    if isinstance(config.pred_model_step, ExecutorStep):
        pred_run_id = get_wandb_run_id_from_step(config.pred_model_step)
    else:
        pred_run_id = config.pred_model_step

    # Get predictions for task loss and all accuracy metrics in one go
    (actual_loss, predicted_loss), accuracy_results = fit_multiple_metrics_scaling_laws(
        runs=input_run_ids,
        accuracy_metrics=config.task_accuracies,
        entity="stanford-mercury",
        project="marin",
        pred_run=pred_run_id,
        task_loss=config.intermediate_task_loss,
        aggregation="all",
        tokens_col="throughput/total_tokens",
        param_col="parameter_count",
        param_col_to_use="computed_params",
        use_log_for_ND=True,
        normalize_ND=True,
    )

    # Log and create a report
    log_and_create_report(
        actual_loss=actual_loss.tolist(),
        predicted_loss=predicted_loss.tolist(),
        accuracy_results={k: (v[0].tolist(), v[1].tolist()) for k, v in accuracy_results.items()},
        input_run_ids=input_run_ids,
        pred_run_id=pred_run_id,
        scaling_law_config=config,
        wandb_project="marin-scaling-laws",
        wandb_entity="stanford-mercury",
    )


def log_and_create_report(
    actual_loss: list,
    predicted_loss: list,
    accuracy_results: dict[str, tuple[list, list]],
    input_run_ids: list,
    pred_run_id: str,
    scaling_law_config: ScalingLawConfig,
    wandb_project: str = "marin-scaling-laws",
    wandb_entity: str = "stanford-mercury",
):
    """
    Logs scaling law analysis creates a concise WandB report with plots and info about runs.

    Args:
        actual_loss (list): List of actual task loss values.
        predicted_loss (list): List of predicted task loss values.
        accuracy_results (dict): Dictionary mapping accuracy metric names to tuples of
                               (actual_acc_list, predicted_acc_list).
        input_run_ids (list): List of input WandB run IDs for smaller models.
        pred_run_id (str): WandB run ID for the larger model.
        scaling_law_config (ScalingLawConfig): Scaling law configuration.
        wandb_project (str): WandB project name.
        wandb_entity (str): WandB entity (user or team) name.
    """
    # Initialize WandB run
    run = wandb.init(
        project=wandb_project,
        entity=wandb_entity,
        name=f"""Scaling Law Report: {pred_run_id}-ngrfuyg""",
        tags=["scaling_laws"],
        config={
            "input_runs": input_run_ids,
            "prediction_run": pred_run_id,
        },
        reinit=True,
    )

    # Create steps array
    steps = list(range(len(actual_loss)))

    # Create plots dictionary for wandb.log
    plots = {}

    # Add loss plot
    plots["Task Loss"] = wandb.plot.line_series(
        xs=steps,
        ys=[actual_loss, predicted_loss],
        keys=["Actual", "Predicted"],
        title="Task Loss: Actual vs Predicted",
        xname="Step",
        split_table=True,
    )

    # Add accuracy plots for each metric
    for metric, (actual_acc, predicted_acc) in accuracy_results.items():
        plots[f"Task Accuracy - {metric}"] = wandb.plot.line_series(
            xs=steps,
            ys=[actual_acc[:30], predicted_acc[:30]],
            keys=["Actual", "Predicted"],
            title=f"Task Accuracy ({metric}): Actual vs Predicted",
            xname="Step",
            split_table=True,
        )

    # Log all plots
    wandb.log(plots)

    # Info about runs and links
    input_run_links = [f"https://wandb.ai/stanford-mercury/marin/runs/{run_id}" for run_id in input_run_ids]
    prediction_run_link = f"https://wandb.ai/stanford-mercury/marin/runs/{pred_run_id}"
    run.summary.update(
        {
            "Input Runs": input_run_links,
            "Prediction Run": prediction_run_link,
            "Task Loss Metric": scaling_law_config.intermediate_task_loss,
            "Task Accuracy Metrics": scaling_law_config.task_accuracies,
        }
    )

    wandb.finish()

##### Projection of performance to larger model sizes/#tokens using scaling laws #####

@dataclass
class ProjectionPoint:
    """Represents a point to project performance to"""
    num_params: int
    num_tokens: int


def get_default_projection_points() -> list[ProjectionPoint]:
    """Default set of model sizes to project to"""
    sizes = [1.4e9, 8e9, 13e9, 22e9, 70e9] 
    chinchilla_multipliers = [0.5, 1, 2, 5, 10, 20]
    
    return [ProjectionPoint(int(size), int(size * m)) for size in sizes for m in chinchilla_multipliers]

def create_projection_df(points: list[ProjectionPoint]) -> pd.DataFrame:
    """Convert projection points to a dataframe format matching scaling law requirements"""
    return pd.DataFrame({
        'parameter_count': [p.num_params for p in points],
        'throughput/total_tokens': [p.num_tokens for p in points],
        'run': [f'projection_{i}' for i in range(len(points))]
    })

def default_scaling_law_projection(
    ladder_runs: Sequence[str],
    intermediate_task_loss: str = "eval/paloma/c4_en/bpb",
    projection_points: list[ProjectionPoint] | None = None,
    entity: str = "stanford-mercury",
    project: str = "marin",
    use_log_for_ND: bool = True,
    normalize_ND: bool = True,
) -> tuple[np.ndarray, list[ProjectionPoint]]:
    """
    Project performance to larger model sizes using scaling laws.
    
    Args:
        ladder_runs: Sequence of run IDs to use for fitting scaling laws
        scaling_law_config: Configuration for scaling law analysis
        projection_points: Optional custom points to project to
        entity: WandB entity
        project: WandB project
    
    Returns:
        Tuple of (predicted_losses, projection_points)
    """
    from marin.scaling_laws.utils import (
        pull_metrics_from_wandb, filter_zero_d, extract_scaling_data,
        fit_power_law, predict_power_law
    )

    
    # Get or create projection points
    points = projection_points or get_default_projection_points()
    projection_df = create_projection_df(points)
    
    # Get ladder model data
    metrics = [intermediate_task_loss, "throughput/total_tokens"]
    ladder_df = pull_metrics_from_wandb(
        runs=ladder_runs,
        metrics=metrics,
        entity=entity,
        project=project
    )
    
    # Prepare data for fitting
    ladder_df_filtered = filter_zero_d(ladder_df)
    N, D, y = extract_scaling_data(
        ladder_df_filtered,
        loss_col=intermediate_task_loss
    )
    
    # Fit power law
    N_proj = np.array([p.num_params for p in points])
    D_proj = np.array([p.num_tokens for p in points])
    
    if use_log_for_ND:
        N = np.log(N)
        D = np.log(D)
        N_proj = np.log(N_proj)
        D_proj = np.log(D_proj)
    
    if normalize_ND:
        N_scale = np.mean(N)
        D_scale = np.mean(D)
        N = N / N_scale
        D = D / D_scale
        N_proj = N_proj / N_scale
        D_proj = D_proj / D_scale
    
    params = fit_power_law(N, D, y, use_log_space=True)
    predictions = predict_power_law(params, N_proj, D_proj)
    
    return predictions, points
