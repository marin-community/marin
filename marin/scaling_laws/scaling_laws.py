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
from dataclasses import dataclass, field

import numpy as np
import wandb
from matplotlib import pyplot as plt

from marin.execution.executor import ExecutorStep
from marin.scaling_laws.utils import ProjectionPoint, get_default_projection_points


@dataclass(frozen=True)
class ScalingLawConfig:

    ladder_model_steps: Sequence[ExecutorStep | str]
    """list of (smaller model) steps or wandb run ids to be used as input for scaling laws"""

    pred_model_step: ExecutorStep | str
    """executor step or wandb run id for the larger model to make predictions for"""

    projection_points: Sequence[ProjectionPoint] | None = None  # Predict for N,D points
    """Points to project to"""

    task_losses: Sequence[str] = field(default_factory=lambda: ["eval/paloma/c4_en/bpb"])
    """task losses to predict for scaling laws (eg. c4en bpb)"""

    task_accuracies: Sequence[str] = field(default_factory=lambda: ["lm_eval/hellaswag_10shot/acc"])
    """task accuracy to predict for the larger model (eg. hellaswag accuracy)"""

    use_log_for_ND: bool = True
    """whether to use log space for N,D in scaling laws"""

    normalize_ND: bool = True
    """whether to normalize N,D in scaling laws"""

    entity: str = "stanford-mercury"
    project: str = "marin"


def get_wandb_run_id_from_step(step: ExecutorStep) -> str:
    """
    Get the wandb run id from a given ExecutorStep.
    """
    return step.config.trainer.tracker.id


def run_scaling_law_analysis(config: ScalingLawConfig) -> None:
    """
    Analyze scaling laws for a given task loss and multiple accuracy metrics.
    """
    from marin.scaling_laws.utils import fit_scaling_laws

    input_run_ids = [
        get_wandb_run_id_from_step(step) if isinstance(step, ExecutorStep) else step
        for step in config.ladder_model_steps
    ]

    pred_run_id = None
    if config.pred_model_step:
        pred_run_id = (
            get_wandb_run_id_from_step(config.pred_model_step)
            if isinstance(config.pred_model_step, ExecutorStep)
            else config.pred_model_step
        )

    projections, predictions, points = fit_scaling_laws(
        runs=input_run_ids,
        loss_metrics=config.task_losses,
        accuracy_metrics=config.task_accuracies,
        entity=config.entity,
        project=config.project,
        pred_run=pred_run_id,
        projection_points=config.projection_points or get_default_projection_points(),
        use_log_for_ND=config.use_log_for_ND,
        normalize_ND=config.normalize_ND,
    )

    log_and_create_report(
        projections=projections,
        points=points if config.projection_points else get_default_projection_points(),
        predictions=predictions,
        input_run_ids=input_run_ids,
        pred_run_id=pred_run_id,
        scaling_law_config=config,
    )


def log_and_create_report(
    projections: dict[str, np.ndarray],
    points: list[ProjectionPoint],
    predictions: tuple[dict, dict] | None,
    input_run_ids: list,
    pred_run_id: str | None,
    scaling_law_config: ScalingLawConfig,
    wandb_project: str = "marin-scaling-laws",
    wandb_entity: str = "stanford-mercury",
):
    """
    Logs scaling law analysis creates a concise WandB report with plots and info about runs.
    """
    # Initialize WandB run
    run = wandb.init(
        project=wandb_project,
        entity=wandb_entity,
        name=f"""Scaling Law Report: {pred_run_id if pred_run_id else 'projection'}- bpb""",
        tags=["scaling_laws"],
        config={
            "input_runs": input_run_ids,
            "prediction_run": pred_run_id,
        },
        reinit=True,
    )

    # Create plots dictionary for wandb.log
    plots = {}

    # Log projections
    for loss_name, projection in projections.items():
        plt.figure()
        plot_scaling_projections(projection, points)
        plots[f"Projection - {loss_name}"] = wandb.Image(plt)
        plt.close()

    if predictions:

        loss_results, accuracy_results = predictions

        if loss_results:
            for loss_name, (actual_loss, predicted_loss) in loss_results.items():
                steps = list(range(len(actual_loss)))
                plots[f"Task Loss - {loss_name}"] = wandb.plot.line_series(
                    xs=steps,
                    ys=[actual_loss.tolist(), predicted_loss.tolist()],
                    keys=["Actual", "Predicted"],
                    title=f"Task Loss: {loss_name}",
                    xname="Step",
                    split_table=True,
                )

        if accuracy_results:
            # Add accuracy plots for each metric
            for metric, (actual_acc, predicted_acc) in accuracy_results.items():
                steps = list(range(len(actual_acc)))
                plots[f"Task Accuracy - {metric}"] = wandb.plot.line_series(
                    xs=steps,
                    ys=[actual_acc.tolist(), predicted_acc.tolist()],
                    keys=["Actual", "Predicted"],
                    title=f"Task Accuracy ({metric})",
                    xname="Step",
                    split_table=True,
                )

    # Log all plots
    wandb.log(plots)

    # Info about runs and links
    input_run_links = [f"https://wandb.ai/stanford-mercury/marin/runs/{run_id}" for run_id in input_run_ids]
    prediction_run_link = f"https://wandb.ai/stanford-mercury/marin/runs/{pred_run_id}" if pred_run_id else None
    run.summary.update(
        {
            "Input Runs": input_run_links,
            "Prediction Run": prediction_run_link,
            "Task Losses": scaling_law_config.task_losses,
            "Task Accuracies": scaling_law_config.task_accuracies,
        }
    )

    wandb.finish()


##### Projection of performance to larger model sizes/#tokens using scaling laws #####


def plot_scaling_projections(predicted: np.ndarray, points: list[ProjectionPoint]):
    """Plot scaling law predictions vs tokens for specified or all model sizes"""
    plt.figure(figsize=(12, 6))
    unique_params = np.unique([p.num_params for p in points])

    for param in unique_params:
        mask = np.array([p.num_params == param for p in points])
        tokens = np.array([p.num_tokens for p in points])[mask]
        preds = predicted[mask]

        plt.plot(tokens, preds, "o-", linewidth=2, label=f"{param/1e9:.1f}B params")
        for t, pred in zip(tokens, preds, strict=False):
            token_str = f"{t/1e9:.1f}B" if t < 1e11 else f"{t/1e12:.1f}T"
            plt.annotate(f"{token_str}, {pred:.3f}", (t, pred), ha="center", va="bottom", fontsize=8)

    plt.xscale("log")
    plt.xlabel("Number of Tokens")
    plt.ylabel("Predicted Loss")
    plt.grid(True)
    plt.legend()
    return plt
