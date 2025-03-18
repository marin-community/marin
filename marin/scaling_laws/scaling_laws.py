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

from marin.execution.executor import ExecutorStep
from marin.scaling_laws.utils import (
    ProjectionPoint,
    get_default_projection_points,
    plot_actual_vs_predicted,
    plot_scaling_projections,
)


@dataclass(frozen=True)
class ScalingLawConfig:
    name: str
    """name of the scaling law analysis or config (used for the report name)"""

    ladder_model_steps: Sequence[ExecutorStep | str]
    """list of (smaller model) steps or wandb run ids to be used as input for scaling laws"""

    pred_model_step: ExecutorStep | str
    """executor step or wandb run id for the larger model to make predictions for"""

    projection_points: list[ProjectionPoint] | None = None
    """Points to project to, consisting of number of parameters and tokens"""

    task_losses: Sequence[str] = field(default_factory=lambda: ["eval/paloma/c4_en/bpb"])
    """task losses to predict for scaling laws (eg. c4en bpb)"""

    task_accuracies: Sequence[str] | None = None
    """task accuracy to predict for the larger model (eg. hellaswag accuracy)"""

    use_log_for_ND: bool = True
    """whether to use log space for N,D in scaling laws"""

    normalize_ND: bool = True
    """whether to normalize N,D in scaling laws"""

    count_embedding_params: bool = False
    """whether to count embedding parameters in scaling laws"""

    entity: str = "stanford-mercury"
    project: str = "marin"

    def __post_init__(self):
        # Set default projection points if none provided
        if self.projection_points is None:
            object.__setattr__(
                self,
                "projection_points",
                get_default_projection_points(count_embedding_params=self.count_embedding_params),
            )


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

    projections, predictions = fit_scaling_laws(
        runs=input_run_ids,
        loss_metrics=config.task_losses,
        accuracy_metrics=config.task_accuracies,
        entity=config.entity,
        project=config.project,
        pred_run=pred_run_id,
        projection_points=config.projection_points,
        count_embedding_params=config.count_embedding_params,
        use_log_for_ND=config.use_log_for_ND,
        normalize_ND=config.normalize_ND,
    )

    log_and_create_report(
        projections=projections,
        points=config.projection_points,
        predictions=predictions,
        input_run_ids=input_run_ids,
        pred_run_id=pred_run_id,
        scaling_law_config=config,
    )


def log_and_create_report(
    projections: dict[str, np.ndarray],
    points: list[ProjectionPoint] | None,
    predictions: tuple[dict, dict, np.ndarray, np.ndarray] | None,
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
        name=f"""Scaling Law Report: {pred_run_id if pred_run_id else 'projection'}-{scaling_law_config.name}""",
        tags=["scaling_laws"],
        config={
            "input_runs": input_run_ids,
            "prediction_run": pred_run_id,
        },
        reinit=True,
    )

    plots = {}

    # Log projections
    if points:
        for loss_name, projection in projections.items():
            figure = plot_scaling_projections(projection, points)
            plots[f"Projection - {loss_name}"] = wandb.Image(figure)

    # Log predictions if available
    if predictions:
        loss_results, accuracy_results, loss_tokens, acc_tokens = predictions

        if loss_results:
            for loss_name, (actual_loss, predicted_loss) in loss_results.items():
                figure = plot_actual_vs_predicted(
                    actual_loss.tolist(),
                    predicted_loss.tolist(),
                    title=f"Actual vs Predicted {loss_name}",
                    task_metric=loss_name,
                    tokens=loss_tokens,
                )
                plots[f"Task Loss - {loss_name}"] = wandb.Image(figure)

        if accuracy_results:
            for metric, (actual_acc, predicted_acc) in accuracy_results.items():
                figure = plot_actual_vs_predicted(
                    actual_acc.tolist(),
                    predicted_acc.tolist(),
                    title=f"Actual vs Predicted {metric}",
                    task_metric=metric,
                    tokens=acc_tokens,
                )
                plots[f"Task Accuracy - {metric}"] = wandb.Image(figure)

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
