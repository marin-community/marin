from dataclasses import dataclass

import wandb

from marin.execution.executor import ExecutorStep
from marin.scaling_laws.utils import plot_actual_vs_predicted, plot_fit


@dataclass(frozen=True)
class ScalingLawConfig:

    ladder_model_steps: list[ExecutorStep | str]
    """list of (smaller model) wandb run ids to be used as input for scaling laws"""

    pred_model_step: ExecutorStep | str
    """wandb run id for the larger model to make predictions for"""

    intermediate_task_loss: str = "eval/paloma/c4_en/bpb"
    """intermediate task loss to be used for scaling laws (eg. c4en bpb)"""

    task_accuracy: str = "lm_eval/hellaswag_10shot/acc"
    """task accuracy to predict for the larger model (eg. hellaswag accuracy)"""


def get_wandb_run_id_from_step(step: ExecutorStep) -> str:
    """
    Get the wandb run id from a given ExecutorStep.
    """
    return step.config.trainer.tracker.id


def run_scaling_law_analysis(config: ScalingLawConfig) -> None:
    """
    Analyze scaling laws for a given task loss and accuracy.
    """
    from marin.scaling_laws.utils import (
        fit_accuracy_from_task_loss,
        fit_task_loss_from_ladder_models,
    )

    input_run_ids = [
        get_wandb_run_id_from_step(step) if isinstance(step, ExecutorStep) else step
        for step in config.ladder_model_steps
    ]

    if isinstance(config.pred_model_step, ExecutorStep):
        pred_run_id = get_wandb_run_id_from_step(config.pred_model_step)
    else:
        pred_run_id = config.pred_model_step

    print(f"Input runs: {input_run_ids}")
    print(f"Prediction run: {pred_run_id}")
    print(f"Intermediate task loss: {config.intermediate_task_loss}")
    print(f"Task accuracy: {config.task_accuracy}")

    # first predict task loss from (N, D) pairs
    actual_loss, predicted_loss = fit_task_loss_from_ladder_models(
        runs=input_run_ids,
        entity="stanford-mercury",
        project="marin",
        metrics=[config.intermediate_task_loss, "throughput/total_tokens", config.task_accuracy],
        pred_run=pred_run_id,
        task_loss=config.intermediate_task_loss,
        aggregation="all",
        tokens_col="throughput/total_tokens",
        param_col="parameter_count",
        param_col_to_use="computed_params",
        use_log_for_ND=True,
        normalize_ND=True,
    )

    # then predict task accuracy from task loss
    actual_acc, pred_acc = fit_accuracy_from_task_loss(
        pred_task_losses=predicted_loss,
        runs=input_run_ids,
        entity="stanford-mercury",
        project="marin",
        x_axis="throughput/total_gflops",
        tokens_col="throughput/total_tokens",
        pred_run=pred_run_id,
        aggregation="all",
        task_loss_col=config.intermediate_task_loss,
        accuracy_col=config.task_accuracy,
    )

    print("Made predictions on accuracy and loss. Going to create a report.")

    # Log and create a report
    log_and_create_report(
        actual_loss=actual_loss.tolist(),
        predicted_loss=predicted_loss.tolist(),
        actual_acc=actual_acc.tolist(),
        predicted_acc=pred_acc.tolist(),
        input_run_ids=input_run_ids,
        pred_run_id=pred_run_id,
        scaling_law_config=config,
        wandb_project="marin-scaling-laws",
        wandb_entity="stanford-mercury",
    )


def log_and_create_report(
    actual_loss: list,
    predicted_loss: list,
    actual_acc: list,
    predicted_acc: list,
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
        actual_acc (list): List of actual task accuracy values.
        predicted_acc (list): List of predicted task accuracy values.
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
        name=f"Scaling Law Report: {pred_run_id}",
        tags=["scaling_laws"],
        config={
            "input_runs": input_run_ids,
            "prediction_run": pred_run_id,
        },
        reinit=True,
    )

    # Task Loss Comparison Plot
    task_loss_plot = plot_actual_vs_predicted(
        actual_loss, predicted_loss, title=f"Prediction on larger run using {len(input_run_ids)} smaller models"
    )
    wandb.log({"Task Loss Comparison": wandb.Image(task_loss_plot)})

    # Task Loss fit plot
    task_loss_fit_plot = plot_fit(
        actual_loss, predicted_loss, title=f"Prediction on larger run using {len(input_run_ids)} smaller models"
    )
    wandb.log({"Task Loss Fit": wandb.Image(task_loss_fit_plot)})

    # Task Accuracy Comparison Plot
    task_accuracy_plot = plot_actual_vs_predicted(
        actual_acc,
        predicted_acc,
        title=f"Prediction on larger run using {len(input_run_ids)} smaller models",
        task_metric=scaling_law_config.task_accuracy,
    )
    wandb.log({"Task Accuracy Comparison": wandb.Image(task_accuracy_plot)})

    # Task Accuracy fit plot
    task_accuracy_fit_plot = plot_fit(
        actual_acc,
        predicted_acc,
        title=f"Prediction on larger run using {len(input_run_ids)} smaller models",
    )
    wandb.log({"Task Accuracy Fit": wandb.Image(task_accuracy_fit_plot)})

    # Info about runs and links
    input_run_links = [f"https://wandb.ai/stanford-mercury/marin/runs/{run_id}" for run_id in input_run_ids]
    prediction_run_link = f"https://wandb.ai/stanford-mercury/marin/runs/{pred_run_id}"
    run.summary.update(
        {
            "Input Runs": input_run_links,
            "Prediction Run": prediction_run_link,
            "Task Loss Metric": scaling_law_config.intermediate_task_loss,
            "Task Accuracy Metric": scaling_law_config.task_accuracy,
        }
    )

    wandb.finish()
