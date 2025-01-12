from dataclasses import dataclass

from marin.execution.executor import ExecutorStep


@dataclass(frozen=True)
class ScalingLawConfig:

    ladder_model_steps: list[ExecutorStep]
    """list of (smaller model) wandb run ids to be used as input for scaling laws"""

    pred_model_step: ExecutorStep
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
    from utils import (
        fit_accuracy_from_task_loss,
        fit_task_loss_from_ladder_models,
        plot_actual_vs_predicted,
    )

    input_run_ids = [get_wandb_run_id_from_step(step) for step in config.ladder_model_steps]
    pred_run_id = get_wandb_run_id_from_step(config.pred_model_step)

    # first predict task loss from (N, D) pairs
    actual, predicted = fit_task_loss_from_ladder_models(
        runs=input_run_ids,
        entity="stanford-mercury",
        project="marin",
        metrics=[config.task_loss, "throughput/total_tokens", config.task_accuracy],
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
        pred_task_losses=predicted,
        runs=config.input_run_ids,
        entity="stanford-mercury",
        project="marin",
        x_axis="throughput/total_gflops",
        tokens_col="throughput/total_tokens",
        pred_run=config.pred_run_id,
        aggregation="all",
        task_loss_col=config.task_loss,
        accuracy_col=config.task_accuracy,
    )

    # produce plots and write to wandb
    title = f"Prediction on larger run using {len(input_run_ids)} smaller models"
    plot_actual_vs_predicted(actual, predicted, config.task_accuracy, title)

    return actual_acc, pred_acc
