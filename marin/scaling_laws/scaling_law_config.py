from dataclasses import dataclass

@dataclass(frozen=True)
class ScalingLawConfig:

    input_run_ids: list[str]
    """list of (smaller model) wandb run ids to be used as input for scaling laws"""

    pred_run_id: str
    """wandb run id for the larger model to make predictions for"""

    task_loss: str = "eval/paloma/c4_en/bpb"
    """intermediate task loss to be used for scaling laws (eg. c4en bpb)"""

    task_accuracy: str = "lm_eval/hellaswag_10shot/acc"
    """task accuracy to predict for the larger model (eg. hellaswag accuracy)"""


def run_scaling_law_analysis(config: ScalingLawConfig) -> None:
    """
    Analyze scaling laws for a given task loss and accuracy.
    """
    from utils import *
    
    # first predict task loss from (N, D) pairs
    actual, predicted = fit_task_loss_from_ladder_models(
        runs=config.input_run_ids,
        entity="stanford-mercury",
        project="marin",
        metrics=[config.task_loss, "throughput/total_tokens", task_accuracy],
        pred_run=config.pred_run_id,
        task_loss=config.task_loss,
        aggregation="all",
        tokens_col="throughput/total_tokens",
        param_col="parameter_count",
        param_col_to_use="computed_params",
        use_log_for_ND=True,
        normalize_ND=True,
    )

    # plot_actual_vs_predicted(actual, predicted, title="Prediction on 7B Run using 5-sub 1.4B ladder models")

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

    return actual_acc, pred_acc

