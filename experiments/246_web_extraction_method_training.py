import logging
import os
from dataclasses import dataclass

import draccus
from levanter.models.llama import LlamaConfig
from levanter.trainer import TrainerConfig

from marin.evaluation.evaluation_config import EvaluationConfig
from marin.evaluation.run import evaluate
from marin.execution.executor import (
    ExecutorMainConfig,
    ExecutorStep,
    executor_main,
    output_path_of,
    this_output_path,
    versioned,
)
from marin.processing.tokenize import TokenizeConfig, lm_training_config, tokenize
from marin.training.training import TrainLmOnPodConfig, run_levanter_train_lm

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("ray")

USER = "herumb"


@dataclass(frozen=True)
class WebExtractionMethodConfig(ExecutorMainConfig):
    """
    Configuration for the quickstart executor

    Attributes:
    - extracted_data: str: The path to the data to train the model on based on the extraction method
    - extraction_method_name: str: The name of the extraction method
    """

    extracted_data: str = ""
    extraction_method_name: str = ""


@draccus.wrap()
def create_steps(config: WebExtractionMethodConfig) -> list[ExecutorStep]:
    ############################################################
    # Tokenize

    tokenize_step = ExecutorStep(
        name=os.path.join(config.prefix, config.extraction_method_name, "tokenized"),
        fn=tokenize,
        config=TokenizeConfig(
            train_paths=config.extracted_data,
            validation_paths=[],
            cache_path=this_output_path(),
            tokenizer=versioned("llama2"),
        ),
    )

    # ############################################################
    # # Train

    train_step = ExecutorStep(
        name=os.path.join(config.prefix, config.commit_hash, "train"),
        fn=run_levanter_train_lm,
        config=TrainLmOnPodConfig(
            output_path=this_output_path(),
            data=lm_training_config(tokenize_step),
            env={"WANDB_API_KEY": None},  # Add your wandb key here
            tpu_type="v4-8",
            hf_save_steps=1,
            model=LlamaConfig(
                seq_len=2048,
                hidden_dim=2048,
                intermediate_dim=2048,
                num_layers=24,
                num_heads=16,
            ),
            trainer=TrainerConfig(train_batch_size=1, num_train_steps=2, max_eval_batches=1, require_accelerator=False),
        ),
    )

    evaluate_step = ExecutorStep(
        name=f"evaluation/fw-small-{config.extraction_method_name}-{USER}",
        fn=evaluate,
        config=EvaluationConfig(
            evaluator="helm",
            model_name=versioned(config.extraction_method_name),
            model_path=output_path_of(train_step),
            evaluation_path=this_output_path(),
            evals=["mmlu"],
        ),
    )

    return [
        tokenize_step,
        train_step,
        evaluate_step,
    ]


if __name__ == "__main__":
    steps = create_steps()
    executor_main(steps=steps)
