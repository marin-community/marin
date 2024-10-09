import logging
import os
from dataclasses import dataclass

import draccus
from levanter.models.gpt2 import Gpt2Config
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
logger = logging.getLogger(__name__)

USER = "herumb"


@dataclass
class QuickstartExecutorConfig:
    extracted_data: str
    extraction_method_name: str


def create_steps(config: QuickstartExecutorConfig) -> list[ExecutorStep]:
    ############################################################
    # Tokenize

    tokenize_step = ExecutorStep(
        name=os.path.join(config.prefix, config.commit_hash, "tokenized"),
        fn=tokenize,
        config=TokenizeConfig(
            train_paths=config.extracted_data,
            validation_paths=[],
            cache_path=this_output_path(),
            tokenizer=versioned("gpt2"),
        ),
    )

    ############################################################
    # Train

    train_step = ExecutorStep(
        name=os.path.join(config.prefix, config.commit_hash, "train"),
        fn=run_levanter_train_lm,
        config=TrainLmOnPodConfig(
            output_path=this_output_path(),
            data=lm_training_config(tokenize_step),
            env={"WANDB_API_KEY": None},
            tpu_type=None,
            hf_save_steps=1,
            model=Gpt2Config(
                num_layers=2,
                num_heads=2,
                seq_len=64,
                hidden_dim=32,
            ),
            trainer=TrainerConfig(train_batch_size=1, num_train_steps=2, max_eval_batches=1, require_accelerator=False),
        ),
    )

    evaluate_step = ExecutorStep(
        name=f"evaluation/hello_world_fw-{USER}",
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


@draccus.wrap()
def main(config: QuickstartExecutorConfig):
    try:
        steps = create_steps(config)
        bucket_prefix = "/tmp"
        config_executor = ExecutorMainConfig(
            prefix=bucket_prefix, executor_info_base_path=os.path.join(bucket_prefix, "experiments")
        )
        executor_main(config_executor, steps=steps)
        logger.info(
            f"Execution completed successfully. All outputs are in {bucket_prefix}/{config.prefix}/{config.commit_hash}"
        )
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        raise e


if __name__ == "__main__":
    main()
