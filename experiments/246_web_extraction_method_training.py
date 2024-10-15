import logging
from dataclasses import dataclass

import draccus

from experiments.defaults import default_tokenize, default_train, llama_1_4b_train_config
from experiments.llama import llama3_tokenizer, llama_1_4b
from marin.evaluation.evaluation_config import EvaluationConfig
from marin.evaluation.run import evaluate
from marin.execution.executor import (
    ExecutorStep,
    executor_main,
    output_path_of,
    this_output_path,
    versioned,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("ray")


@dataclass(frozen=True)
class WebExtractionMethodConfig:
    """
    Configuration for the quickstart executor

    Attributes:
    - extracted_data: str: The path to the data to train the model on based on the extraction method
    - extraction_method_name: str: The name of the extraction method
    """

    extracted_data: str
    extraction_method_name: str


@draccus.wrap()
def create_steps(config: WebExtractionMethodConfig) -> list[ExecutorStep]:
    fw_tokenized = default_tokenize(
        name=f"fw-small-100B-{config.extraction_method_name}",
        dataset=config.extracted_data,
        tokenizer=llama3_tokenizer,
    )
    fw_100b_model = default_train(
        name=f"fw-small-100B-1.4b-{config.extraction_method_name}",
        tokenized=fw_tokenized,
        model_config=llama_1_4b,
        train_config=llama_1_4b_train_config,
    )

    evaluate_step = ExecutorStep(
        name=f"evaluation/fw-small-{config.extraction_method_name}",
        fn=evaluate,
        config=EvaluationConfig(
            evaluator="helm",
            model_name=versioned(config.extraction_method_name),
            model_path=output_path_of(fw_100b_model),
            evaluation_path=this_output_path(),
            evals=["mmlu"],
        ),
    )

    return [
        fw_tokenized,
        fw_100b_model,
        evaluate_step,
    ]


if __name__ == "__main__":
    steps = create_steps()
    executor_main(steps=steps)
