"""
Test different html->text transformation methods (on FineWeb, train 1.4B models).
https://github.com/stanford-crfm/marin/issues/246
"""

import logging

from experiments.defaults import default_tokenize, default_train
from experiments.llama import llama3_tokenizer, llama_1_4b, llama_1_4b_train_config
from marin.execution.executor import (
    ExecutorStep,
    executor_main,
)

logger = logging.getLogger("ray")


def create_steps() -> list[ExecutorStep]:
    step_name = "fineweb-small-resiliparse-custom-fork"

    fw_tokenized = default_tokenize(
        name=step_name,
        dataset="gs://marin-us-central2/documents/fineweb-small-resiliparse-custom-fork-ca2156/md/CC-MAIN-2024-18",
        tokenizer=llama3_tokenizer,
    )
    fw_100b_model = default_train(
        name=f"{step_name}-1.4b",
        tokenized=fw_tokenized,
        model_config=llama_1_4b,
        train_config=llama_1_4b_train_config,
    )

    return [
        fw_tokenized,
        fw_100b_model,
    ]


if __name__ == "__main__":
    executor_main(steps=create_steps())
