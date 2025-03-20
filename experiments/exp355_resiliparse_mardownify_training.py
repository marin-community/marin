"""
Test the Resiliparse custom fork on the fineweb-small dataset.
https://github.com/stanford-crfm/marin/issues/355
"""

import logging

from experiments.defaults import default_tokenize, default_train
from experiments.evals.evals import default_eval
from experiments.llama import llama3_tokenizer, llama_1_4b, llama_1_4b_train_config
from marin.execution.executor import executor_main

logger = logging.getLogger("ray")
step_name = "fineweb-small-resiliparse-custom-fork"

fineweb_resiliparse_custom_fork_tokenized = default_tokenize(
    name=step_name,
    dataset="gs://marin-us-central2/documents/fineweb-small-resiliparse-custom-fork-ca2156/md/CC-MAIN-2024-18",
    tokenizer=llama3_tokenizer,
)
fineweb_resiliparse_custom_fork_1_4b_model = default_train(
    name=f"{step_name}-1.4b",
    tokenized=fineweb_resiliparse_custom_fork_tokenized,
    model_config=llama_1_4b,
    train_config=llama_1_4b_train_config,
)

fineweb_resiliparse_custom_fork_1_4b_model_eval = default_eval(fineweb_resiliparse_custom_fork_1_4b_model)

if __name__ == "__main__":
    executor_main(
        steps=[
            fineweb_resiliparse_custom_fork_tokenized,
            fineweb_resiliparse_custom_fork_1_4b_model,
            fineweb_resiliparse_custom_fork_1_4b_model_eval,
        ]
    )
