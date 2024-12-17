"""
Test different html->text transformation methods (on Ar5iv Dump, train 1.4B models).
https://github.com/stanford-crfm/marin/issues/648
"""

import logging

from experiments.defaults import default_tokenize, default_train
from experiments.evals.evals import default_eval
from experiments.llama import llama3_tokenizer, llama_1_4b, llama_1_4b_train_config
from marin.execution.executor import (
    executor_main,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("ray")


ar5iv_no_problem_readablity_tokenized = default_tokenize(
    name="fineweb-small-readablity",
    dataset="",
    tokenizer=llama3_tokenizer,
)

ar5iv_no_problem_readablity_1_4b_model = default_train(
    name="fineweb-small-1.4b-readablity",
    tokenized=ar5iv_no_problem_readablity_tokenized,
    model_config=llama_1_4b,
    train_config=llama_1_4b_train_config,
)

ar5iv_no_problem_readablity_1_4b_evals = default_eval(step=ar5iv_no_problem_readablity_1_4b_model)


ar5iv_no_problem_resiliparse_with_preserve_formatting_tokenized = default_tokenize(
    name="fineweb-small-resiliparse-with-preserving-formatting",
    dataset="",
    tokenizer=llama3_tokenizer,
)

ar5iv_no_problem_resiliparse_with_preserve_formatting_1_4b_model = default_train(
    name="fineweb-small-1.4b-resiliparse-with-preserving-formatting",
    tokenized=ar5iv_no_problem_resiliparse_with_preserve_formatting_tokenized,
    model_config=llama_1_4b,
    train_config=llama_1_4b_train_config,
)

ar5iv_no_problem_resiliparse_with_preserve_formatting_1_4b_evals = default_eval(step=ar5iv_no_problem_resiliparse_with_preserve_formatting_1_4b_model)


if __name__ == "__main__":
    executor_main(
        steps=[
            ar5iv_no_problem_readablity_tokenized,
            ar5iv_no_problem_readablity_1_4b_model,
            ar5iv_no_problem_readablity_1_4b_evals,
            ar5iv_no_problem_resiliparse_with_preserve_formatting_tokenized,
            ar5iv_no_problem_resiliparse_with_preserve_formatting_1_4b_model,
            ar5iv_no_problem_resiliparse_with_preserve_formatting_1_4b_evals,
        ]
    )
