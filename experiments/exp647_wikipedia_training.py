"""
Test different html->text transformation methods (on Wikipedia Dump, train 1.4B models).
https://github.com/stanford-crfm/marin/issues/647
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


wiki_readablity_tokenized = default_tokenize(
    name="fineweb-small-readablity",
    dataset="",
    tokenizer=llama3_tokenizer,
)

wiki_readablity_1_4b_model = default_train(
    name="fineweb-small-1.4b-readablity",
    tokenized=wiki_readablity_tokenized,
    model_config=llama_1_4b,
    train_config=llama_1_4b_train_config,
)

wiki_readablity_1_4b_evals = default_eval(step=wiki_readablity_1_4b_model)


wiki_resiliparse_with_preserve_formatting_tokenized = default_tokenize(
    name="fineweb-small-resiliparse-with-preserving-formatting",
    dataset="",
    tokenizer=llama3_tokenizer,
)

wiki_resiliparse_with_preserve_formatting_1_4b_model = default_train(
    name="fineweb-small-1.4b-resiliparse-with-preserving-formatting",
    tokenized=wiki_resiliparse_with_preserve_formatting_tokenized,
    model_config=llama_1_4b,
    train_config=llama_1_4b_train_config,
)

wiki_resiliparse_with_preserve_formatting_1_4b_evals = default_eval(step=wiki_resiliparse_with_preserve_formatting_1_4b_model)


if __name__ == "__main__":
    executor_main(
        steps=[
            wiki_readablity_tokenized,
            wiki_readablity_1_4b_model,
            wiki_readablity_1_4b_evals,
            wiki_resiliparse_with_preserve_formatting_tokenized,
            wiki_resiliparse_with_preserve_formatting_1_4b_model,
            wiki_resiliparse_with_preserve_formatting_1_4b_evals,
        ]
    )
