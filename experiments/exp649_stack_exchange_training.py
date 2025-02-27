"""
Test different html->text transformation methods (on Stack Exchange Dump, train 1.4B models).
https://github.com/stanford-crfm/marin/issues/649
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

# Tokenized Threads from Stack Exchange, Threaded here means that
# we keep all the answers to the question in the same document, this also
# contains metadata like votes and tags. Template is:
#
# Question:
# <question>
# Answer:
# > <votes>
# <answer_1>
#
# Answer 2:
# > <votes>
# <answer_2>
#
# Tags: <tags>

stack_exchange_threaded_tokenized = default_tokenize(
    name="stack-exchange-threaded",
    dataset="",
    tokenizer=llama3_tokenizer,
)

stack_exchange_threaded_1_4b_model = default_train(
    name="stack-exchange-threaded-1.4b",
    tokenized=stack_exchange_threaded_tokenized,
    model_config=llama_1_4b,
    train_config=llama_1_4b_train_config,
)

stack_exchange_threaded_1_4b_evals = default_eval(step=stack_exchange_threaded_1_4b_model)

stack_exchange_accepted_answer_only_tokenized = default_tokenize(
    name="stack-exchange-accepted-answer-only",
    dataset="",
    tokenizer=llama3_tokenizer,
)

stack_exchange_accepted_answer_only_1_4b_model = default_train(
    name="stack-exchange-accepted-answer-only-1.4b",
    tokenized=stack_exchange_accepted_answer_only_tokenized,
    model_config=llama_1_4b,
    train_config=llama_1_4b_train_config,
)

stack_exchange_accepted_answer_only_1_4b_evals = default_eval(step=stack_exchange_accepted_answer_only_1_4b_model)

stack_exchange_isolated_tokenized = default_tokenize(
    name="stack-exchange-isolated",
    dataset="",
    tokenizer=llama3_tokenizer,
)

stack_exchange_isolated_1_4b_model = default_train(
    name="stack-exchange-isolated-1.4b",
    tokenized=stack_exchange_isolated_tokenized,
    model_config=llama_1_4b,
    train_config=llama_1_4b_train_config,
)

stack_exchange_isolated_1_4b_evals = default_eval(step=stack_exchange_isolated_1_4b_model)


if __name__ == "__main__":
    executor_main(
        steps=[
            stack_exchange_threaded_tokenized,
            stack_exchange_threaded_1_4b_model,
            stack_exchange_threaded_1_4b_evals,
            stack_exchange_accepted_answer_only_tokenized,
            stack_exchange_accepted_answer_only_1_4b_model,
            stack_exchange_accepted_answer_only_1_4b_evals,
            stack_exchange_isolated_tokenized,
            stack_exchange_isolated_1_4b_model,
            stack_exchange_isolated_1_4b_evals,
        ]
    )
