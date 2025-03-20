"""
This experiment compares different settings for converting Stack Exchange HTML to text for training.
We use Resiliparse to extract the main content from the Stack Exchange HTML dumps and then convert the text
to markdown using markdownify. We generate 3 variants that are:

* Threaded: All the answers to the question are in the same thread as the question, this also
            contains metadata like votes and tags.
* Accepted Answer Only: Only the accepted answer is in the QA pair.
* Isolated: Each question and answer is a separate document.

We then train 1.4B parameter models on these modified datasets to evaluate which preprocessing
approach produces better training data for language models. Idea is to seen which configuration of
social media content produces better training data for language models.

Reference Issue: https://github.com/stanford-crfm/marin/issues/649
"""

import logging

from experiments.defaults import default_tokenize, default_train
from experiments.evals.evals import default_eval
from experiments.exp822_stackexchange_markdownify import stackexchange_text_resiliparse_custom_fork
from experiments.llama import llama3_tokenizer, llama_1_4b, llama_1_4b_train_config
from marin.execution.executor import executor_main

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("ray")

# Tokenize the threaded markdown
stack_exchange_threaded_tokenized = default_tokenize(
    name="stack-exchange-threaded",
    dataset=stackexchange_text_resiliparse_custom_fork,
    tokenizer=llama3_tokenizer,
)

# Train the threaded model
stack_exchange_threaded_1_4b_model = default_train(
    name="stack-exchange-threaded-1.4b",
    tokenized=stack_exchange_threaded_tokenized,
    model_config=llama_1_4b,
    train_config=llama_1_4b_train_config,
)

# Evaluate the threaded model
stack_exchange_threaded_1_4b_evals = default_eval(step=stack_exchange_threaded_1_4b_model)

# Tokenize the accepted answer only markdown
stack_exchange_accepted_answer_only_tokenized = default_tokenize(
    name="stack-exchange-accepted-answer-only",
    dataset="gs://marin-us-central2/documents/stackexchange/v2024-04-02/md-seperate",
    tokenizer=llama3_tokenizer,
)

# Train the accepted answer only model
stack_exchange_accepted_answer_only_1_4b_model = default_train(
    name="stack-exchange-accepted-answer-only-1.4b",
    tokenized=stack_exchange_accepted_answer_only_tokenized,
    model_config=llama_1_4b,
    train_config=llama_1_4b_train_config,
)

# Evaluate the accepted answer only model
stack_exchange_accepted_answer_only_1_4b_evals = default_eval(step=stack_exchange_accepted_answer_only_1_4b_model)

# Tokenize the isolated markdown
stack_exchange_isolated_tokenized = default_tokenize(
    name="stack-exchange-isolated",
    dataset="gs://marin-us-central2/documents/stackexchange/v2024-04-02/md-qa-pair",
    tokenizer=llama3_tokenizer,
)

# Train the isolated model
stack_exchange_isolated_1_4b_model = default_train(
    name="stack-exchange-isolated-1.4b",
    tokenized=stack_exchange_isolated_tokenized,
    model_config=llama_1_4b,
    train_config=llama_1_4b_train_config,
)

# Evaluate the isolated model
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
