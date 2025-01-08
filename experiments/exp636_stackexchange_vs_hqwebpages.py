"""Experiment comparing training on StackExchange filtered web pages vs. training on StackExchange directly.

StackExchange is a dataset of high quality web pages, but the dataset is small. Web pages from FineWeb
are plentiful but they are not as high quality as StackExchange. We compare a model trained on
many epochs of StackExchange with a model trained on a single epoch of FineWeb.
"""

from experiments.defaults import default_tokenize, default_train
from experiments.dolma.tokenize_dolma import tokenize_dolma_steps
from experiments.evals.evals import default_eval
from experiments.exp596_stackexchange_classifier import stackexchange_experiment_config
from experiments.llama import llama3_tokenizer, llama_1_4b, llama_1_4b_train_config
from experiments.pretraining_datasets import dolmino
from experiments.quality_classifier_experiment_utils import create_steps
from marin.execution.executor import ExecutorStep, executor_main, this_output_path
from operations.transform.dolmino.filter_dolmino import FilterDolminoConfig, filter_dolmino

dolmino_stackexchange_jsonl = ExecutorStep(
    name="documents/dolmino_stackexchange",
    fn=filter_dolmino,
    config=FilterDolminoConfig(
        input_path=dolmino,
        output_path=this_output_path(),
        split="stackexchange",
        min_length=0,
    ),
)

dolmino_stackexchange_tokenized = default_tokenize(
    name="quality_filtering/dolmino_stackexchange",
    dataset=dolmino_stackexchange_jsonl,
    tokenizer=llama3_tokenizer,
)

dolmino_stackexchange_model = default_train(
    name="quality_filtering/dolmino_stackexchange",
    tokenized=dolmino_stackexchange_tokenized,
    model_config=llama_1_4b,
    train_config=llama_1_4b_train_config,
)

dolmino_stackexchange_eval = default_eval(dolmino_stackexchange_model)

dolma_stackexchange_tokenized = tokenize_dolma_steps()["dolma/stackexchange"]

dolma_stackexchange_model = default_train(
    name="quality_filtering/dolma_stackexchange",
    tokenized=dolma_stackexchange_tokenized,
    model_config=llama_1_4b,
    train_config=llama_1_4b_train_config,
)

dolma_stackexchange_eval = default_eval(dolma_stackexchange_model)

stackexchange_filtered_hq_webpages_experiment_steps = create_steps(stackexchange_experiment_config)

# NOTE(chris): Normally this is not required because default_train will eval on the last step, but I had not
# pushed the changes to the repo when I created this experiment, which is why I'm evaluating after the run.
stackexchange_filtered_hq_webpages_eval = default_eval(stackexchange_filtered_hq_webpages_experiment_steps[-1])
stackexchange_filtered_hq_webpages_experiment_steps.append(stackexchange_filtered_hq_webpages_eval)

if __name__ == "__main__":
    steps = [
        dolmino_stackexchange_eval,
        dolma_stackexchange_eval,
    ]
    steps.extend(stackexchange_filtered_hq_webpages_experiment_steps)
    executor_main(steps=steps)
