from experiments.defaults import default_tokenize, default_train
from experiments.dolma.tokenize_dolma import tokenize_dolma_steps
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

dolma_stackexchange_tokenized = tokenize_dolma_steps()["dolma/stackexchange"]

dolma_stackexchange_model = default_train(
    name="quality_filtering/dolma_stackexchange",
    tokenized=dolma_stackexchange_tokenized,
    model_config=llama_1_4b,
    train_config=llama_1_4b_train_config,
)

_, stackexchange_filtered_hq_webpages_experiment_steps = create_steps(stackexchange_experiment_config)

if __name__ == "__main__":
    steps = [
        dolmino_stackexchange_model,
        dolma_stackexchange_model,
    ]
    steps.extend(stackexchange_filtered_hq_webpages_experiment_steps)
    executor_main(steps=steps)
