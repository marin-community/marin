from experiments.defaults import default_tokenize, default_train
from experiments.exp596_stackexchange_classifier import stackexchange_experiment_config
from experiments.llama import llama3_tokenizer, llama_1_4b, llama_1_4b_train_config
from experiments.pretraining_datasets import dolmino
from experiments.quality_classifier_experiment_utils import create_steps
from marin.execution.executor import ExecutorStep, executor_main, this_output_path
from marin.processing.tokenize.data_configs import lm_mixture_data_config
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

stackexchange_filtered_hq_webpages_experiment_dict, stackexchange_filtered_hq_webpages_experiment_steps = create_steps(
    stackexchange_experiment_config
)

stackexchange_filtered_hq_webpages_model = stackexchange_filtered_hq_webpages_experiment_dict[
    f"{stackexchange_experiment_config.experiment_name}-train"
]

stackexchange_filtered_hq_webpages_tokenized = stackexchange_filtered_hq_webpages_experiment_dict[
    f"{stackexchange_experiment_config.experiment_name}-tokenize"
]

data_config = lm_mixture_data_config(
    components={
        "stackexchange": dolmino_stackexchange_tokenized,
        "hqwebpages": stackexchange_filtered_hq_webpages_tokenized,
    },
    weights={"stackexchange": 0.5, "hqwebpages": 0.5},
)

half_stackexchange_half_hqwebpages_model = default_train(
    name="quality_filtering/half_stackexchange_half_hqwebpages",
    tokenized=data_config,
    model_config=llama_1_4b,
    train_config=llama_1_4b_train_config,
)

if __name__ == "__main__":
    executor_main(
        steps=[
            dolmino_stackexchange_model,
            # stackexchange_filtered_hq_webpages_model,
            # half_stackexchange_half_hqwebpages_model,
        ]
    )
