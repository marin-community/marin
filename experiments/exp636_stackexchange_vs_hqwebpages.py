from experiments.defaults import default_tokenize, default_train
from experiments.exp102_classifier_ablations import create_steps
from experiments.exp596_stackexchange_classifier import stackexchange_experiment_config
from experiments.llama import llama3_tokenizer, llama_1_4b, llama_1_4b_train_config
from experiments.pretraining_datasets import dolmino
from marin.execution.executor import executor_main
from marin.processing.tokenize.data_configs import lm_mixture_data_config

dolmino_stackexchange = dolmino.cd("stackexchange")

dolmino_stackexchange_tokenized = default_tokenize(
    name="quality_filtering/dolmino_stackexchange_model",
    dataset=dolmino_stackexchange,
    tokenizer=llama3_tokenizer,
)

dolmino_stackexchange_model = default_train(
    name="quality_filtering/dolmino_stackexchange_model",
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
    name="quality_filtering/half_stackexchange_half_hqwebpages_model",
    tokenized=data_config,
    model_config=llama_1_4b,
    train_config=llama_1_4b_train_config,
)

if __name__ == "__main__":
    executor_main(
        steps=[
            dolmino_stackexchange_model,
            stackexchange_filtered_hq_webpages_model,
            half_stackexchange_half_hqwebpages_model,
        ]
    )
