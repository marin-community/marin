from experiments.defaults import default_tokenize, default_train
from experiments.llama import llama3_tokenizer, llama_1_4b, llama_1_4b_train_config
from experiments.pretraining_datasets import fineweb_edu
from marin.execution.executor import ExecutorStep, executor_main
from marin.utilities.metrics_utils import upload_metrics_to_gcs

# Train a default 1.4B model on 42B tokens with internal evals, so that we can track relevant metrics.
fineweb_edu_tokenized = default_tokenize(name="fineweb-edu", dataset=fineweb_edu, tokenizer=llama3_tokenizer)
fineweb_edu_model = default_train(
    name="exp446-fineweb-edu-1.4b",
    tokenized=fineweb_edu_tokenized,
    model_config=llama_1_4b,
    train_config=llama_1_4b_train_config,
)

dev_metrics = ExecutorStep(
    name="dev-metrics",
    fn=upload_metrics_to_gcs,
    config={
        # TODO (WIP)
        # Here we'd somehow specify the config for metrics we want to
        # track/upload- wandb metrics, GitHub API metadata, GCS info, etc.
        # this can include evals for the model from the previous step
    },
)

############################################################

if __name__ == "__main__":
    executor_main(
        steps=[
            fineweb_edu_tokenized,
            fineweb_edu_model,
            dev_metrics,
        ],
        description="Simple experiment for tracking dev metrics.",
    )
