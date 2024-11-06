from evals.evals import evaluate_helm

from experiments.defaults import default_tokenize, default_train
from experiments.llama import llama3_tokenizer, llama_1_4b, llama_1_4b_train_config
from experiments.pretraining_datasets import fineweb_edu
from marin.execution.executor import executor_main, output_path_of

fineweb_edu_tokenized = default_tokenize(name="fineweb-edu", dataset=fineweb_edu, tokenizer=llama3_tokenizer)
fineweb_edu_model = default_train(
    name="exp446-fineweb-edu-1.4b",
    tokenized=fineweb_edu_tokenized,
    model_config=llama_1_4b,
    train_config=llama_1_4b_train_config,
)

eval_step = evaluate_helm(
    model_name="exp446-fineweb-edu-1.4b",
    model_path=output_path_of(fineweb_edu_model),
    evals=["mmlu"],
)

############################################################

if __name__ == "__main__":
    executor_main(
        steps=[
            fineweb_edu_tokenized,
            fineweb_edu_model,
            eval_step,
        ],
        description="Train 1.4B model on standard datasets for tracking dev metrics.",
    )
