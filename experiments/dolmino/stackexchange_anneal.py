from experiments.anneal_config import AnnealConfig
from experiments.defaults import default_anneal
from experiments.dolmino.tokenize_dolmino import get_dolmino_step
from experiments.evals.evals import default_eval
from experiments.evals.task_configs import MMLU_TASKS
from marin.execution.executor import executor_main
from marin.processing.tokenize.data_configs import lm_mixture_data_config

dolmino_dclm = get_dolmino_step("dclm")
stackexchange_tokenized = get_dolmino_step("stackexchange")


dataset_config = lm_mixture_data_config(
    components={
        "stackexchange": stackexchange_tokenized,
        "dclm": dolmino_dclm,
    },
    weights={"stackexchange": 0.30, "dclm": 0.70},
)
# Dolmino Stack Exchange dataset has 1.26B tokens.
# Our mixed dataset is 30% dolmino and 70% high-quality web data.
# This means we will epoch dolmino dataset 2 times.
stackexchange_anneal_config = AnnealConfig(
    dataset_config=dataset_config,
    num_anneal_training_tokens=8_400_000_000,
)

stackexchange_anneal_model = default_anneal(
    name="llama-8b-anneal-stackexchange-0",
    anneal_config=stackexchange_anneal_config,
)

eval_stackexchange_anneal_model = default_eval(
    step="gs://marin-us-central2/checkpoints/llama-8b-anneal-stackexchange-0-db2e46/hf/step-200470",
    evals=MMLU_TASKS,
)

if __name__ == "__main__":
    executor_main(
        steps=[
            # stackexchange_anneal_model,
            eval_stackexchange_anneal_model,
        ],
    )
