"""An experiment to cooldown a 8B model on a 30/70 mixture of megamath-web-pro and Dolmino DCLM.

Evaluates the quality of megamath-web-pro: shows boost in GSM8K and MMLU mathematics.
"""

from experiments.pretraining_datasets import megamath

from experiments.anneal_config import AnnealConfig
from experiments.defaults import default_anneal, default_tokenize
from experiments.dolmino.tokenize_dolmino import get_dolmino_step
from experiments.evals.evals import default_eval
from experiments.evals.task_configs import MMLU_TASKS
from marin.execution.executor import executor_main
from marin.processing.tokenize.data_configs import lm_mixture_data_config
from experiments.llama import llama3_tokenizer

# Get the Dolmino DCLM dataset
dolmino_dclm = get_dolmino_step("dclm")

# Focus cooldown on megamath-web-pro subset only
dataset_megamath_web_pro = megamath.cd("megamath-web-pro")

tokenized_megamath_web_pro = default_tokenize(
    name="megamath-web-pro",
    dataset=dataset_megamath_web_pro,
    tokenizer=llama3_tokenizer,
)

# Configure the megamath annealing experiment with 30/70 mixture
megamath_anneal_config = AnnealConfig(
    dataset_config=lm_mixture_data_config(
        components={
            "megamath-web-pro": tokenized_megamath_web_pro,
            "dolmino": dolmino_dclm,
        },
        weights={"megamath-web-pro": 0.3, "dolmino": 0.7},
    ),
)

# Configure the control experiment with 100% Dolmino DCLM
control_dclm_anneal_config = AnnealConfig(
    dataset_config=lm_mixture_data_config(
        components={
            "dolmino": dolmino_dclm,
        },
        weights={"dolmino": 1.0},
    ),
)

# Create the annealing experiments
control_model = default_anneal(name="llama-8b-anneal-dclm", anneal_config=control_dclm_anneal_config)
annealed_model = default_anneal(name="llama-8b-anneal-megamath-dclm", anneal_config=megamath_anneal_config)

# Set up evaluation tasks
eval_anneal_on_megamath = default_eval(
    step="gs://marin-us-central2/checkpoints/llama-8b-anneal-megamath-dclm/hf/step-210388",
    evals=MMLU_TASKS,
)

eval_control_model = default_eval(
    step="gs://marin-us-central2/checkpoints/llama-8b-anneal-dclm/hf/step-210388",
    evals=MMLU_TASKS,
)

if __name__ == "__main__":
    executor_main(
        steps=[
            megamath,
            tokenized_megamath_web_pro,
            annealed_model, 
            control_model,
            # eval_anneal_on_megamath,
            # eval_control_model,
        ],
        description="Evaluate megamath-web-pro quality via cooldown on 8B model with 30/70 mixture of megamath and Dolmino DCLM",
    )