"""An experiment to cooldown a 8B model on a 30/70 mixture of MegaMath and Dolmino DCLM.

Evaluates the quality of MegaMath: shows boost in MMLU mathematics.
"""

from experiments.anneal_config import AnnealConfig
from experiments.defaults import default_anneal, default_tokenize
from experiments.dolmino.tokenize_dolmino import get_dolmino_step_llama3
from experiments.evals.evals import default_eval
from experiments.evals.task_configs import MMLU_TASKS
from experiments.llama import llama3_tokenizer
from marin.download.huggingface.download_hf import DownloadConfig, download_hf
from marin.execution.executor import ExecutorStep, executor_main, this_output_path
from marin.processing.tokenize.data_configs import lm_mixture_data_config

# Define MegaMath dataset source
megamath_source = ExecutorStep(
    name="raw/megamath",
    fn=download_hf,
    config=DownloadConfig(
        hf_dataset_id="LLM360/MegaMath",
        # No specific revision/commit hash mentioned, so we'll use main.
        # Consider adding a specific commit hash for reproducibility if available.
        gcs_output_path=this_output_path(),  # Output path will be managed by the executor
        wait_for_completion=True,
    ),
)

# Tokenize the MegaMath dataset
megamath_tokenized = default_tokenize(
    name="megamath_tokenized",  # More specific name for the tokenized output
    dataset=megamath_source,
    tokenizer=llama3_tokenizer,
)

dolmino_dclm = get_dolmino_step_llama3("dclm")

megamath_anneal_config = AnnealConfig(
    dataset_config=lm_mixture_data_config(
        components={
            "megamath": megamath_tokenized,
            "dolmino": dolmino_dclm,
        },
        weights={"megamath": 0.3, "dolmino": 0.7},
    ),
)

control_dclm_anneal_config = AnnealConfig(
    dataset_config=lm_mixture_data_config(
        components={
            "dolmino": dolmino_dclm,
        },
        weights={"dolmino": 1.0},
    ),
)
control_model = default_anneal(
    name="llama-8b-anneal-dclm-control-for-megamath", anneal_config=control_dclm_anneal_config
)  # Changed name to be more specific

annealed_model = default_anneal(name="llama-8b-anneal-megamath-dclm", anneal_config=megamath_anneal_config)

# Note: Checkpoint paths will be determined dynamically by the annealing process.
# We define what to evaluate, not a specific checkpoint here.
eval_annealed_model = default_eval(
    step=annealed_model,  # Evaluate the output of the annealed_model step
    evals=MMLU_TASKS,
    name="eval_annealed_megamath_dclm",  # Added a specific name
)

eval_control_model = default_eval(
    step=control_model,  # Evaluate the output of the control_model step
    evals=MMLU_TASKS,
    name="eval_control_dclm_for_megamath",  # Added a specific name
)

if __name__ == "__main__":
    executor_main(
        steps=[
            megamath_tokenized,  # Ensure the dataset is processed
            annealed_model,
            control_model,
            eval_annealed_model,
            eval_control_model,
        ],
        description="Train 8B model on MegaMath and DCLM mixture.",
    )
