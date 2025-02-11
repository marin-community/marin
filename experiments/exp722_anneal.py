from experiments.anneal_config import AnnealConfig
from experiments.defaults import default_anneal
from experiments.dolmino.tokenize_dolmino import get_dolmino_step
from experiments.evals.evals import default_eval
from experiments.evals.task_configs import MMLU_TASKS
from experiments.exp72_baselines import fineweb_edu_tokenized
from experiments.midtraining_datasets import finemath_3_plus_tokenized
from marin.execution.executor import executor_main

dolmino_dclm = get_dolmino_step("dclm")

finemath_anneal_config = AnnealConfig(
    target_dataset=finemath_3_plus_tokenized,
    high_quality_web_text_dataset=dolmino_dclm,
)

control_dclm_anneal_config = AnnealConfig(
    high_quality_web_text_dataset=dolmino_dclm,
    target_dataset=None,
)
control_model = default_anneal(name="llama-8b-anneal-dclm", anneal_config=control_dclm_anneal_config)

annealed_model = default_anneal(name="llama-8b-anneal-finemath-dclm", anneal_config=finemath_anneal_config)


fineweb_edu_finemath_anneal_config = AnnealConfig(
    target_dataset=finemath_3_plus_tokenized,
    high_quality_web_text_dataset=fineweb_edu_tokenized,
)
anneal_on_finemath_with_fineweb_edu = default_anneal(
    name="llama-8b-anneal-finemath",
    anneal_config=fineweb_edu_finemath_anneal_config,
)

eval_anneal_on_finemath_with_fineweb_edu = default_eval(
    # step=anneal_on_finemath_with_fineweb_edu, # not working
    step="gs://marin-us-central2/checkpoints/llama-8b-anneal-finemath-fd2597/hf/step-210388",
    evals=MMLU_TASKS,
)

eval_control_model = default_eval(
    # step=anneal_on_finemath_with_fineweb_edu, # not working
    step="gs://marin-us-central2/checkpoints/llama-8b-control-00f31b/hf/step-210388",
    evals=MMLU_TASKS,
)

if __name__ == "__main__":
    executor_main(
        steps=[
            # annealed_model,
            # control_model,
            eval_anneal_on_finemath_with_fineweb_edu,
            # eval_control_model,
        ],
        description="Train 8B model on DCLM using WSD-S.",
    )
