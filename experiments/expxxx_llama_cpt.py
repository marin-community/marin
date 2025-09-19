from levanter.trainer import BatchSchedule
from experiments.defaults import default_anneal
from experiments.anneal_config import AnnealConfig
from experiments.evals.task_configs import MMLU_5_SHOT
from experiments.llama import llama_3_2_1b
from marin.evaluation.evaluation_config import EvalTaskConfig
from marin.processing.tokenize.data_configs import lm_mixture_data_config, lm_varying_mixture_data_config
from experiments.midtrain_sweep import (
    synthetic_data_generated_tokenized,
    nemo_qa_tokenized,
    decay_data_tokenized,
    dclm_mcq_tokenized,
)
from experiments.dclm.tokenize_dclm import dclm_components_llama3
from experiments.evals.evals import evaluate_levanter_lm_evaluation_harness
from experiments.evals.resource_configs import SINGLE_TPU_V4_8
from marin.resources import TpuPodConfig
from marin.execution.executor import executor_main, output_path_of
from experiments.midtraining_datasets import finemath_3_plus_tokenized

llama_1b_cpt_wandb_tags = ["llama-1b", "qa-30", "dclm-70", "8.4b", "lr=5e-5", "wd=1e-7", "warmup=0.05", "bsz=256"]
llama_1b_cpt_wrapqa_config = AnnealConfig(
    dataset_config=lm_mixture_data_config(
        components={"wrapqa": synthetic_data_generated_tokenized, "dclm": dclm_components_llama3["dclm_baseline"]},
        weights={"wrapqa": 0.30, "dclm": 0.70},
    ),
    learning_rate=5e-5,
    weight_decay=1e-7,
    warmup=0.05,
    train_batch_size=256,
    num_anneal_training_tokens=8_400_000_000,
    resources=TpuPodConfig(tpu_type="v4-64", slice_count=1),
    wandb_tags=llama_1b_cpt_wandb_tags,
    initialize_from_hf="meta-llama/Llama-3.2-1B",
    model_config=llama_3_2_1b,
)

llama_1b_cpt_wrapqa = default_anneal(
    name="llama-1b-wrapqa-cpt",
    anneal_config=llama_1b_cpt_wrapqa_config,
)


llama_1b_cpt_control_tags = ["llama-1b", "dclm-100", "8.4b", "lr=5e-5", "wd=1e-7", "warmup=0.05", "bsz=256"]
llama_1b_cpt_control = default_anneal(
    name="llama-1b-control-cpt",
    anneal_config=AnnealConfig(
        dataset_config=lm_mixture_data_config(
            components={"dclm": dclm_components_llama3["dclm_baseline"]},
            weights={"dclm": 1.0},
        ),
        learning_rate=5e-5,
        weight_decay=1e-7,
        warmup=0.05,
        train_batch_size=256,
        num_anneal_training_tokens=8_400_000_000,
        resources=TpuPodConfig(tpu_type="v4-64", slice_count=1),
        wandb_tags=llama_1b_cpt_control_tags,
        initialize_from_hf="meta-llama/Llama-3.2-1B",
        model_config=llama_3_2_1b,
    ),
)

llama_1b_cpt_nemo_qa_tags = ["llama-1b", "nemo-qa-30", "dclm-70", "8.4b", "lr=5e-5", "wd=1e-7", "warmup=0.05", "bsz=256"]
llama_1b_cpt_nemo_qa = default_anneal(
    name="llama-1b-nemo-qa-cpt",
    anneal_config=AnnealConfig(
        dataset_config=lm_mixture_data_config(
            components={"nemo_qa": nemo_qa_tokenized, "dclm": dclm_components_llama3["dclm_baseline"]},
            weights={"nemo_qa": 0.30, "dclm": 0.70},
        ),
        learning_rate=5e-5,
        weight_decay=1e-7,
        warmup=0.05,
        train_batch_size=256,
        num_anneal_training_tokens=8_400_000_000,
        resources=TpuPodConfig(tpu_type="v4-64", slice_count=1),
        wandb_tags=llama_1b_cpt_nemo_qa_tags,
        initialize_from_hf="meta-llama/Llama-3.2-1B",
        model_config=llama_3_2_1b,
    ),
)

llama_1b_cpt_flan_tags = ["llama-1b", "flan-30", "dclm-70", "8.4b", "lr=5e-5", "wd=1e-7", "warmup=0.05", "bsz=256"]
llama_1b_cpt_flan = default_anneal(
    name="llama-1b-flan-cpt",
    anneal_config=AnnealConfig(
        dataset_config=lm_mixture_data_config(
            components={"flan": decay_data_tokenized, "dclm": dclm_components_llama3["dclm_baseline"]},
            weights={"flan": 0.30, "dclm": 0.70},
        ),
        learning_rate=5e-5,
        weight_decay=1e-7,
        warmup=0.05,
        train_batch_size=256,
        num_anneal_training_tokens=8_400_000_000,
        resources=TpuPodConfig(tpu_type="v4-64", slice_count=1),
        wandb_tags=llama_1b_cpt_flan_tags,
        initialize_from_hf="meta-llama/Llama-3.2-1B",
        model_config=llama_3_2_1b,
    ),
)

eval_tasks = (
    MMLU_5_SHOT,
    # MMLU_PRO_5_SHOT,
    # EvalTaskConfig("commonsense_qa_sl", num_fewshot=10),
    EvalTaskConfig("mmlu_sl", num_fewshot=0, task_alias="mmlu_sl_0_shot"),
    EvalTaskConfig("mmlu_sl", num_fewshot=5, task_alias="mmlu_sl_5_shot"),
    EvalTaskConfig("mmlu_sl_verb", num_fewshot=0, task_alias="mmlu_sl_verb_0_shot"),
    EvalTaskConfig("mmlu_sl_verb", num_fewshot=5, task_alias="mmlu_sl_verb_5_shot"),
    EvalTaskConfig("gsm8k_loss", num_fewshot=8, task_alias="gsm8k_loss_8shot"),
    EvalTaskConfig("math_500_loss", num_fewshot=0),
)

eval_steps = []
all_tags = [llama_1b_cpt_wandb_tags, llama_1b_cpt_control_tags, llama_1b_cpt_nemo_qa_tags, llama_1b_cpt_flan_tags]
all_steps = [llama_1b_cpt_wrapqa, llama_1b_cpt_control, llama_1b_cpt_nemo_qa, llama_1b_cpt_flan]
for step, tags in zip(all_steps, all_tags, strict=False):
    experiment_name = step.name.split("/")[-1]
    eval_step = evaluate_levanter_lm_evaluation_harness(
        model_name=experiment_name,
        model_path=output_path_of(step),
        evals=eval_tasks,
        resource_config=SINGLE_TPU_V4_8,
        wandb_tags=tags,
    )

    eval_steps.append(eval_step)

microanneal_suite = [
    64_000_000,
    128_000_000,
    256_000_000,
    512_000_000,
    1_024_000_000,
    2_048_000_000,
    4_096_000_000,
    8_192_000_000,
]

microanneal_steps = []
for num_anneal_tokens in microanneal_suite:
    num_anneal_tokens_str = f"{num_anneal_tokens//1_000_000}M"
    llama_1b_cpt_mcq_tags = [
        "llama-1b",
        "mcq-30",
        "dclm-70",
        f"{num_anneal_tokens_str}",
        "lr=5e-5",
        "wd=1e-7",
        "warmup=0.05",
        "bsz=256",
    ]
    llama_1b_cpt_mcq = default_anneal(
        name=f"llama-1b-mcq-cpt-{num_anneal_tokens_str}",
        anneal_config=AnnealConfig(
            dataset_config=lm_mixture_data_config(
                components={"mcq": dclm_mcq_tokenized, "dclm": dclm_components_llama3["dclm_baseline"]},
                weights={"mcq": 0.30, "dclm": 0.70},
            ),
            learning_rate=5e-5,
            weight_decay=1e-7,
            warmup=0.05,
            train_batch_size=256,
            num_anneal_training_tokens=num_anneal_tokens,
            resources=TpuPodConfig(tpu_type="v4-64", slice_count=1),
            wandb_tags=llama_1b_cpt_mcq_tags,
            initialize_from_hf="meta-llama/Llama-3.2-1B",
            model_config=llama_3_2_1b,
        ),
    )
    experiment_name = llama_1b_cpt_mcq.name.split("/")[-1]
    eval_step = evaluate_levanter_lm_evaluation_harness(
        model_name=experiment_name,
        model_path=output_path_of(llama_1b_cpt_mcq),
        evals=eval_tasks,
        resource_config=SINGLE_TPU_V4_8,
        wandb_tags=llama_1b_cpt_mcq_tags,
    )
    microanneal_steps.append(eval_step)

# Pretraining, Midtraining ratio
fixed_microanneal_suite_ratios = [
    (0.40, 0.60),
    (0.50, 0.50),
    (0.60, 0.40),
    (0.70, 0.30),
    (0.80, 0.20),
    (0.90, 0.10),
    # (1.00, 0.00),
]
fixed_microanneal_steps = []
for pretraining_ratio, midtraining_ratio in fixed_microanneal_suite_ratios:
    llama_1b_cpt_mcq_tags = [
        "llama-1b",
        f"dclm-{int(pretraining_ratio*100)}",
        f"mcq-{int(midtraining_ratio*100)}",
        "lr=5e-5",
        "wd=1e-7",
        "warmup=0.05",
        "bsz=256",
        "v2",
    ]
    llama_1b_cpt_mcq = default_anneal(
        name=f"llama-1b-mcq-cpt-v2-{pretraining_ratio}-{midtraining_ratio}",
        anneal_config=AnnealConfig(
            dataset_config=lm_mixture_data_config(
                components={"mcq": dclm_mcq_tokenized, "dclm": dclm_components_llama3["dclm_baseline"]},
                weights={"mcq": midtraining_ratio, "dclm": pretraining_ratio},
            ),
            learning_rate=5e-5,
            weight_decay=1e-7,
            warmup=0.05,
            train_batch_size=256,
            num_anneal_training_tokens=64_000_000,
            resources=TpuPodConfig(tpu_type="v4-64", slice_count=1),
            wandb_tags=llama_1b_cpt_mcq_tags,
            initialize_from_hf="meta-llama/Llama-3.2-1B",
            model_config=llama_3_2_1b,
        ),
    )
    experiment_name = llama_1b_cpt_mcq.name.split("/")[-1]
    eval_step = evaluate_levanter_lm_evaluation_harness(
        model_name=experiment_name,
        model_path=output_path_of(llama_1b_cpt_mcq),
        evals=eval_tasks,
        resource_config=SINGLE_TPU_V4_8,
        wandb_tags=llama_1b_cpt_mcq_tags,
    )
    fixed_microanneal_steps.append(eval_step)


# Ablation: 1B tokens @ 70/30 dclm/finemath, then 64M tokens @ 70/30 dclm/mcq
# We implement this as a varying mixture over sequences. Indices are in sequences, not batches.
# With seq_len=4096 and train_batch_size=256, sequences per step = 256.
# 1B tokens => 1_000_000_000 / 4096 ≈ 244140 sequences.
# Transition index in sequences is 244_140.
_SEQ_LEN = 4096
_MIXTURE_BLOCK_SIZE = 2048
_BATCH_SIZE = 256
_TOKENS_FIRST_STAGE = 1_000_000_000
_TOKENS_SECOND_STAGE = 64_000_000
_FIRST_STAGE_SEQUENCES = _TOKENS_FIRST_STAGE // (_SEQ_LEN * _BATCH_SIZE)
_TOTAL_TOKENS = _TOKENS_FIRST_STAGE + _TOKENS_SECOND_STAGE

# Use actual batch schedule to convert steps -> example offset, then align to mixture block size.
batch_schedule = BatchSchedule(_BATCH_SIZE)
target_offset = batch_schedule.global_data_offset_by_step(_FIRST_STAGE_SEQUENCES)
aligned_offset = (target_offset // _MIXTURE_BLOCK_SIZE) * _MIXTURE_BLOCK_SIZE
_FIRST_STAGE_SEQUENCES = batch_schedule.find_step_containing_offset(aligned_offset)

llama_1b_cpt_finemath_then_mcq_tags = [
    "llama-1b",
    "dclm-70",
    "finemath-30->mcq-30",
    "1.064b",
    "lr=5e-5",
    "wd=1e-7",
    "warmup=0.05",
    "bsz=256",
]

llama_1b_cpt_finemath_then_mcq = default_anneal(
    name="llama-1b-finemath-then-mcq-cpt",
    anneal_config=AnnealConfig(
        dataset_config=lm_varying_mixture_data_config(
            components={
                "dclm": dclm_components_llama3["dclm_baseline"],
                "finemath_3_plus": finemath_3_plus_tokenized,
                "mcq": dclm_mcq_tokenized,
            },
            weights_list=[
                (0, {"dclm": 0.70, "finemath_3_plus": 0.30, "mcq": 0.0}),
                (int(_FIRST_STAGE_SEQUENCES), {"dclm": 0.70, "finemath_3_plus": 0.0, "mcq": 0.30}),
            ],
        ),
        learning_rate=5e-5,
        weight_decay=1e-7,
        warmup=0.05,
        train_batch_size=_BATCH_SIZE,
        num_anneal_training_tokens=_TOTAL_TOKENS,
        resources=TpuPodConfig(tpu_type="v4-64", slice_count=1),
        wandb_tags=llama_1b_cpt_finemath_then_mcq_tags,
        initialize_from_hf="meta-llama/Llama-3.2-1B",
        model_config=llama_3_2_1b,
    ),
)

experiment_name = llama_1b_cpt_finemath_then_mcq.name.split("/")[-1]
llama_1b_cpt_finemath_then_mcq_eval = evaluate_levanter_lm_evaluation_harness(
    model_name=experiment_name,
    model_path=output_path_of(llama_1b_cpt_finemath_then_mcq),
    evals=eval_tasks,
    resource_config=SINGLE_TPU_V4_8,
    wandb_tags=llama_1b_cpt_finemath_then_mcq_tags,
)


# Control: constant 70/30 dclm/finemath for the full 1.064B tokens (no MCQ stage)
# Sweep over pretraining stage dclm/finemath ratios: (0.50, 0.50), (0.60, 0.40), (0.70, 0.30), (0.80, 0.20), (1.00, 0.00)
finemath_then_mcq_sweep_steps = []
pretraining_ratio_options = [0.70, 0.80]

for pretraining_ratio in pretraining_ratio_options:
    finemath_ratio = round(1.0 - pretraining_ratio, 2)
    llama_1b_cpt_finemath_control_tags = [
        "llama-1b",
        f"dclm-{int(pretraining_ratio*100)}",
        f"finemath-{int(finemath_ratio*100)}",
        "1.064b",
        "lr=5e-5",
        "wd=1e-7",
        "warmup=0.05",
        "bsz=256",
    ]

    llama_1b_cpt_finemath_control = default_anneal(
        name=f"llama-1b-finemath-control-cpt-{pretraining_ratio}-{finemath_ratio}",
        anneal_config=AnnealConfig(
            dataset_config=lm_mixture_data_config(
                components={
                    "dclm": dclm_components_llama3["dclm_baseline"],
                    "finemath_3_plus": finemath_3_plus_tokenized,
                },
                weights={"dclm": pretraining_ratio, "finemath_3_plus": finemath_ratio},
            ),
            learning_rate=5e-5,
            weight_decay=1e-7,
            warmup=0.05,
            train_batch_size=_BATCH_SIZE,
            num_anneal_training_tokens=_TOTAL_TOKENS,
            resources=TpuPodConfig(tpu_type="v4-64", slice_count=1),
            wandb_tags=llama_1b_cpt_finemath_control_tags,
            initialize_from_hf="meta-llama/Llama-3.2-1B",
            model_config=llama_3_2_1b,
        ),
    )

    control_experiment_name = llama_1b_cpt_finemath_control.name.split("/")[-1]
    llama_1b_cpt_finemath_control_eval = evaluate_levanter_lm_evaluation_harness(
        model_name=control_experiment_name,
        model_path=llama_1b_cpt_finemath_control,
        evals=eval_tasks,
        resource_config=SINGLE_TPU_V4_8,
        wandb_tags=llama_1b_cpt_finemath_control_tags,
    )
    finemath_then_mcq_sweep_steps.append(llama_1b_cpt_finemath_control_eval)


for pretraining_ratio in [0.50, 0.60, 0.70, 0.80, 1.00]:
    finemath_ratio = round(1.0 - pretraining_ratio, 2)
    tags = [
        "llama-1b",
        f"dclm-{int(pretraining_ratio*100)}",
        f"finemath-{int(finemath_ratio*100)}->mcq-30",
        "1.064b",
        "lr=5e-5",
        "wd=1e-7",
        "warmup=0.05",
        "bsz=256",
    ]

    step = default_anneal(
        name=f"llama-1b-finemath-then-mcq-cpt-{pretraining_ratio}-{finemath_ratio}",
        anneal_config=AnnealConfig(
            dataset_config=lm_varying_mixture_data_config(
                components={
                    "dclm": dclm_components_llama3["dclm_baseline"],
                    "finemath_3_plus": finemath_3_plus_tokenized,
                    "mcq": dclm_mcq_tokenized,
                },
                weights_list=[
                    (0, {"dclm": pretraining_ratio, "finemath_3_plus": finemath_ratio, "mcq": 0.0}),
                    (int(_FIRST_STAGE_SEQUENCES), {"dclm": 0.70, "finemath_3_plus": 0.0, "mcq": 0.30}),
                ],
            ),
            learning_rate=5e-5,
            weight_decay=1e-7,
            warmup=0.05,
            train_batch_size=_BATCH_SIZE,
            num_anneal_training_tokens=_TOTAL_TOKENS,
            resources=TpuPodConfig(tpu_type="v4-64", slice_count=1),
            wandb_tags=tags,
            initialize_from_hf="meta-llama/Llama-3.2-1B",
            model_config=llama_3_2_1b,
        ),
    )

    expt_name = step.name.split("/")[-1]
    eval_step = evaluate_levanter_lm_evaluation_harness(
        model_name=expt_name,
        model_path=step,
        evals=eval_tasks,
        # evals=[EvalTaskConfig("gsm8k_loss", num_fewshot=8, task_alias="gsm8k_loss_8shot")],
        # evals=[EvalTaskConfig("math_500_loss", num_fewshot=0)],
        resource_config=SINGLE_TPU_V4_8,
        wandb_tags=tags,
    )

    finemath_then_mcq_sweep_steps.append(eval_step)

if __name__ == "__main__":
    executor_main(
        # steps=eval_steps,
        # steps=microanneal_steps,
        # steps=fixed_microanneal_steps,
        steps=finemath_then_mcq_sweep_steps,
    )
