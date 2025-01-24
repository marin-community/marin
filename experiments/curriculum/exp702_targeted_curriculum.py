"""
Test continued training from checkpoint to support different mixtures.
Issue: https://github.com/stanford-crfm/marin/issues/702
"""

from itertools import chain
import random

from levanter.optim import AdamConfig

from experiments.defaults import _prepare_data_config
from experiments.llama import llama_150m, llama_300m, llama_600m

from marin.execution.executor import executor_main, output_path_of
from marin.processing.tokenize.data_configs import lm_mixture_data_config

from experiments.curriculum.curriculum_stages import train_executor_step, tokenize_train_validation
from experiments.instruction_datasets import get_instruction_dataset

BASE_DIR_STACK_PYTHON = "gs://marin-us-central2/raw/the-stack-dedup-4ba450/17cad72/data/python"
BASE_DIR_STACK_CPP = "gs://marin-us-central2/raw/the-stack-dedup-4ba450/17cad72/data/cpp"
BASE_DIR_DOLMA = "gs://marin-us-central2/raw/dolma/v1.7"
BASE_DIR_TULU3 = "gs://marin-us-central2/documents/allenai--tulu-3-sft-mixture-0a99cb/data"

tulu_3_dataset = get_instruction_dataset("allenai/tulu-3-sft-mixture")

# randomly split stack python parquet files into two seperate groups
stack_file_ids = list(range(144))
random.seed(42)
random.shuffle(stack_file_ids)
stack_file_ids_stage1 = stack_file_ids[0:72]
stack_file_ids_stage2 = stack_file_ids[72:143]
stack_file_ids_validation = stack_file_ids[143:144]

# randomly split dolma c4 json.gz files into two seperate groups
dolma_file_ids = list(range(171))
random.shuffle(dolma_file_ids)
dolma_file_ids_stage1 = dolma_file_ids[0:85]
dolma_file_ids_stage2 = dolma_file_ids[85:170]
dolma_file_ids_validation = dolma_file_ids[170:171]

# randomly split stack cpp parquet files into two seperate groups
stack_cpp_file_ids = list(range(110))
random.shuffle(stack_cpp_file_ids)
stack_cpp_file_ids_stage1 = stack_cpp_file_ids[0:55]
stack_cpp_file_ids_stage2 = stack_cpp_file_ids[55:109]
stack_cpp_file_ids_validation = stack_cpp_file_ids[109:110]

tulu_file_ids = list(range(6))
random.shuffle(tulu_file_ids)
tulu_file_ids_stage1 = tulu_file_ids[0:1]
tulu_file_ids_stage2 = tulu_file_ids[1:5]
tulu_file_ids_validation = tulu_file_ids[5:6]

stack_dedup_stage1_tokenized = tokenize_train_validation(
    train_files=[f"{BASE_DIR_STACK_PYTHON}/data-{id:05d}-of-00144.parquet" for id in stack_file_ids_stage1],
    validation_files=[f"{BASE_DIR_STACK_PYTHON}/data-{id:05d}-of-00144.parquet" for id in stack_file_ids_validation],
    name="stack_dedup_stage1",
    text_key="content"
)

dolma_c4_stage1_tokenized = tokenize_train_validation(
    train_files=[f"{BASE_DIR_DOLMA}/c4-{id:04d}.json.gz" for id in dolma_file_ids_stage1],
    validation_files=[f"{BASE_DIR_DOLMA}/c4-{id:04d}.json.gz" for id in dolma_file_ids_validation],
    name="dolma_c4_stage1",
    text_key="text"
)

stack_cpp_stage1_tokenized = tokenize_train_validation(
    train_files=[f"{BASE_DIR_STACK_CPP}/data-{id:05d}-of-00110.parquet" for id in stack_cpp_file_ids_stage1],
    validation_files=[f"{BASE_DIR_STACK_CPP}/data-{id:05d}-of-00110.parquet" for id in stack_cpp_file_ids_validation],
    name="stack_cpp_stage1",
    text_key="content"
)

# tulu_stage1_tokenized = tokenize_train_validation(
#     train_files=[output_path_of(tulu_3_dataset, f"train-{id:05d}-of-00006/**.jsonl.gz") for id in tulu_file_ids_stage1],
#     validation_files=[output_path_of(tulu_3_dataset, f"train-{id:05d}-of-00006/**.jsonl.gz") for id in tulu_file_ids_validation],
#     name="tulu_stage1",
#     input_field="user",
#     output_field="assistant",
# )

stack_dedup_stage2_tokenized = tokenize_train_validation(
    train_files=[f"{BASE_DIR_STACK_PYTHON}/data-{id:05d}-of-00144.parquet" for id in stack_file_ids_stage2],
    validation_files=[f"{BASE_DIR_STACK_PYTHON}/data-{id:05d}-of-00144.parquet" for id in stack_file_ids_validation],
    name="stack_dedup_stage2",
    text_key="content"
)

dolma_c4_stage2_tokenized = tokenize_train_validation(
    train_files=[f"{BASE_DIR_DOLMA}/c4-{id:04d}.json.gz" for id in dolma_file_ids_stage2],
    validation_files=[f"{BASE_DIR_DOLMA}/c4-{id:04d}.json.gz" for id in dolma_file_ids_validation],
    name="dolma_c4_stage2",
    text_key="text"
)

stack_cpp_stage2_tokenized = tokenize_train_validation(
    train_files=[f"{BASE_DIR_STACK_CPP}/data-{id:05d}-of-00110.parquet" for id in stack_cpp_file_ids_stage2],
    validation_files=[f"{BASE_DIR_STACK_CPP}/data-{id:05d}-of-00110.parquet" for id in stack_cpp_file_ids_validation],
    name="stack_cpp_stage2",
    text_key="content"
)

# tulu_stage2_tokenized = tokenize_train_validation(
#     train_files=[output_path_of(tulu_3_dataset, f"train-{id:05d}-of-00006/**.jsonl.gz") for id in tulu_file_ids_stage2],
#     validation_files=[output_path_of(tulu_3_dataset, f"train-{id:05d}-of-00006/**.jsonl.gz") for id in tulu_file_ids_validation],
#     name="tulu_stage2",
#     input_field="user",
#     output_field="assistant",
# )

stage_data = {
    "stack_dedup": {
        "stage1": stack_dedup_stage1_tokenized,
        "stage2": stack_dedup_stage2_tokenized,
    },
    "c4": {
        "stage1": dolma_c4_stage1_tokenized,
        "stage2": dolma_c4_stage2_tokenized,
    },
    "stack_cpp": {
        "stage1": stack_cpp_stage1_tokenized,
        "stage2": stack_cpp_stage2_tokenized,
    },
    # "tulu": {
    #     "stage1": tulu_stage1_tokenized,
    #     "stage2": tulu_stage2_tokenized,
    # },
}

def full_training_stage_varsched(data1_name, data2_name, total_data1_portion, duration_frac_stage2, data1_frac_alloc_stage2, learning_rate=3e-3, cooldown_frac=None, schedule_type="cosine", model_size="150m", num_train_steps=3000, version_tag="", additional_tags=[]):
    """
    Generalized version of varsched that works with any two datasets.
    
    Args:
        data1_name: Name of first dataset (e.g. "stack_dedup", "stack_cpp")
        data2_name: Name of second dataset (e.g. "c4")
        total_data1_portion: Total portion of data1 across both stages
        duration_frac_stage2: Fraction of total steps to spend in stage 2
        data1_frac_alloc_stage2: Fraction of data1's total portion to allocate to stage 2
    """
    duration_frac_stage1 = 1 - duration_frac_stage2
    data1_frac_alloc_stage1 = 1 - data1_frac_alloc_stage2

    data1_weight_stage1 = round(total_data1_portion * data1_frac_alloc_stage1 / duration_frac_stage1, 7)
    data1_weight_stage2 = round(total_data1_portion * data1_frac_alloc_stage2 / duration_frac_stage2, 7)

    print('-' * 100)
    print(f"total_data1_portion: {total_data1_portion}")
    print(f"duration_frac_stage1: {duration_frac_stage1}, data1_frac_alloc_stage1: {data1_frac_alloc_stage1}, data1_weight_stage1: {data1_weight_stage1}")
    print(f"duration_frac_stage2: {duration_frac_stage2}, data1_frac_alloc_stage2: {data1_frac_alloc_stage2}, data1_weight_stage2: {data1_weight_stage2}")

    assert 0 <= data1_weight_stage1 <= 1, f"data1_weight_stage1: {data1_weight_stage1}"
    assert 0 <= data1_weight_stage2 <= 1, f"data1_weight_stage2: {data1_weight_stage2}"

    data_config_stage1 = lm_mixture_data_config(
        components={data1_name: stage_data[data1_name]["stage1"], data2_name: stage_data[data2_name]["stage1"]},
        weights={data1_name: data1_weight_stage1, data2_name: 1 - data1_weight_stage1},
    )

    pretraining_data_stage1, evaluation_data_stage1 = _prepare_data_config(data_config_stage1, use_default_validation=True, use_default_evaluation=True)

    data_config_stage2 = lm_mixture_data_config(
        components={data1_name: stage_data[data1_name]["stage2"], data2_name: stage_data[data2_name]["stage2"]},
        weights={data1_name: data1_weight_stage2, data2_name: 1 - data1_weight_stage2},
    )

    pretraining_data_stage2, evaluation_data_stage2 = _prepare_data_config(data_config_stage2, use_default_validation=True, use_default_evaluation=True)

    # Construct executor steps for training

    tpu_type="v4-128"
    model = {
        "150m": llama_150m,
        "300m": llama_300m,
        "600m": llama_600m,
    }[model_size]
    train_batch_size=1024
    num_train_steps=num_train_steps 

    steps_stage1 = int(num_train_steps * duration_frac_stage1)
    steps_stage2 = num_train_steps - steps_stage1

    weight_decay=0.1
    steps_per_eval=num_train_steps // 20
    name_prefix = f"{data1_name}-{data2_name}-vs-{data1_frac_alloc_stage2}"
    if model_size == "300m" or model_size == "600m":
        name_prefix += f"-{model_size}"

    if schedule_type == "linear":
        optimizer_config = AdamConfig(
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            lr_schedule=schedule_type,
            decay=cooldown_frac,
        )
        name_prefix += f"-{schedule_type}-{cooldown_frac}"
    elif schedule_type == "cosine":
        optimizer_config = None
    else:
        raise ValueError(f"Invalid schedule type: {schedule_type}")

    train_step_stage1 = train_executor_step(
        name=f"{name_prefix}-{data1_weight_stage1}-{data1_weight_stage2}-stage1{version_tag}",
        pretraining_data=pretraining_data_stage1,
        evaluation_data=evaluation_data_stage1,
        model=model,
        model_checkpoint=None,
        train_batch_size=train_batch_size,
        num_train_steps=num_train_steps,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        steps_per_eval=steps_per_eval,
        steps_per_export=steps_stage1,
        tpu_type=tpu_type,
        optimizer_config=optimizer_config,
        additional_tags=additional_tags + ["stage1"],
    )

    train_step_stage2 = train_executor_step(
        name=f"{name_prefix}-{data1_weight_stage1}-{data1_weight_stage2}-stage2{version_tag}",
        pretraining_data=pretraining_data_stage2,
        evaluation_data=evaluation_data_stage2,
        model=model,
        model_checkpoint=output_path_of(train_step_stage1).cd(f"checkpoints/step-{steps_stage1}"),
        train_batch_size=train_batch_size,
        num_train_steps=num_train_steps,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        steps_per_eval=steps_per_eval,
        steps_per_export=num_train_steps,
        tpu_type=tpu_type,
        optimizer_config=optimizer_config,
        additional_tags=additional_tags + ["stage2"],
    )

    return [train_step_stage1, train_step_stage2]

def full_training_stage_baseline_sweep(data1_name, data2_name, learning_rate, schedule_type, cooldown_frac=None, num_train_steps=3000, model_size="150m", additional_tags=[], data1_portion=0.005, train_batch_size=1024):
    data_config = lm_mixture_data_config(
        components={data1_name: stage_data[data1_name]["stage1"], data2_name: stage_data[data2_name]["stage2"]},
        weights={data1_name: data1_portion, data2_name: 1 - data1_portion},
    )

    pretraining_data, evaluation_data = _prepare_data_config(data_config, use_default_validation=True, use_default_evaluation=True)

    # Construct executor steps for training

    tpu_type="v4-128"
    model = {
        "150m": llama_150m,
        "300m": llama_300m,
    }[model_size]
    weight_decay=0.1
    steps_per_eval=num_train_steps // 20
    steps_per_export=num_train_steps // 2
    name_prefix = f"{data1_name}-{data2_name}-{num_train_steps // 1000}B-{model_size}-baseline"

    if schedule_type == "linear":
        optimizer_config = AdamConfig(
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            lr_schedule=schedule_type,
            decay=cooldown_frac,
        )

        schedule_type = f"linear-{cooldown_frac}"
    elif schedule_type == "cosine":
        optimizer_config = None
    else:
        raise ValueError(f"Invalid schedule type: {schedule_type}")
    
    train_step = train_executor_step(
        name=name_prefix,
        pretraining_data=pretraining_data,
        evaluation_data=evaluation_data,
        model=model,
        model_checkpoint=None,
        train_batch_size=train_batch_size,
        num_train_steps=num_train_steps,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        steps_per_eval=steps_per_eval,
        steps_per_export=steps_per_export,
        tpu_type=tpu_type,
        optimizer_config=optimizer_config,
        additional_tags=additional_tags,
    )

    return [train_step]

############################################################

if __name__ == "__main__":
    # stage_pairs = [
    #     full_training_stage_varsched(
    #         data1_name="stack_dedup",
    #         data2_name="stack_cpp",
    #         total_data1_portion=0.005,
    #         duration_frac_stage2=duration_frac_stage2,
    #         data1_frac_alloc_stage2=data1_frac_alloc_stage2,
    #         schedule_type=schedule_type,
    #         cooldown_frac=cooldown_frac,
    #         additional_tags=["python-cpp-0.005-allstage2-sweep"],
    #         version_tag="-v1"
    #     )
    #     for duration_frac_stage2 in [0.4, 0.2, 0.1, 0.05, 0.025]
    #     for schedule_type, cooldown_frac in [("cosine", None), ("linear", 0.0), ("linear", 0.05), ("linear", 0.2)]
    #     for data1_frac_alloc_stage2 in [1.0]
    # ]

    stage_pairs = [
        full_training_stage_varsched(
            data1_name="stack_dedup",
            data2_name="c4",
            total_data1_portion=0.005,
            duration_frac_stage2=duration_frac_stage2,
            data1_frac_alloc_stage2=data1_frac_alloc_stage2,
            schedule_type=schedule_type,
            cooldown_frac=cooldown_frac,
            model_size="600m",
            num_train_steps=12000,
            additional_tags=["python-c4-0.005-600m-allstage2-sweep"],
        )
        for duration_frac_stage2 in [0.4, 0.2, 0.1, 0.05, 0.025]
        for schedule_type, cooldown_frac in [("cosine", None), ("linear", 0.0), ("linear", 0.05), ("linear", 0.2)]
        for data1_frac_alloc_stage2 in [1.0]
    ]

    steps = list(chain(*stage_pairs))

    executor_main(
        steps=steps,
        description=f"Test training with varying mixtures",
    )
