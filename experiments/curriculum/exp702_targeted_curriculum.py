"""
Test continued training from checkpoint to support different mixtures.
Issue: https://github.com/stanford-crfm/marin/issues/702
"""

from itertools import chain

from levanter.optim import AdamConfig, SGDConfig

from experiments.defaults import _prepare_data_config
from experiments.llama import llama_150m, llama_300m

from marin.execution.executor import executor_main, output_path_of
from marin.processing.tokenize.data_configs import lm_mixture_data_config

from experiments.curriculum.curriculum_stages import train_executor_step
from experiments.curriculum.datasets import stage_data

def full_training_stage_varsched(data1_name, data2_name, total_data1_portion, duration_frac_stage2, data1_frac_alloc_stage2, learning_rate=3e-3, cooldown_frac=None, schedule_type="cosine", version_tag="", additional_tags=[]):
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
    model = llama_150m
    train_batch_size=1024
    num_train_steps=3000 # 1024 * 1024 * 3000 = 3B tokens

    steps_stage1 = int(num_train_steps * duration_frac_stage1)
    steps_stage2 = num_train_steps - steps_stage1

    weight_decay=0.1
    steps_per_eval=num_train_steps // 20
    name_prefix = f"varsched-bf-{data1_frac_alloc_stage2}"

    if schedule_type == "linear":
        optimizer_config = AdamConfig(
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            lr_schedule=schedule_type,
            decay=cooldown_frac,
        )
        name_prefix += f"-{schedule_type}-{cooldown_frac}"
    elif schedule_type == "linear-sgd":
        optimizer_config = SGDConfig(
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            momentum=0.0,
        )
        name_prefix += f"-linear-sgd"
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

def full_training_stage_halfsched(data1_name, data2_name, starting_data1_portion, ending_data1_portion, num_train_steps=3000, model_size="150m", version_tag="", additional_tags=[]):
    """
    Generalized version of halfsched that works with any two datasets.
    
    Args:
        data1_name: Name of first dataset (e.g. "stack_dedup", "stack_cpp")
        data2_name: Name of second dataset (e.g. "c4")
        starting_data1_portion: Portion of data1 in first stage
        ending_data1_portion: Portion of data1 in second stage
    """
    data_config_stage1 = lm_mixture_data_config(
        components={data1_name: stage_data[data1_name]["stage1"], data2_name: stage_data[data2_name]["stage1"]},
        weights={data1_name: starting_data1_portion, data2_name: 1 - starting_data1_portion},
    )

    pretraining_data_stage1, evaluation_data_stage1 = _prepare_data_config(data_config_stage1, use_default_validation=True, use_default_evaluation=True)

    data_config_stage2 = lm_mixture_data_config(
        components={data1_name: stage_data[data1_name]["stage2"], data2_name: stage_data[data2_name]["stage2"]},
        weights={data1_name: ending_data1_portion, data2_name: 1 - ending_data1_portion},
    )

    pretraining_data_stage2, evaluation_data_stage2 = _prepare_data_config(data_config_stage2, use_default_validation=True, use_default_evaluation=True)

    # Construct executor steps for training

    tpu_type="v4-128"
    model = {
        "150m": llama_150m,
        "300m": llama_300m,
    }[model_size]
    train_batch_size=1024
    learning_rate=3e-3
    weight_decay=0.1
    steps_per_eval=num_train_steps // 20
    steps_per_export=3000 // 2
    name_prefix = f"stack-dedup-c4-curriculum-{num_train_steps // 1000}B-{model_size}-halfsched"

    train_step_stage1 = train_executor_step(
        name=f"{name_prefix}-{starting_data1_portion}-stage1{version_tag}",
        pretraining_data=pretraining_data_stage1,
        evaluation_data=evaluation_data_stage1,
        model=model,
        model_checkpoint=None,
        train_batch_size=train_batch_size,
        num_train_steps=num_train_steps,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        steps_per_eval=steps_per_eval,
        steps_per_export=steps_per_export,
        tpu_type=tpu_type,
        additional_tags=additional_tags + ["stage1"],
    )

    train_step_stage2 = train_executor_step(
        name=f"{name_prefix}-{starting_data1_portion}-{ending_data1_portion}-stage2{version_tag}",
        pretraining_data=pretraining_data_stage2,
        evaluation_data=evaluation_data_stage2,
        model=model,
        model_checkpoint=output_path_of(train_step_stage1).cd(f"checkpoints/step-{num_train_steps // 2}"),
        train_batch_size=train_batch_size,
        num_train_steps=num_train_steps,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        steps_per_eval=steps_per_eval,
        steps_per_export=steps_per_export,
        tpu_type=tpu_type,
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
    stage_pairs = [
        full_training_stage_varsched(
            data1_name="stack_dedup",
            data2_name="stack_cpp",
            total_data1_portion=0.005,
            duration_frac_stage2=duration_frac_stage2,
            data1_frac_alloc_stage2=data1_frac_alloc_stage2,
            schedule_type=schedule_type,
            cooldown_frac=cooldown_frac,
            additional_tags=["python-cpp-0.005-allstage2-sweep"],
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
