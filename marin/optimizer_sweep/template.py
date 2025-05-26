# https://github.com/stanford-crfm/marin/issues/725
# Sweep to determine optimal hyperparameters for Adam on small scale
from experiments.dclm.tokenize_dclm import dclm_mixture_config_llama3
from marin.execution.executor import executor_main
from marin.optimizer_sweep.config import map_tag_to_config
from marin.optimizer_sweep.format import map_tag_to_format
from experiments.optimizer_sweep.defaults import default_train
from marin.optimizer_sweep.models import calculate_chinchilla, map_tag_to_model
from marin.optimizer_sweep.utils import (
    approximate,
    check_baseline_run,
    config_to_train_config,
    create_configs,
    grab_best_run,
    make_sweep_steps,
)

# Sweep to determine optimal training config


def template(
    model_size,
    target_chinchilla,
    optimizer,
    baseline_config,
    sweep_grids,
    tpu_type="v5litepod-128",
    DEBUG_MODE=False,
    random_suffix=None,
    force_run=False,
):
    llama_model = map_tag_to_model[model_size]
    target_data = target_chinchilla * calculate_chinchilla(llama_model)
    data_size = f"{target_data//1_000_000_000}B"
    optimizer_config_generator = map_tag_to_config[optimizer]
    tags = (model_size, data_size, optimizer)
    approximate_best_config_list = []

    def optimal_run_set(baseline_config):
        target_steps, config_in_dict = create_configs(baseline_config, sweep_grids, target_data=target_data)
        train_configs = config_to_train_config(
            config_in_dict, target_steps, config_generator=optimizer_config_generator, tpu_type=tpu_type
        )
        # use wandb to avoid rerunning
        new_train_configs = []
        for train_config, config in zip(train_configs, config_in_dict, strict=False):
            if not check_baseline_run(config, tags):
                print(f"Unfinished: {config}")
                new_train_configs.append(train_config)
        return new_train_configs

    if force_run:
        # For force run, just create a single train config from baseline
        target_steps, _ = create_configs(baseline_config, {}, target_data=target_data)  # Empty sweep grid
        train_configs = config_to_train_config(
            [baseline_config], target_steps, config_generator=optimizer_config_generator, tpu_type=tpu_type
        )
        tags = ("debug",) + tags
    else:
        approximate_best_config_list = []
        if check_baseline_run(baseline_config, tags):
            current_best_config, approximate_best_config_list = grab_best_run(baseline_config.keys(), tags)
            baseline_config = current_best_config
            print(f"Current best config: {baseline_config}")
            print(f"Current Approximate best config: {approximate_best_config_list}")

        if len(approximate_best_config_list) <= 1:
            print("Dont have a choice")
            train_configs = optimal_run_set(baseline_config)
        else:
            train_configs = optimal_run_set(baseline_config)
            approximate_train_configs_len = [
                len(optimal_run_set(approx_baseline_config)) for approx_baseline_config in approximate_best_config_list
            ]
            if min(approximate_train_configs_len) <= len(train_configs) - 10:
                for i in range(len(approximate_best_config_list)):
                    if approximate_train_configs_len[i] == min(approximate_train_configs_len):
                        break
                baseline_config = approximate_best_config_list[i]
                train_configs = optimal_run_set(baseline_config)
            else:
                print("All configs are nearly equally close to finish")

    print(f"Choose: {baseline_config}")
    print(f"Closest to finish: {len(train_configs)}")
    if len(train_configs) > 0 and (not DEBUG_MODE):
        prefix = f"sweep-{model_size}-{data_size}-{optimizer}"
        if random_suffix is not None:
            prefix = prefix + random_suffix
        steps = make_sweep_steps(
            prefix=prefix,
            model_config=llama_model,
            train_configs=train_configs,
            tokenized_data=dclm_mixture_config_llama3,
            format_train_config=map_tag_to_format[optimizer],
            default_train=default_train,
            tags=("llama", "dclm") + tags,
        )
        executor_main(steps)

    if not force_run:  # Only do final examination if not force_run
        current_best_config, approximate_best_config_list = grab_best_run(baseline_config.keys(), tags)
        in_side = False
        for approximate_best_config in approximate_best_config_list:
            if approximate(approximate_best_config, baseline_config):
                in_side = True
                break
        if in_side:
            print(f"Found: {baseline_config}")
            current_run_set = optimal_run_set(baseline_config)
            if len(current_run_set) > 0:
                print("Stupid Ray", flush=True)
            else:
                print("Succeed!")
        else:
            print("Stupid Ray", flush=True)
