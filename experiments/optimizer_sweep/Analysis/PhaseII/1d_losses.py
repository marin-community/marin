import hashlib
import json
import os

import tqdm

from marin.optimizer_sweep.utils_simp import calculate_data_tag, check_baseline_run, create_configs, grab_best_run

# Cache file path
CACHE_FILE = "experiments/optimizer_sweep/Analysis/PhaseII/wandb_cache.json"


def get_cache():
    """Load the cache file if it exists, otherwise return an empty dict"""
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r") as f:
            return json.load(f)
    return {}

def map_optimizer_name(optimizer):
    optimizer_name_dict = {
        'adamw': 'AdamW',
        'lion': 'Lion',
        'mini': 'Adam-Mini',
        'scion': 'Scion',
        'cautious': 'Cautious',
        'mars': 'Mars',
        'nadamw': 'NAdamW',
        'muon': 'Muon',
        'soape': 'Soap',
        'kron': 'Kron',
    }
    return optimizer_name_dict[optimizer]

def save_cache(cache):
    """Save the cache to file"""
    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f, indent=2)


def get_cache_key(optimizer, model_size, data_size, target_chinchilla):
    """Generate a unique cache key for the query"""
    key_str = f"{optimizer}_{model_size}_{data_size}_{target_chinchilla}"
    return hashlib.md5(key_str.encode()).hexdigest()


def process_optimizer(optimizer, model_and_data_size, sweep_configs, position):
    """
    Process a single optimizer's configurations and return its monitoring results
    """
    optimizer_results = {}
    keys = key_of_optimizer[optimizer]

    # Load cache
    cache = get_cache()
    output_dict = {}

    # Create progress bar for model sizes with position parameter
    for model_size, data_size, target_chinchilla in tqdm.tqdm(
        model_and_data_size, desc=f"Processing {optimizer}", position=position, leave=False
    ):
        tags = (model_size, data_size, optimizer)
        optimizer_results[(model_size, data_size)] = {}
        target_data, data_size = calculate_data_tag(model_size, target_chinchilla)

        # Check cache first
        cache_key = get_cache_key(optimizer, model_size, data_size, target_chinchilla)
        optimizer_results[(model_size, data_size)] = cache[cache_key]

        # Get sweep configuration from saved configs
        optimizer_configs = sweep_configs.get(optimizer.lower(), {})
        model_configs = optimizer_configs.get(model_size, {})

        if not optimizer_configs or not model_configs:
            continue

        current_best_config, approximate_best_config_list, min_loss = grab_best_run(
            keys, tags, return_loss=True, thshold=6e-3
        )
        # print(current_best_config)
        if approximate_best_config_list is None or not approximate_best_config_list:
            optimizer_results[(model_size, data_size)]["min_num"] = "Missing Runs"
            continue

        sweep_config = model_configs.get(str(target_chinchilla), [{}])[0]
        sweep_grids = sweep_config.get("sweep_grids", None)

        if not sweep_grids:
            optimizer_results[(model_size, data_size)]["min_num"] = "Missing Grids"
            continue

        
        current_num_left = 10

        for best_config in approximate_best_config_list:
            num_left, config_to_loss_dict, config_to_name_dict = config_to_loss(best_config, sweep_grids, target_data, tags, printing=True)
            if num_left < current_num_left:
                current_num_left = num_left
                output_dict[(model_size, target_chinchilla, optimizer)] = {}
                output_dict[(model_size, target_chinchilla, optimizer)]['result'] = config_to_loss_dict
                output_dict[(model_size, target_chinchilla, optimizer)]['name'] = config_to_name_dict
                output_dict[(model_size, target_chinchilla, optimizer)]['num_left'] = num_left
                output_dict[(model_size, target_chinchilla, optimizer)]['best_config'] = best_config
                output_dict[(model_size, target_chinchilla, optimizer)]['min_loss'] = min_loss

    return optimizer, output_dict


def config_to_loss(baseline_config, sweep_grids, target_data, tags, printing=False):
    """Modified num_left function that takes required parameters"""
    target_steps, config_in_dict = create_configs(baseline_config, sweep_grids, target_data=target_data)
    config_to_loss = {}
    config_to_name = {}
    num_left = 0
    for config in config_in_dict:
        exist, loss, name = check_baseline_run(config, tags, return_loss=True)
        num_left += 1 if not exist else 0
        different_key = [key for key in config.keys() if config[key] != baseline_config[key]]
        if len(different_key) == 1:
            config_to_loss[(different_key[0], config[different_key[0]])] = loss
            config_to_name[(different_key[0], config[different_key[0]])] = name
        else:
            config_to_loss['Baseline'] = loss
            config_to_name['Baseline'] = name
    return num_left, config_to_loss, config_to_name


key_of_optimizer = dict()
key_of_optimizer["mars"] = [
    "learning_rate",
    "weight_decay",
    "min_lr_ratio",
    "warmup",
    "beta1",
    "beta2",
    "gamma",
    "epsilon",
    "max_grad_norm",
    "train_batch_size",
]
key_of_optimizer["muon"] = [
    "learning_rate",
    "weight_decay",
    "min_lr_ratio",
    "warmup",
    "momentum",
    "beta1",
    "beta2",
    "epsilon",
    "muon_epsilon",
    "max_grad_norm",
    "lr_schedule",
    "muon_to_adam_lr",
    "decay",
    "train_batch_size",
]
key_of_optimizer["lion"] = [
    "learning_rate",
    "weight_decay",
    "min_lr_ratio",
    "warmup",
    "beta1",
    "beta2",
    "max_grad_norm",
    "train_batch_size",
]
key_of_optimizer["nadamw"] = [
    "learning_rate",
    "weight_decay",
    "min_lr_ratio",
    "warmup",
    "beta1",
    "beta2",
    "epsilon",
    "max_grad_norm",
    "nesterov",
    "train_batch_size",
]
key_of_optimizer["kron"] = [
    "learning_rate",
    "weight_decay",
    "beta1",
    "preconditioner_lr",
    "preconditioner_init_scale",
    "max_grad_norm",
    "normalize_grads",
    "partition_grads_into_blocks",
    "block_size",
    "preconditioner_update_probability",
    "update_prob_flat_start",
    "warmup",
    "min_lr_ratio",
    "train_batch_size",
]
key_of_optimizer["scion"] = [
    "learning_rate",
    "weight_decay",
    "min_lr_ratio",
    "warmup",
    "momentum",
    "beta1",
    "scion_epsilon",
    "max_grad_norm",
    "lr_schedule",
    "scion_to_signum_lr",
    "decay",
    "train_batch_size",
]
key_of_optimizer["cautious"] = [
    "learning_rate",
    "weight_decay",
    "min_lr_ratio",
    "warmup",
    "beta1",
    "beta2",
    "epsilon",
    "max_grad_norm",
    "train_batch_size",
]
key_of_optimizer["soape"] = [
    "learning_rate",
    "weight_decay",
    "min_lr_ratio",
    "warmup",
    "beta1",
    "beta2",
    "shampoo_beta",
    "precondition_frequency",
    "partition_grads_into_blocks",
    "block_size",
    "epsilon",
    "max_grad_norm",
    "train_batch_size",
]
key_of_optimizer["adamw"] = [
    "learning_rate",
    "weight_decay",
    "min_lr_ratio",
    "warmup",
    "beta1",
    "beta2",
    "epsilon",
    "max_grad_norm",
    "nesterov",
    "train_batch_size",
]
key_of_optimizer["mini"] = [
    "learning_rate",
    "weight_decay",
    "min_lr_ratio",
    "warmup",
    "beta1",
    "beta2",
    "epsilon",
    "max_grad_norm",
    "train_batch_size",
]


optimizers = list(key_of_optimizer.keys())


new_key_of_optimizer = {}
for optimizer in key_of_optimizer.keys():
    new_key_of_optimizer[optimizer] = key_of_optimizer[optimizer]
key_of_optimizer = new_key_of_optimizer


def process_with_index(idx, process_funcs, optimizer_keys):
    """Helper function to avoid using lambda"""
    return process_funcs[idx](optimizer_keys[idx])


if __name__ == "__main__":
    # Load the saved sweep configurations
    with open("experiments/optimizer_sweep/Analysis/PhaseII/sweep_configurations.json", "r") as f:
        sweep_configs = json.load(f)

    model_and_data_size = [
        ("300m", "12B", 2),
        ("300m", "24B", 4),
        ("300m", "48B", 8),
        ("520m", "21B", 2),
        ("520m", "42B", 4),
        ("520m", "85B", 8),
    ]
    monitoring_results = {}

    # Process each optimizer sequentially with a progress bar
    with tqdm.tqdm(total=len(key_of_optimizer), desc="Overall progress", position=0, leave=True) as pbar:
        for position, optimizer in enumerate(key_of_optimizer.keys(), 1):
            optimizer, result = process_optimizer(
                optimizer, model_and_data_size=model_and_data_size, sweep_configs=sweep_configs, position=position
            )
            monitoring_results[optimizer] = result
            print(result)
            pbar.update(1)
    import pickle

    with open("experiments/optimizer_sweep/Analysis/PhaseII/1d_losses.pkl", "wb") as f:
        pickle.dump(monitoring_results, f)
