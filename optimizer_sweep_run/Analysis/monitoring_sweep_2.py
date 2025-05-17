import glob
import re
import ast
from optimizer_sweep.utils_simp import approximate, create_configs, check_baseline_run, grab_best_run, calculate_data_tag, grab_run
import subprocess
import random
import json
import tqdm
import multiprocessing
from functools import partial
import tqdm
import os
import hashlib
from datetime import datetime

# Cache file path
CACHE_FILE = 'wandb_cache.json'

benchmarks = ['lambada_openai', 'openbookqa', 'winogrande', 'piqa', 'boolq', 'wsc273', 'hellaswag_0shot', 'arc_challenge', 'arc_easy', 'copa']

def get_benchmark_acc(run):
    acc_dict = {}
    for benchmark in benchmarks:
        acc = run.summary.get(f'lm_eval/{benchmark}/acc', 0.0)
        acc_norm = run.summary.get(f'lm_eval/{benchmark}/acc_norm', 0.0)
        acc_dict[benchmark] = max(acc, acc_norm)
    print(acc_dict)
    return acc_dict

def get_cache():
    """Load the cache file if it exists, otherwise return an empty dict"""
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_cache(cache):
    """Save the cache to file"""
    with open(CACHE_FILE, 'w') as f:
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
    
    # Create progress bar for model sizes with position parameter
    for model_size, data_size, target_chinchilla in tqdm.tqdm(
        model_and_data_size, 
        desc=f"Processing {optimizer}",
        position=position,
        leave=False
    ):
        tags = (model_size, data_size, optimizer)
        optimizer_results[(model_size, data_size)] = {}
        target_data, data_size = calculate_data_tag(model_size, target_chinchilla)
        
        
        # Get sweep configuration from saved configs
        optimizer_configs = sweep_configs.get(optimizer.lower(), {})
        model_configs = optimizer_configs.get(model_size, {})

        cache_key = get_cache_key(optimizer, model_size, data_size, target_chinchilla)
        best_config = None
        if cache_key in cache:
            optimizer_results[(model_size, data_size)] = cache[cache_key]
            best_config = optimizer_results[(model_size, data_size)]["best_config"]
        sweep_config = model_configs.get(str(target_chinchilla), [{}])[0]
        sweep_grids = sweep_config.get('sweep_grids', None)
        
        # if not sweep_grids:
        #     optimizer_results[(model_size, data_size)]["min_num"] = "Missing Grids"
        #     continue
        if best_config is None:
            continue
        print(best_config)
        min_num_left = float('inf')
        num_left_config, losses = num_left(best_config, sweep_grids, target_data, tags, printing= True, return_loss = True)
        optimizer_results[(model_size, data_size)]['losses'] = losses
        optimizer_results[(model_size, data_size)]['best_config'] = best_config
        run = grab_run(best_config, tags)
        if run is not None:
            paloma_loss = run.history(keys=['eval/paloma/c4_en/loss'], pandas=True)
            optimizer_results[(model_size, data_size)]['paloma_loss'] = paloma_loss
            optimizer_results[(model_size, data_size)]['lm_eval_acc'] = get_benchmark_acc(run)
    
    return optimizer, optimizer_results


def num_left(baseline_config, sweep_grids, target_data, tags, printing = False, return_loss = False):
    """Modified num_left function that takes required parameters"""
    target_steps, config_in_dict = create_configs(baseline_config, sweep_grids, target_data=target_data)
    num_left = 0
    losses = {}
    for config in config_in_dict:
        if not return_loss:
            flag = check_baseline_run(config, tags)
        else:
            flag, loss = check_baseline_run(config, tags, return_loss = True)
        if flag:
            for key, value in config.items():
                if baseline_config[key] != value:
                    if loss is not None:
                        losses[(key, value)] = loss
        else:
            num_left += 1
    if return_loss:
        return num_left, losses
    else:
        return num_left

key_of_optimizer = dict()
key_of_optimizer['mars'] = ['learning_rate', 'weight_decay', 'min_lr_ratio', 'warmup', 'beta1', 'beta2', 'gamma', 'epsilon', 'max_grad_norm', 'train_batch_size']
key_of_optimizer['muon'] = ['learning_rate', 'weight_decay', 'min_lr_ratio', 'warmup', 'momentum', 'beta1', 'beta2', 'epsilon', 'muon_epsilon', 'max_grad_norm', 'lr_schedule', 'muon_to_adam_lr', 'decay', 'train_batch_size']
key_of_optimizer['lion'] = ['learning_rate', 'weight_decay', 'min_lr_ratio', 'warmup', 'beta1', 'beta2', 'max_grad_norm', 'train_batch_size']
key_of_optimizer['nadamw'] = ['learning_rate', 'weight_decay', 'min_lr_ratio', 'warmup', 'beta1', 'beta2', 'epsilon', 'max_grad_norm', 'nesterov', 'train_batch_size']
key_of_optimizer['kron'] = ['learning_rate', 'weight_decay', 'beta1', 'preconditioner_lr', 'preconditioner_init_scale', 'max_grad_norm', 'normalize_grads', 'partition_grads_into_blocks', 'block_size', 'preconditioner_update_probability', 'update_prob_flat_start', 'warmup', 'min_lr_ratio', 'train_batch_size']
key_of_optimizer['scion'] = ['learning_rate', 'weight_decay', 'min_lr_ratio', 'warmup', 'momentum', 'beta1', 'scion_epsilon', 'max_grad_norm', 'lr_schedule', 'scion_to_signum_lr', 'decay', 'train_batch_size']
key_of_optimizer['cautious'] = ['learning_rate', 'weight_decay', 'min_lr_ratio', 'warmup', 'beta1', 'beta2', 'epsilon', 'max_grad_norm', 'train_batch_size']
key_of_optimizer['soap'] = ['learning_rate', 'weight_decay', 'min_lr_ratio', 'warmup', 'beta1', 'beta2', 'shampoo_beta', 'precondition_frequency', 'partition_grads_into_blocks', 'block_size', 'epsilon', 'max_grad_norm', 'train_batch_size']
key_of_optimizer['soape'] = key_of_optimizer['soap']
key_of_optimizer['soapb'] = key_of_optimizer['soap']
key_of_optimizer['adamw'] = ['learning_rate', 'weight_decay', 'min_lr_ratio', 'warmup', 'beta1', 'beta2', 'epsilon', 'max_grad_norm', 'nesterov', 'train_batch_size']
key_of_optimizer['mini'] = ['learning_rate', 'weight_decay', 'min_lr_ratio', 'warmup', 'beta1', 'beta2', 'epsilon', 'max_grad_norm', 'train_batch_size']


optimizers =  list(key_of_optimizer.keys())



new_key_of_optimizer = {}
for optimizer in key_of_optimizer.keys():
    new_key_of_optimizer[optimizer] = key_of_optimizer[optimizer]
key_of_optimizer = new_key_of_optimizer

def process_with_index(idx, process_funcs, optimizer_keys):
    """Helper function to avoid using lambda"""
    return process_funcs[idx](optimizer_keys[idx])

if __name__ == '__main__':
    # Load the saved sweep configurations
    with open('sweep_configurations.json', 'r') as f:
        sweep_configs = json.load(f)
    
    model_and_data_size = [('130m', '2B', 1), ('130m', '5B', 2), ('130m', '10B', 4), 
                          ('130m', '21B', 8), ('300m', '6B', 1), ('520m', '10B', 1)] + [('300m', '12B', 2), ('300m', '24B', 4), ('300m', '48B', 8), ('520m', '21B', 2), ('520m', '42B', 4), ('520m', '85B', 8)]
    # model_and_data_size = [('130m', '21B', 8)]
    # model_and_data_size = [ ('130m', '10B', 4), ('130m', '21B', 8),  ('520m', '10B', 1)]
    # model_and_data_size = [('130m', '42B', 16)]

    # model_and_data_size = [('300m', '12B', 2), ('300m', '24B', 4), ('300m', '48B', 8), ('520m', '21B', 2), ('520m', '42B', 4), ('520m', '85B', 8)]
    # model_and_data_size = [('1.2b', '24B', 1), ('1.2b', '48B', 2), ('1.2b', '96B', 4)]
    # model_and_data_size = [('520m', '10B', 1)]
    # model_and_data_size = [('300m', '48B', 8)]


    monitoring_results = {}
    
    # Process each optimizer sequentially with a progress bar
    with tqdm.tqdm(
        total=len(key_of_optimizer),
        desc="Overall progress",
        position=0,
        leave=True
    ) as pbar:
        for position, optimizer in enumerate(key_of_optimizer.keys(), 1):
            optimizer, result = process_optimizer(
                optimizer,
                model_and_data_size=model_and_data_size,
                sweep_configs=sweep_configs,
                position=position
            )
            monitoring_results[optimizer] = result
            pbar.update(1)
    import pickle
    with open('losses_all.pkl', 'wb') as f:
        pickle.dump(monitoring_results, f)
    
    





