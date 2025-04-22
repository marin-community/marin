import json
import hashlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from utils_simp import grab_best_run

cache_file = "wandb_cache.json"

with open(cache_file, "r") as f:
    cache = json.load(f)

print(cache)

def get_cache_key(optimizer, model_size, data_size, target_chinchilla):
    """Generate a unique cache key for the query"""
    key_str = f"{optimizer}_{model_size}_{data_size}_{target_chinchilla}"
    return hashlib.md5(key_str.encode()).hexdigest()




model_and_data_size = [('130m', '2B', 1), ('130m', '5B', 2), ('130m', '10B', 4), 
                        ('130m', '21B', 8), ('300m', '6B', 1), ('520m', '10B', 1)]

optimizers = ["mini", "lion", "sophia", "adamw", "nadamw", "mars", "cautious",  "soap","muon", "scion", "soape", "soapb", "kron"]


actual_list = {}

for optimizer in optimizers:
    for model_size, data_size, target_chinchilla in model_and_data_size:
        cache_key = get_cache_key(optimizer, model_size, data_size, target_chinchilla)
        if cache_key in cache:
            actual_list[(optimizer, model_size, target_chinchilla)] = cache[cache_key]
        else:
            print(f"Cache key {cache_key} not found")


for model_size, data_size, target_chinchilla in model_and_data_size:
    min_loss = 10000
    soap_list = ['soap', 'soape', 'soapb']
    actual_optimizer = None
    for soap_optimizer in soap_list:
        if (soap_optimizer, model_size, target_chinchilla) in actual_list:
            if actual_list[(soap_optimizer, model_size, target_chinchilla)]["min_loss"] < min_loss:
                actual_optimizer = soap_optimizer
                min_loss = actual_list[(soap_optimizer, model_size, target_chinchilla)]["min_loss"]
    if actual_optimizer is not None:
        actual_list[('soap', model_size, target_chinchilla)] = actual_list[(actual_optimizer, model_size, target_chinchilla)]

for model_size, data_size, target_chinchilla in model_and_data_size:
    if ('soapb', model_size, target_chinchilla) in actual_list:
        actual_list.pop(('soapb', model_size, target_chinchilla))
    if ('soape', model_size, target_chinchilla) in actual_list:
        actual_list.pop(('soape', model_size, target_chinchilla))
    

optimizers.remove('soapb')
optimizers.remove('soape')


import os
import re
root_dir = "optimizer_sweep"
from optimizer_sweep.external_monitor import parse_command_file, calculate_data_tag

# This regex assumes filenames follow a pattern such as:
#   exp<digits>_<optimizer>sweep_<model_size>_<target>.py
pattern = re.compile(r"exp\d+_([a-z]+)sweep_(\d+M)_(\d+)\.py$", re.IGNORECASE)

def rewrite(optimizer_name, base_model_size, base_target_chinchilla, real_model_size, real_target_chinchilla, recommended_config, sweep_grids):
    # Walk through the directory tree.
    try:
        for dirpath, dirnames, filenames in os.walk(root_dir):
            for filename in filenames:
                if optimizer_name == 'soap':
                    optimizer_name = 'soape'
                # We look for files ending with '_130M_1.py'
                if filename.endswith(f"_{optimizer_name}sweep_{base_model_size.upper()}_{base_target_chinchilla}.py"):
                    match = pattern.match(filename)
                    if match:
                        # Parse the file for configuration data.
                        full_file_path = os.path.join(dirpath, filename)
                        parsed_data = parse_command_file(full_file_path)
                        first_baseline = parsed_data['baseline_config']
                        model_size = parsed_data['model_size']
                        target_chinchilla = parsed_data['target_chinchilla']
                        optimizer = parsed_data['optimizer_name']
                        
                        # Calculate additional tags if needed.
                        target_data, data_size = calculate_data_tag(model_size, target_chinchilla)
                        tags = (model_size, data_size, optimizer)
                        
                        # Retrieve the best configuration.
                        current_best_config, approximate_best_config_list = grab_best_run(first_baseline.keys(), tags)
                        print(optimizer, current_best_config)

                        for key in current_best_config:
                            if key not in recommended_config:
                                recommended_config[key] = current_best_config[key]

                        # try to be smart with learning rate and batch size
                        lr = recommended_config['learning_rate']
                        if 'learning_rate' not in sweep_grids:
                            sweep_grids['learning_rate'] = [lr/2, lr]
                        sweep_grids['learning_rate'] = [lr for lr in sweep_grids['learning_rate'] if lr <= recommended_config['learning_rate']]
                        if min(sweep_grids['learning_rate']) == lr:
                            lr = sweep_grids['learning_rate'][0]
                            sweep_grids['learning_rate'] = [lr/2, lr]
                        # try larget batch size
                        bs = recommended_config['train_batch_size']
                        if 'train_batch_size' not in sweep_grids:
                            sweep_grids['train_batch_size'] = [bs * 2, bs]
                        sweep_grids['train_batch_size'] = [bs for bs in sweep_grids['train_batch_size'] if bs >= recommended_config['train_batch_size']]
                        if max(sweep_grids['train_batch_size']) == bs:
                            sweep_grids['train_batch_size'] = [bs * 2, bs]
                        if 'warmup' in sweep_grids:
                            sweep_grids['warmup'] = [w for w in sweep_grids['warmup'] if w >= recommended_config['warmup']]
                        if 'partition_grads_into_blocks' in sweep_grids and optimizer_name == 'soape':
                            sweep_grids['partition_grads_into_blocks'] = [True]
                        if 'precondition_frequency' in sweep_grids and optimizer_name == 'soape':
                            sweep_grids['precondition_frequency'] = [10]

                        # Create the new filename. Digits are fixed to 1.
                        if optimizer_name == 'soape':
                            optimizer_name = 'soap'
                        
                        new_filename = f"exp725_{optimizer_name}sweep_{real_model_size.upper()}_{real_target_chinchilla}.py"
                        new_file_path = os.path.join(root_dir, new_filename)
                        print(f"key_of_optimizer['{optimizer_name}'] = {list(first_baseline.keys())}")
                        
                        # Generate the new script using current_best_config.
                        with open(new_file_path, "w") as new_file:
                            new_file.write("from optimizer_sweep.template import template\n")
                            new_file.write("\n")
                            new_file.write("if __name__ == '__main__':\n")
                            new_file.write(f"    sweep_grids = {sweep_grids}\n")
                            new_file.write(f"    baseline_config = {current_best_config}\n")
                            new_file.write(f"    model_size = '{real_model_size}'\n")
                            new_file.write(f"    target_chinchilla = {real_target_chinchilla}\n")
                            new_file.write("    my_suffix = None\n")
                            new_file.write(f"    template(model_size, target_chinchilla, '{optimizer_name}', baseline_config, sweep_grids, random_suffix=my_suffix)\n")
                        
                        print(f"Generated new script: {new_file_path}")
    except Exception as e:
        print(f"Error: {e}")

target_model_and_data_sizes = [('300m', 2), ('300m', 4), ('300m', 8)] + [('520m', 2), ('520m', 4), ('520m', 8)]

optimizers = ['adamw', 'nadamw', 'lion', 'mini', 'cautious', 'mars', 'scion', 'muon', 'soap', 'kron']

for optimizer in optimizers:
    best_config_list = {}
    for model_size, data_size, target_chinchilla in model_and_data_size:
        if (optimizer, model_size, target_chinchilla) in actual_list:
            approximate_best_config_list = actual_list[(optimizer, model_size, target_chinchilla)]["approximate_best_config_list"]
            best_config_list[(model_size, target_chinchilla)] = approximate_best_config_list
    if len(best_config_list) == 0:
        continue
    keys = list(best_config_list[list(best_config_list.keys())[0]][0].keys())
    non_stable_keys = []
    recommended_config = {}
    sweep_grids = {}
    for key in keys:
        best_config_key_list = {
            (model_size, target_chinchilla): [config[key] for config in best_config_list[(model_size, target_chinchilla)]] for model_size, target_chinchilla in best_config_list
        }
        # find whether one value of key is in all model_size, target_chinchilla
        potential_value_of_key = best_config_key_list[list(best_config_key_list.keys())[0]]
        stable = False
        for value in potential_value_of_key:
            if all(value in best_config_key_list[(model_size, target_chinchilla)] for model_size, target_chinchilla in best_config_key_list):
                recommended_config[key] = value
                stable = True
                break
        if stable:
            # print(f"{key} = {value} is stable for all model_size, target_chinchilla")
            pass
        else:
            # print(f"{key} is not stable for all model_size, target_chinchilla")
            non_stable_keys.append(key)
    print("Optimizer: ", optimizer)
    print("Recommended config: ", recommended_config)
    # plot a heatmap with x axis as model_size, y axis as target_chinchilla, and the value is the number of non_stable_keys for each non_stable_key, save them to figs/optimizer_non_stable_keys.png
    for key in non_stable_keys:
        best_config_key_list = {
            (model_size, target_chinchilla): set([config[key] for config in best_config_list[(model_size, target_chinchilla)]]) for model_size, target_chinchilla in best_config_list
        }
        # add all possible values of key to sweep_grids
        sweep_grids[key] = set()
        for model_size, target_chinchilla in best_config_key_list:
            for value in best_config_key_list[(model_size, target_chinchilla)]:
                sweep_grids[key].add(value)
        sweep_grids[key] = list(sweep_grids[key])
        # print(key)
        # pretty print the best_config_key_list
        # import pprint
        # pprint.pprint(best_config_key_list)
    import pprint
    pprint.pprint(sweep_grids)

    for target_model_size, target_data_size in target_model_and_data_sizes:
        rewrite(optimizer, target_model_size, 1, target_model_size, target_data_size, recommended_config, sweep_grids)
    print("--------------------------------")
print(optimizers)