import json
import hashlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

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

optimizers = ["mini", "lion", "sophia", "adamw", "nadamw", "mars", "cautious",   "soap","muon", "scion", "soape", "soapb", "kron"]


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
    for key in keys:
        best_config_key_list = {
            (model_size, target_chinchilla): [config[key] for config in best_config_list[(model_size, target_chinchilla)]] for model_size, target_chinchilla in best_config_list
        }
        # find whether one value of key is in all model_size, target_chinchilla
        potential_value_of_key = best_config_key_list[list(best_config_key_list.keys())[0]]
        stable = False
        for value in potential_value_of_key:
            if all(value in best_config_key_list[(model_size, target_chinchilla)] for model_size, target_chinchilla in best_config_key_list):
                stable = True
                break
        if stable:
            # print(f"{key} = {value} is stable for all model_size, target_chinchilla")
            pass
        else:
            # print(f"{key} is not stable for all model_size, target_chinchilla")
            non_stable_keys.append(key)
    print("Optimizer: ", optimizer)
    print("Non-stable keys: ", non_stable_keys)

