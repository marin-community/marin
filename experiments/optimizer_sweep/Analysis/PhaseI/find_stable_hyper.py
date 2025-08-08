import hashlib
import json

cache_file = "experiments/optimizer_sweep/Analysis/PhaseI/wandb_cache.json"

with open(cache_file, "r") as f:
    cache = json.load(f)


def get_cache_key(optimizer, model_size, data_size, target_chinchilla):
    """Generate a unique cache key for the query"""
    key_str = f"{optimizer}_{model_size}_{data_size}_{target_chinchilla}"
    return hashlib.md5(key_str.encode()).hexdigest()


model_and_data_size = [
    ("130m", "2B", 1),
    ("130m", "5B", 2),
    ("130m", "10B", 4),
    ("130m", "21B", 8),
    ("300m", "6B", 1),
    ("520m", "10B", 1),
]

optimizers = ["mini", "lion", "adamw", "nadamw", "mars", "cautious", "soape", "muon", "scion", "soape", "kron"]


actual_list = {}

for optimizer in optimizers:
    for model_size, data_size, target_chinchilla in model_and_data_size:
        cache_key = get_cache_key(optimizer, model_size, data_size, target_chinchilla)
        if cache_key in cache:
            actual_list[(optimizer, model_size, target_chinchilla)] = cache[cache_key]
        else:
            print(
                f"optimizer: {optimizer}, model_size: {model_size}, data_size: {data_size}, target_chinchilla: {target_chinchilla} not found"
            )


optimizers = ["adamw", "nadamw", "lion", "mini", "cautious", "mars", "scion", "muon", "soape", "kron"]

non_stable_keys_by_optimizer = {}

for optimizer in optimizers:
    best_config_list = {}
    for model_size, data_size, target_chinchilla in model_and_data_size:
        if (optimizer, model_size, target_chinchilla) in actual_list:
            approximate_best_config_list = actual_list[(optimizer, model_size, target_chinchilla)][
                "approximate_best_config_list"
            ]
            best_config_list[(model_size, target_chinchilla)] = approximate_best_config_list
    if len(best_config_list) == 0:
        continue
    keys = list(best_config_list[list(best_config_list.keys())[0]][0].keys())
    non_stable_keys = []
    recommended_config = {}
    sweep_grids = {}
    for key in keys:
        best_config_key_list = {
            (model_size, target_chinchilla): [
                config[key] for config in best_config_list[(model_size, target_chinchilla)]
            ]
            for model_size, target_chinchilla in best_config_list
        }
        # find whether one value of key is in all model_size, target_chinchilla
        potential_value_of_key = best_config_key_list[list(best_config_key_list.keys())[0]]
        stable = False
        for value in potential_value_of_key:
            if all(
                value in best_config_key_list[(model_size, target_chinchilla)]
                for model_size, target_chinchilla in best_config_key_list
            ):
                recommended_config[key] = value
                stable = True
                break
        if stable:
            pass
        else:
            non_stable_keys.append(key)
    non_stable_keys_by_optimizer[optimizer] = non_stable_keys
    print(f"Optimizer: {optimizer} has the following non-stable keys: {non_stable_keys}")


import json

json.dump(non_stable_keys_by_optimizer, open("experiments/optimizer_sweep/Analysis/PhaseI/non_stable_keys_by_optimizer.json", "w"))
