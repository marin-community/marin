import os
import json

RESULTS_DIR_DEFAULT = "experiments/optimizer_sweep/Analysis/Results"


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
        result_path = os.path.join(
            RESULTS_DIR_DEFAULT, optimizer, model_size, str(target_chinchilla), "result.json"
        )
        if os.path.exists(result_path):
            with open(result_path, "r") as f:
                payload = json.load(f)
            actual_list[(optimizer, model_size, target_chinchilla)] = payload
        else:
            print(
                f"Result not found at {result_path} for optimizer: {optimizer}, model_size: {model_size}, data_size: {data_size}, target_chinchilla: {target_chinchilla}"
            )


optimizers = ["adamw", "nadamw", "lion", "mini", "cautious", "mars", "scion", "muon", "soape", "kron"]

non_stable_keys_by_optimizer = {}

for optimizer in optimizers:
    best_config_list = {}
    for model_size, data_size, target_chinchilla in model_and_data_size:
        if (optimizer, model_size, target_chinchilla) in actual_list:
            payload = actual_list[(optimizer, model_size, target_chinchilla)]
            if "approximate_best_config_list" in payload:
                approximate_best_config_list = payload["approximate_best_config_list"]
            elif "best_config" in payload:
                approximate_best_config_list = [payload["best_config"]]
            else:
                continue
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



with open("experiments/optimizer_sweep/Analysis/non_stable_keys_by_optimizer.json", "w") as f:
    json.dump(non_stable_keys_by_optimizer, f)
