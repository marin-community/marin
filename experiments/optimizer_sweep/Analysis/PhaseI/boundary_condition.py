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

with open("experiments/optimizer_sweep/Analysis/PhaseI/sweep_configurations.json", "r") as f:
    sweep_configs = json.load(f)

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

boundary_keys_by_optimizer = {}

recheck_pairs = []

for optimizer in optimizers:
    for model_size, data_size, target_chinchilla in model_and_data_size:
        if (optimizer, model_size, target_chinchilla) in actual_list:
            best_config = actual_list[(optimizer, model_size, target_chinchilla)]["best_config"]
            optimizer_configs = sweep_configs.get(optimizer.lower(), {})
            model_configs = optimizer_configs.get(model_size, {})
            sweep_config = model_configs.get(str(target_chinchilla), [{}])[0]['sweep_grids']
            for key, value in best_config.items():
                if sweep_config.get(key, None) is None:
                    continue
                else:
                    values = sorted(sweep_config[key])
                    if type(value) not in [float, int]:
                        continue
                    if value not in values:
                        if value < 1e-25:
                            continue
                        print(f"This is a problem for {optimizer} on {model_size} with {target_chinchilla}, {key}:{value} is not in {values}")
                    if value in [values[0], values[-1]]:
                        if (optimizer, model_size, target_chinchilla) not in boundary_keys_by_optimizer:
                            boundary_keys_by_optimizer[(optimizer, model_size, target_chinchilla)] = []
                        boundary_keys_by_optimizer[(optimizer, model_size, target_chinchilla)].append((key, value, values))


from experiments.optimizer_sweep.spin_up_more import rewrite
root_dir = "experiments/optimizer_sweep/PhaseI"
target_dir = "experiments/optimizer_sweep/PhaseI_Bound"
for optimizer in optimizers:
    for model_size, data_size, target_chinchilla in model_and_data_size:
        keys = boundary_keys_by_optimizer.get((optimizer, model_size, target_chinchilla), [])
        if len(keys) == 0:
            continue
        selected_keys = [key for key, value, values in keys]
        best_value = [value for key, value, values in keys]
        best_values = [values for key, value, values in keys]
        overwrite_sweep_grids = {}
        print(keys)
        for key, value, values in keys:
            if 'batch_size' in key:
                if value == 128:
                    continue
                else:
                    overwrite_sweep_grids[key] = [value, 2 * value]
            elif 'learning_rate' in key:
                if values[0] == value:
                    overwrite_sweep_grids[key] = [values[0] * values[0] / values[1], values[0]]
                else:
                    overwrite_sweep_grids[key] = [values[-1], values[-1] * values[-1] / values[-2]]
            elif 'weight_decay' in key or 'preconditioner_lr' in key:
                if values[0] == value:
                    raise Exception(f"This never happens")
                else:
                    overwrite_sweep_grids[key] = [0.2, 0.3]
            elif 'scion_to_signum_lr' in key:
                if values[0] == value:
                    overwrite_sweep_grids[key] = [0.0005, 0.001]
                else:
                    raise Exception(f"This never happens")
            elif 'muon_to_adam_lr' in key:
                if values[-1] == value:
                    overwrite_sweep_grids[key] = [0.3, 0.4]
                else:
                    raise Exception(f"This never happens")
            else:
                continue
        if len(overwrite_sweep_grids) > 0:
            recheck_pairs.append((optimizer, model_size, target_chinchilla))
            rewrite(root_dir, optimizer, model_size, target_chinchilla, model_size, target_chinchilla, selected_keys = overwrite_sweep_grids.keys(), overwrite_sweep_grids = overwrite_sweep_grids, target_dir = target_dir)


with open("recheck.sh", "w") as f:
    for pair in recheck_pairs:
        f.write(f"bash run.sh {pair[0]} {pair[1]} {pair[2]} \n")



