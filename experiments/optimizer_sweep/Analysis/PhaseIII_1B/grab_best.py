import hashlib
import json

cache_file = "experiments/optimizer_sweep/Analysis/PhaseIII_1B/wandb_cache.json"

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

optimizer = "adamw"
model_size = "130m"
data_size = "2B"
target_chinchilla = 1


for model_size, data_size, target_chinchilla in [('1.2b', '24B', 1)]:

    cache_key = get_cache_key(optimizer, model_size, data_size, target_chinchilla)
    if cache_key in cache:
        actual_list[(optimizer, model_size, target_chinchilla)] = cache[cache_key]
        best_config = (actual_list[(optimizer, model_size, target_chinchilla)]['best_config'])
        # best_config['adam_lr'] = best_config['learning_rate'] * best_config['muon_to_adam_lr']
        print(model_size, best_config)