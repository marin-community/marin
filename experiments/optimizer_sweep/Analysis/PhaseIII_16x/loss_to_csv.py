import hashlib
import json

import pandas as pd

color_map = {
    "mars": "#1f77b4",  # blue
    "muon": "#ff7f0e",  # orange
    "lion": "#2ca02c",  # green
    "adamw": "#d62728",  # red
    "nadamw": "#9467bd",  # purple
    "kron": "#8c564b",  # brown
    "scion": "#e377c2",  # pink
    "cautious": "#7f7f7f",  # gray
    "soape": "#bcbd22",  # yellow-green
    "mini": "#aec7e8",  # light blue
}


with open("experiments/optimizer_sweep/Analysis/PhaseI/wandb_cache.json", "r") as f:
    cache_phase_I = json.load(f)

with open("experiments/optimizer_sweep/Analysis/PhaseII/wandb_cache.json", "r") as f:
    cache_phase_II = json.load(f)


with open("experiments/optimizer_sweep/Analysis/PhaseIII_16x/wandb_cache.json", "r") as f:
    cache = json.load(f)

cache.update(cache_phase_II)
cache.update(cache_phase_I)


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
    ("300m", "12B", 2),
    ("300m", "24B", 4),
    ("300m", "48B", 8),
    ("300m", "96B", 16),
    ("130m", "42B", 16),
]
model_sizes = ["130m", "300m"]

optimizers = ["muon", "adamw", "nadamw", "soape"]

actual_list = {}

for optimizer in optimizers:
    for model_size, data_size, target_chinchilla in model_and_data_size:
        cache_key = get_cache_key(optimizer, model_size, data_size, target_chinchilla)
        if cache_key in cache:
            actual_list[(optimizer, model_size, target_chinchilla)] = cache[cache_key]
        else:
            print(f"{optimizer} {model_size} {target_chinchilla} not found")


expected_params = {
    "130m": 134217728,  # 32 * (512*2048*3 + 512*512*4)
    "300m": 301989888,  # 32 * (768*3072*3 + 768*768*4)
    "520m": 536870912,  # 32 * (1024*4096*3 + 1024*1024*4)
    "1.2b": 1207959552,  # 32 * (1536*6144*3 + 1536*1536*4)
}


df = pd.DataFrame(columns=["optimizer", "model_size", "chinchilla", "loss"])
for model_size in model_sizes:
    for optimizer in optimizers:
        optimizer_loss_list = []
        for chinchilla in [1, 2, 4, 8, 16]:
            if (optimizer, model_size, chinchilla) in actual_list:
                df.loc[len(df)] = [
                    optimizer,
                    model_size,
                    chinchilla,
                    actual_list[(optimizer, model_size, chinchilla)]["min_loss"],
                ]
                optimizer_loss_list.append(actual_list[(optimizer, model_size, chinchilla)]["min_loss"])
            else:
                optimizer_loss_list.append(None)


df.to_csv("experiments/optimizer_sweep/Analysis/PhaseIII_16x/loss_to_csv.csv", index=False)
