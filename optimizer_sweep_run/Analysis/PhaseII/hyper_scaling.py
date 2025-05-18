import hashlib
import json

import numpy as np
from scipy.optimize import curve_fit

with open("optimizer_sweep_run/Analysis/PhaseI/wandb_cache.json", "r") as f:
    cache_phase_I = json.load(f)


with open("optimizer_sweep_run/Analysis/PhaseII/wandb_cache.json", "r") as f:
    cache_phase_II = json.load(f)

cache_phase_I.update(cache_phase_II)
cache = cache_phase_I


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
] + [
    ("300m", "12B", 2),
    ("300m", "24B", 4),
    ("300m", "48B", 8),
    ("520m", "21B", 2),
    ("520m", "42B", 4),
    ("520m", "85B", 8),
]
model_sizes = ["130m", "300m", "520m"]

optimizers = ["mini", "lion", "adamw", "nadamw", "mars", "cautious", "muon", "scion", "kron", "soape"]

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

predicted_configs = {}
# optimal hyperparameters for AdamW
for optimizer_name in ["muon", "adamw", "nadamw"]:
    hyperparameters_dict = {}
    for model_size in model_sizes:
        for chinchilla in [1, 2, 4, 8]:
            if (optimizer_name, model_size, chinchilla) in actual_list:
                hyperparameters_dict[(model_size, chinchilla)] = actual_list[(optimizer_name, model_size, chinchilla)][
                    "best_config"
                ]
    keys = list(hyperparameters_dict[(model_sizes[0], 1)].keys())

    with open(f"hyperparameters_fit_{optimizer_name}.md", "w") as f:
        for key in keys:
            # fit a power law that is A * model_size^B * chinchilla^C + D
            x = [(expected_params[model_size], chinchilla) for model_size in model_sizes for chinchilla in [1, 2, 4, 8]]
            y = [
                hyperparameters_dict[(model_size, chinchilla)][key]
                for model_size in model_sizes
                for chinchilla in [1, 2, 4, 8]
            ]
            # fit a power law and print error
            if type(y[-1]) == float or type(y[-1]) == int:
                if key == "muon_to_adam_lr":
                    continue
                baseline = np.mean(y[:-1])
                popt, _ = curve_fit(
                    lambda t, A, B, C, D: A * t[:, 0] ** B * t[:, 1] ** C + D,
                    x[1:-1],
                    y[1:-1],
                    p0=[0.0, -0.5, -0.5, baseline],
                    maxfev=80000,
                )
                # print error on the last point
                predicted_loss = popt[0] * x[-1][0] ** popt[1] * x[-1][1] ** popt[2] + popt[3]
                error = np.sqrt(np.mean((predicted_loss - y[-1]) ** 2))
                f.write(f"Relative error for {key}: {error / (y[-1] + 1e-6)}\n")
                parameter = expected_params["1.2b"]
                for chinchilla in [1, 2, 4, 8]:
                    f.write(
                        f"For 1.2B with {chinchilla} chinchilla, {key} = {popt[0] * parameter**popt[1] * chinchilla**popt[2] + popt[3]}\n"
                    )
                    if (optimizer_name, "1.2b", chinchilla) not in predicted_configs:
                        predicted_configs[(optimizer_name, "1.2b", chinchilla)] = {}
                    predicted_configs[(optimizer_name, "1.2b", chinchilla)][key] = float(
                        popt[0] * parameter ** popt[1] * chinchilla ** popt[2] + popt[3]
                    )
    print(f"Predicted configs for {optimizer_name}: {predicted_configs}")

import pickle

with open("optimizer_sweep_run/Analysis/PhaseII/predicted_configs.pkl", "wb") as f:
    pickle.dump(predicted_configs, f)
