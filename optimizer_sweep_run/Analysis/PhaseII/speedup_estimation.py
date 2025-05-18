import json
import hashlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.optimize import curve_fit

with open("optimizer_sweep_run/Analysis/PhaseI/wandb_cache.json", "r") as f:
    cache_phase_I = json.load(f)


with open("optimizer_sweep_run/Analysis/PhaseII/wandb_cache.json","r") as f:
    cache_phase_II = json.load(f)

cache_phase_I.update(cache_phase_II)
cache = cache_phase_I

def get_cache_key(optimizer, model_size, data_size, target_chinchilla):
    """Generate a unique cache key for the query"""
    key_str = f"{optimizer}_{model_size}_{data_size}_{target_chinchilla}"
    return hashlib.md5(key_str.encode()).hexdigest()


model_and_data_size = [('130m', '2B', 1), ('130m', '5B', 2), ('130m', '10B', 4), 
                        ('130m', '21B', 8), ('300m', '6B', 1), ('520m', '10B', 1)] + [('300m', '12B', 2), ('300m', '24B', 4), ('300m', '48B', 8), ('520m', '21B', 2), ('520m', '42B', 4), ('520m', '85B', 8)]
model_sizes = ['130m', '300m', '520m']

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
    "130m": 134217728,    # 32 * (512*2048*3 + 512*512*4)
    "300m": 301989888,    # 32 * (768*3072*3 + 768*768*4)
    "520m": 536870912,    # 32 * (1024*4096*3 + 1024*1024*4)
    "1.2b": 1207959552,   # 32 * (1536*6144*3 + 1536*1536*4)
}

import pandas as pd
df = pd.DataFrame(columns=['optimizer', 'model_size', 'chinchilla', 'loss'])
for idx, model_size in enumerate(model_sizes):
    for optimizer in optimizers:
        optimizer_loss_list = []
        for chinchilla in [1, 2, 4, 8]:
            if (optimizer, model_size, chinchilla) in actual_list:
                df.loc[len(df)] = [optimizer, model_size, chinchilla, actual_list[(optimizer, model_size, chinchilla)]["min_loss"]]
                optimizer_loss_list.append(actual_list[(optimizer, model_size, chinchilla)]["min_loss"])
            else:
                optimizer_loss_list.append(None)
        # linewidth = 2
        plt.plot([0, 1, 2, 3], optimizer_loss_list, label=optimizer if idx == 0 else None, color=color_map[optimizer])

    plt.title(f'{model_size} Model', fontsize=20)
    plt.xlabel('Chinchilla Ratio', fontsize=20)
    plt.ylabel('Loss', fontsize=20)
    # ax.set_xscale('log')
    plt.xticks([0, 1, 2, 3], [1, 2, 4, 8], fontsize=20)
    # plt.xscale('log')
    # ax.tick_params(axis='both', which='major', labelsize=18)
    plt.legend(loc='upper right', fontsize=18)
    plt.tight_layout()
    plt.savefig(f'optimizer_loss_scaling_{model_size}.pdf', bbox_inches='tight')
    plt.close()



