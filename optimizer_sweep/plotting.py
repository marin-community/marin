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

# Define distinctive colors for each optimizer
color_map = {
    'mars': '#1f77b4',    # blue
    'muon': '#ff7f0e',    # orange
    'lion': '#2ca02c',    # green
    'adamw': '#d62728',   # red
    'nadamw': '#9467bd',  # purple
    'kron': '#8c564b',    # brown
    'scion': '#e377c2',   # pink
    'cautious': '#7f7f7f', # gray
    'soap': '#bcbd22',    # yellow-green
    'sophia': '#17becf',  # cyan
    'mini': '#aec7e8',    # light blue
    'soape': '#ffbb78',   # light orange
    'soapb': '#98df8a'    # light green
}

# Define line styles
line_styles = {
    'kron': '--',
    'scion': '--',
    'muon': '--',
    'soap': '--',
    'mars': '-',
    'adamw': '-',
    'nadamw': '-',
    'cautious': '-',
    'mini': '-',
    'sophia': '-',
    'lion': '-'
}

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

expected_params = {
    "130m": 134217728,    # 32 * (512*2048*3 + 512*512*4)
    "300m": 301989888,    # 32 * (768*3072*3 + 768*768*4)
    "520m": 536870912,    # 32 * (1024*4096*3 + 1024*1024*4)
    "1.2b": 1207959552,   # 32 * (1536*6144*3 + 1536*1536*4)
}

# Plot 1: Model size scaling
plt.figure(figsize=(10, 6))
plt.rcParams.update({'font.size': 14})
sns.set_style("whitegrid")

model_sizes = ['130m', '300m', '520m']
model_sizes_in_params = [expected_params[model_size] for model_size in model_sizes]

for optimizer in optimizers:
    optimizer_loss_list = []
    for model_size in model_sizes:
        if (optimizer, model_size, 1) in actual_list:
            optimizer_loss_list.append(actual_list[(optimizer, model_size, 1)]["min_loss"])
        else:
            optimizer_loss_list.append(None)
    linewidth = 3 if optimizer in ['adamw', 'nadamw', 'muon', 'soap'] else 1
    plt.plot(model_sizes_in_params, optimizer_loss_list, label=optimizer, 
             linewidth=linewidth, alpha=0.9, color=color_map[optimizer],
             linestyle=line_styles[optimizer])

plt.title('Model Size Scaling', fontsize=16)
plt.xlabel('Number of Parameters', fontsize=14)
plt.ylabel('Loss', fontsize=14)
plt.xticks(model_sizes_in_params, [f'{p/1e6:.0f}M' for p in model_sizes_in_params], fontsize=12)
plt.legend(fontsize=12, loc='upper right', framealpha=0.9)
plt.yticks(fontsize=12)
plt.savefig('optimizer_loss_chinchilla_1.png')
plt.close()

# Plot 2: Chinchilla scaling
plt.figure(figsize=(10, 6))
plt.rcParams.update({'font.size': 14})
sns.set_style("whitegrid")

chinchillas = [1, 2, 4, 8]

for optimizer in optimizers:
    optimizer_loss_list = []
    for chinchilla in chinchillas:
        if (optimizer, '130m', chinchilla) in actual_list:
            optimizer_loss_list.append(actual_list[(optimizer, '130m', chinchilla)]["min_loss"])
        else:
            optimizer_loss_list.append(None)
    linewidth = 3 if optimizer in ['adamw', 'nadamw', 'muon', 'soap'] else 1
    plt.plot(chinchillas, optimizer_loss_list, label=optimizer, 
             linewidth=linewidth, alpha=0.9, color=color_map[optimizer],
             linestyle=line_styles[optimizer])

plt.title('Chinchilla Scaling', fontsize=16)
plt.xlabel('Chinchilla', fontsize=14)
plt.ylabel('Loss', fontsize=14)
plt.xticks(chinchillas, [str(c) for c in chinchillas], fontsize=12)
plt.legend(fontsize=12, loc='upper right', framealpha=0.9)
plt.yticks(fontsize=12)
plt.savefig('optimizer_loss_chinchilla_130m.png')
plt.close()

# check configs for each optimizer and figure out all the keys that are not constant

for optimizer in optimizers:
    best_config_list = {}
    for model_size, data_size, target_chinchilla in model_and_data_size:
        if (optimizer, model_size, target_chinchilla) in actual_list:
            best_config = actual_list[(optimizer, model_size, target_chinchilla)]["best_config"]
            best_config_list[model_size] = best_config
    not_constant_keys = []
    for key in best_config_list[model_sizes[0]].keys():
        for model_size in model_sizes:
            if best_config_list[model_size][key] != best_config_list[model_sizes[0]][key]:
                not_constant_keys.append(key)
                break
    print(f"{optimizer} not constant keys: {not_constant_keys}")
