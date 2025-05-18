import json
import hashlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.optimize import curve_fit

with open("optimizer_sweep_run/Analysis/PhaseI/wandb_cache.json", "r") as f:
    cache_phase_I = json.load(f)
with open("optimizer_sweep_run/Analysis/PhaseI/wandb_cache.json", "r") as f:
    cache_phase_II = json.load(f)
with open("optimizer_sweep_run/Analysis/PhaseIII_1B/wandb_cache.json", "r") as f:
    cache = json.load(f)

cache.update(cache_phase_I)
cache.update(cache_phase_II)


def get_cache_key(optimizer, model_size, data_size, target_chinchilla):
    """Generate a unique cache key for the query"""
    key_str = f"{optimizer}_{model_size}_{data_size}_{target_chinchilla}"
    return hashlib.md5(key_str.encode()).hexdigest()


model_and_data_size = [('130m', '2B', 1), ('130m', '5B', 2), ('130m', '10B', 4), 
                        ('130m', '21B', 8), ('300m', '6B', 1), ('520m', '10B', 1)] + [('300m', '12B', 2), ('300m', '24B', 4), ('300m', '48B', 8), ('520m', '21B', 2), ('520m', '42B', 4), ('520m', '85B', 8)] + [('1.2b', '24B', 1), ('1.2b', '48B', 2), ('1.2b', '96B', 4), ('1.2b', '193B', 8)]

optimizers = ["mini", "lion", "adamw", "nadamw", "mars", "cautious", "muon", "scion", "kron", "soape"]

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
    # 'soape': '#ffbb78',   # light orange
    # 'soapb': '#98df8a'    # light green
}

# Define line styles
# line_styles = {
#     'kron': '--',
#     'scion': '--',
#     'muon': '--',
#     'soap': '--',
#     'mars': '-',
#     'adamw': '-',
#     'nadamw': '-',
#     'cautious': '-',
#     'mini': '-',
#     'sophia': '-',
#     'lion': '-'
# }

actual_list = {}

for optimizer in optimizers:
    for model_size, data_size, target_chinchilla in model_and_data_size:
        cache_key = get_cache_key(optimizer, model_size, data_size, target_chinchilla)
        if cache_key in cache:
            actual_list[(optimizer, model_size, target_chinchilla)] = cache[cache_key]
        # else:
        #     print(f"{optimizer} {model_size} {target_chinchilla} not found")


for model_size, data_size, target_chinchilla in model_and_data_size:
    num_left = 10000
    soap_list = ['soap', 'soape', 'soapb']
    actual_optimizer = None
    for soap_optimizer in soap_list:
        if (soap_optimizer, model_size, target_chinchilla) in actual_list:
            if actual_list[(soap_optimizer, model_size, target_chinchilla)]["min_num"] < num_left:
                actual_optimizer = soap_optimizer
                num_left = actual_list[(soap_optimizer, model_size, target_chinchilla)]["min_num"]
    if actual_optimizer is not None:
        actual_list[('soap', model_size, target_chinchilla)] = actual_list[(actual_optimizer, model_size, target_chinchilla)]

for model_size, data_size, target_chinchilla in model_and_data_size:
    if ('soapb', model_size, target_chinchilla) in actual_list:
        actual_list.pop(('soapb', model_size, target_chinchilla))
    if ('soape', model_size, target_chinchilla) in actual_list:
        actual_list.pop(('soape', model_size, target_chinchilla))


for optimizer in optimizers:
    if optimizer in ["soapb", "soape", "sophia"]:
        continue
    for model_size, data_size, target_chinchilla in model_and_data_size:
        if (optimizer, model_size, target_chinchilla) not in actual_list:
            print(f"{optimizer} {model_size} {target_chinchilla} not found")
        elif actual_list[(optimizer, model_size, target_chinchilla)]["min_num"] > 0:
            print(f"{optimizer} {model_size} {target_chinchilla} has {actual_list[(optimizer, model_size, target_chinchilla)]['min_num']} missing runs")
optimizers.remove('soapb')
optimizers.remove('soape')

expected_params = {
    "130m": 134217728,    # 32 * (512*2048*3 + 512*512*4)
    "300m": 301989888,    # 32 * (768*3072*3 + 768*768*4)
    "520m": 536870912,    # 32 * (1024*4096*3 + 1024*1024*4)
    "1.2b": 1207959552,   # 32 * (1536*6144*3 + 1536*1536*4)
}

# Plot 1: Model size scaling
plt.figure(figsize=(15, 5))  # Wider figure to accommodate subplots
plt.rcParams.update({'font.size': 14})
sns.set_style("whitegrid")

model_sizes = ['1.2b']
model_sizes_in_params = [expected_params[model_size] for model_size in model_sizes]

plt.close()
from utils_simp import grab_run, create_configs, calculate_data_tag

username = "stanford-mercury"
project = "optimizer-scaling"
import wandb
api = wandb.Api()


# tag = "sweep-130m-10B-muonzf7964alr0.008-wd0.1-minlr0-warmup0-b10.8-b20-a9a044"
# tag = "sweep-130m-2B-nadamw96aba0lr0.008-wd0.1-minlr0-warmup2000-b10.95-2ac247"
tag = "sweep-300m-12B-marskb945b4lr0.008-wd0.1-minlr0-warmup1000-b10.95-2538fa"


def get_benchmark_acc(name):
    run = api.run(f"{username}/{project}/{name}")
    benchmarks = ['lambada_openai', 'openbookqa', 'winogrande', 'piqa', 'boolq', 'wsc273', 'hellaswag_0shot', 'arc_challenge', 'arc_easy', 'copa']
    acc_dict = {}
    for benchmark in benchmarks:
        acc = run.summary.get(f'lm_eval/{benchmark}/acc', 0.0)
        acc_norm = run.summary.get(f'lm_eval/{benchmark}/acc_norm', 0.0)
        acc_dict[benchmark] = max(acc, acc_norm)
    print(acc_dict)
    return acc_dict


import pandas as pd
df = pd.DataFrame(columns=['optimizer', 'model_size', 'chinchilla', 'loss'])
with open('optimizer_loss_scaling.md', 'w') as f:
    for idx, model_size in enumerate(model_sizes):
        for optimizer in optimizers:
            optimizer_loss_list = []
            for chinchilla in [1, 2, 4, 8]:
                if (optimizer, model_size, chinchilla) in actual_list:
                    f.write(f"{optimizer} {model_size} {chinchilla} {actual_list[(optimizer, model_size, chinchilla)]['min_loss']}\n")
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
df.to_csv('optimizer_loss_scaling_large.csv', index=False)


import pandas as pd
benchmarks = ['lambada_openai', 'openbookqa', 'winogrande', 'piqa', 'boolq', 'wsc273', 'hellaswag_0shot', 'arc_challenge', 'arc_easy', 'copa']
performance_table = pd.DataFrame(columns=['optimizer', 'model_size', 'chinchilla'] + benchmarks)

with open('optimizer_loss_eval.md', 'w') as f:
    for idx, model_size in enumerate(model_sizes):
        fig, ax = plt.subplots(figsize=(10, 6))
        for optimizer in ['adamw', 'nadamw', 'muon']:
            optimizer_loss_list = []
            for chinchilla in [1, 2, 4, 8]:
                if (optimizer, model_size, chinchilla) in actual_list:
                    best_config = actual_list[(optimizer, model_size, chinchilla)]["best_config"]
                    target_data, data_size = calculate_data_tag(model_size, chinchilla)
                    tags = (model_size, data_size, optimizer)
                    run = grab_run(best_config, tags)
                    print(run.name)
                    if run is not None: 
                        acc_dict = get_benchmark_acc(run.name)
                        f.write(f"{optimizer} {model_size} {chinchilla} {acc_dict}\n")
                        # add to performance table
                        performance_table.loc[len(performance_table)] = [optimizer, model_size, chinchilla] + [acc_dict[benchmark] for benchmark in benchmarks]
                    else:
                        if optimizer == 'soap':
                            run = grab_run(best_config, (model_size, data_size, 'soape'))
                            if run is not None:
                                print(run.name)
                                acc_dict = get_benchmark_acc(run.name)
                                f.write(f"{optimizer} {model_size} {chinchilla} {acc_dict}\n")
                                performance_table.loc[len(performance_table)] = [optimizer, model_size, chinchilla] + [acc_dict[benchmark] for benchmark in benchmarks]
                        else:
                            f.write(f"{optimizer} {model_size} {chinchilla} None\n")
                else:
                    optimizer_loss_list.append(None)


performance_table.to_csv('optimizer_loss_eval_large.csv', index=False)
