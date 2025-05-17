import json
import hashlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.optimize import curve_fit

cache_file = "wandb_cache.json"

with open(cache_file, "r") as f:
    cache = json.load(f)

print(cache)

def get_cache_key(optimizer, model_size, data_size, target_chinchilla):
    """Generate a unique cache key for the query"""
    key_str = f"{optimizer}_{model_size}_{data_size}_{target_chinchilla}"
    return hashlib.md5(key_str.encode()).hexdigest()


model_and_data_size = [('130m', '2B', 1), ('130m', '5B', 2), ('130m', '10B', 4), 
                        ('130m', '21B', 8), ('300m', '6B', 1), ('520m', '10B', 1)] + [('300m', '12B', 2), ('300m', '24B', 4), ('300m', '48B', 8), ('520m', '21B', 2), ('520m', '42B', 4), ('520m', '85B', 8)]

optimizers = ["mini", "lion", "adamw", "nadamw", "mars", "cautious", "soap","muon", "scion", "kron", "soape", "soapb"]

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

model_sizes = ['130m', '300m', '520m']
model_sizes_in_params = [expected_params[model_size] for model_size in model_sizes]

plt.close()

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
df.to_csv('optimizer_loss_scaling.csv', index=False)


import pandas as pd
df = pd.DataFrame(columns=['optimizer', 'model_size', 'chinchilla', 'wd'])
with open('optimizer_wd.md', 'w') as f:
    for idx, model_size in enumerate(model_sizes):
        for optimizer in optimizers:
            optimizer_loss_list = []
            for chinchilla in [1, 2, 4, 8]:
                if (optimizer, model_size, chinchilla) in actual_list:
                    f.write(f"{optimizer} {model_size} {chinchilla} {actual_list[(optimizer, model_size, chinchilla)]['best_config']['weight_decay']}\n")
                    df.loc[len(df)] = [optimizer, model_size, chinchilla, actual_list[(optimizer, model_size, chinchilla)]["best_config"]["weight_decay"]]
                else:
                    df.loc[len(df)] = [optimizer, model_size, chinchilla, None]
df.to_csv('optimizer_wd.csv', index=False)

from utils_simp import grab_run, create_configs, calculate_data_tag

def get_benchmark_acc(run):
    benchmarks = ['lambada_openai', 'openbookqa', 'winogrande', 'piqa', 'boolq', 'wsc273', 'hellaswag_0shot', 'arc_challenge', 'arc_easy', 'copa']
    acc_dict = {}
    for benchmark in benchmarks:
        acc = run.summary.get(f'lm_eval/{benchmark}/acc', 0.0)
        acc_norm = run.summary.get(f'lm_eval/{benchmark}/acc_norm', 0.0)
        acc_dict[benchmark] = max(acc, acc_norm)
    print(acc_dict)
    return acc_dict
# import pandas as pd
# benchmarks = ['lambada_openai', 'openbookqa', 'winogrande', 'piqa', 'boolq', 'wsc273', 'hellaswag_0shot', 'arc_challenge', 'arc_easy', 'copa']
# performance_table = pd.DataFrame(columns=['optimizer', 'model_size', 'chinchilla'] + benchmarks)

# with open('optimizer_loss_eval.md', 'w') as f:
#     for idx, model_size in enumerate(model_sizes):
#         fig, ax = plt.subplots(figsize=(10, 6))
#         for optimizer in ['mars']:
#             optimizer_loss_list = []
#             for chinchilla in [1, 2, 4, 8]:
#                 if (optimizer, model_size, chinchilla) in actual_list:
#                     best_config = actual_list[(optimizer, model_size, chinchilla)]["best_config"]
#                     target_data, data_size = calculate_data_tag(model_size, chinchilla)
#                     tags = (model_size, data_size, optimizer)
#                     run = grab_run(best_config, tags)
#                     if run is not None: 
#                         acc_dict = get_benchmark_acc(run)
#                         f.write(f"{optimizer} {model_size} {chinchilla} {acc_dict}\n")
#                         # add to performance table
#                         performance_table.loc[len(performance_table)] = [optimizer, model_size, chinchilla] + [acc_dict[benchmark] for benchmark in benchmarks]
#                     else:
#                         if optimizer == 'soap':
#                             run = grab_run(best_config, (model_size, data_size, 'soape'))
#                             if run is not None:
#                                 acc_dict = get_benchmark_acc(run)
#                                 f.write(f"{optimizer} {model_size} {chinchilla} {acc_dict}\n")
#                                 performance_table.loc[len(performance_table)] = [optimizer, model_size, chinchilla] + [acc_dict[benchmark] for benchmark in benchmarks]
#                         else:
#                             f.write(f"{optimizer} {model_size} {chinchilla} None\n")
#                 else:
#                     optimizer_loss_list.append(None)


# performance_table.to_csv('optimizer_loss_eval.csv', index=False)


x_loss_list = []
y_loss_list = []
label_list = []
labeled_already = []
for optimizer in optimizers:
    optimizer_loss_list = []
    for chinchilla in [1, 2, 4, 8]:
        if (optimizer, '300m', chinchilla) in actual_list and (optimizer, '520m', chinchilla) in actual_list:
            x_loss_list.append(actual_list[(optimizer, '300m', chinchilla)]["min_loss"])
            y_loss_list.append(actual_list[(optimizer, '520m', chinchilla)]["min_loss"])
            if optimizer not in labeled_already:
                labeled_already.append(optimizer)
                plt.scatter(x_loss_list[-1], y_loss_list[-1], label=f"{optimizer}", color=color_map[optimizer])
            else:
                plt.scatter(x_loss_list[-1], y_loss_list[-1], color=color_map[optimizer])

    # linewidth = 3 if optimizer in ['adamw', 'nadamw', 'muon', 'soap'] else 1
    # linewidth = 1
# perform a linear fit on the data
# popt, _ = curve_fit(lambda t, A, B: A * t + B, x_loss_list, y_loss_list)
# plt.plot(sorted(x_loss_list), popt[0] * np.array(sorted(x_loss_list)) + popt[1], label=f"linear fit: {popt[0]:.2f} * x + {popt[1]:.2f}")
# # fix the size of the plot
# plt.gcf().set_size_inches(10, 6)
# plt.title(f'Using Loss of 300m to Predict Loss of 520m', fontsize=16)
# plt.xlabel('Loss of 300m', fontsize=14)
# plt.ylabel('Loss of 520m', fontsize=14)
# plt.legend(fontsize=12, loc='upper left', framealpha=0.9)
# plt.yticks(fontsize=12)
# plt.savefig(f'optimizer_loss_scaling_300m_520m.png')
# plt.close()


# # provide a heatmap of the difference between muon and adamw
# difference_list_of_list = []

# for model_size in model_sizes:
#     difference_list = []
#     difference_list.append(actual_list[('nadamw', model_size, 1)]["min_loss"] - actual_list[('adamw', model_size, 1)]["min_loss"])
#     difference_list.append(actual_list[('nadamw', model_size, 2)]["min_loss"] - actual_list[('adamw', model_size, 2)]["min_loss"])
#     difference_list.append(actual_list[('nadamw', model_size, 4)]["min_loss"] - actual_list[('adamw', model_size, 4)]["min_loss"])
#     difference_list.append(actual_list[('nadamw', model_size, 8)]["min_loss"] - actual_list[('adamw', model_size, 8)]["min_loss"])
#     difference_list_of_list.append(difference_list)

# import seaborn as sns

# # Create a heatmap
# plt.figure(figsize=(10, 6))
# sns.heatmap(difference_list_of_list, annot=True, cmap='coolwarm', center=0)
# plt.title('Difference between NadamW and AdamW')
# plt.xticks([0.5, 1.5, 2.5, 3.5], [1, 2, 4, 8])
# plt.yticks([0.5, 1.5, 2.5], model_sizes)
# plt.xlabel('Chinchilla')
# plt.ylabel('Model Size')
# plt.savefig(f'optimizer_loss_scaling_nadamw_adamw_heatmap.png')
# plt.close()

# # plot the curve in the same plot 
# # plt.figure(figsize=(10, 6))
# # for i in range(len(model_sizes)):
# #     plt.scatter([1, 2, 4, 8], difference_list_of_list[i] - min(difference_list_of_list[i]), label=model_sizes[i])
# # plt.plot(np.arange(-0.07, 0.01, 0.01), np.arange(-0.07, 0.01, 0.01), label='y = x')
# # plt.legend()
# # plt.savefig(f'optimizer_loss_scaling_muon_adamw_fit.png')
# # plt.close()



# fit a power law that is A * model_size^B  + C * data_size^D + E to simulate the difference between muon and adamw

x = [(expected_params[model_size], chinchilla) for model_size in model_sizes for chinchilla in [1, 2, 4, 8]]
y = [actual_list[('muon', model_size, chinchilla)]["min_loss"] - actual_list[('nadamw', model_size, chinchilla)]["min_loss"] for model_size in model_sizes for chinchilla in [1, 2, 4, 8]]

# provide a guess for the parameters
initial_guess = [0.0, 0.0, 0.0, -1.0, 0.0]  # A, B, C, D, E
popt, _ = curve_fit(lambda t, A, B, C, D, E: A * t[:, 0]**B + C * t[:, 1]**D + E, x, y, p0=initial_guess, maxfev=10000)

with open("muon_nadamw_difference.md", "w") as f:
    for model_size in model_sizes:
        for chinchilla in [1, 2, 4, 8]:
            f.write(f"{model_size} {chinchilla} {actual_list[('muon', model_size, chinchilla)]['min_loss'] - actual_list[('nadamw', model_size, chinchilla)]['min_loss']}\n")

# simulated_difference_list_of_list = []
# for model_size in model_sizes:
#     simulated_difference_list = []
#     for chinchilla in [1, 2, 4, 8]:
#         simulated_difference_list.append(popt[0] * expected_params[model_size]**popt[1] + popt[2] * (chinchilla)**popt[3] + popt[4])
#     simulated_difference_list_of_list.append(simulated_difference_list)
# plt.xlabel("Simulated Difference")
# plt.ylabel("Actual Difference")

# # only show two numbers in total
# plt.scatter(simulated_difference_list_of_list, difference_list_of_list, label=f"difference fit:\n {popt[0]:.2f} * D^{popt[1]:.2f} + {popt[2]:.2e} * C^{popt[3]:.2f} + {popt[4]:.2f}")
# plt.plot(np.arange(-0.07, 0.01, 0.01), np.arange(-0.07, 0.01, 0.01), label='y = x')
# plt.legend()
# plt.savefig(f'optimizer_loss_scaling_muon_adamw_fit.png')
# plt.close()







scaling_laws = {}
plt.close()
# fit a scaling law for optimizer in optimizers
fitting_dict_by_optimizer = {}
best_loss_given_size = {}
for optimizer in optimizers:
    fitting_dict = {}
    for model_size in model_sizes:
        for chinchilla in [1, 2, 4, 8]:
            if (optimizer, model_size, chinchilla) in actual_list:
                fitting_dict[(expected_params[model_size], expected_params[model_size] * 20 * chinchilla)] = actual_list[(optimizer, model_size, chinchilla)]["min_loss"]
                if (expected_params[model_size], expected_params[model_size] * 20 * chinchilla) not in best_loss_given_size:
                    best_loss_given_size[(expected_params[model_size], expected_params[model_size] * 20 * chinchilla)] = actual_list[(optimizer, model_size, chinchilla)]["min_loss"]
                else:
                    if actual_list[(optimizer, model_size, chinchilla)]["min_loss"] < best_loss_given_size[(expected_params[model_size], expected_params[model_size] * 20 * chinchilla)]:
                        best_loss_given_size[(expected_params[model_size], expected_params[model_size] * 20 * chinchilla)] = actual_list[(optimizer, model_size, chinchilla)]["min_loss"]
    fitting_dict_by_optimizer[optimizer] = fitting_dict


# fit a scaling law for each optimizer

x = np.array(list(best_loss_given_size.keys()))
y = np.array(list(best_loss_given_size.values()))
initial_guess = [1.0, -0.5, 1.0, -0.5, 1.0]  # A, B, C, D, E
base_popt, _ = curve_fit(lambda t, A, B, C, D, E: A * t[:, 0]**B + C * t[:, 1]**D + E, x, y, p0=initial_guess, maxfev=10000)
print(base_popt)

with open("AdamW.md", "w") as f:
    fitting_dict = fitting_dict_by_optimizer["adamw"]
    for key in fitting_dict:
        f.write(f"{key[0]}, {key[1]}, {fitting_dict[key]}\n")

with open("Muon.md", "w") as f:
    fitting_dict = fitting_dict_by_optimizer["muon"]
    for key in fitting_dict:
        f.write(f"{key[0]}, {key[1]}, {fitting_dict[key]}\n")

with open("Soap.md", "w") as f:
    fitting_dict = fitting_dict_by_optimizer["soap"]
    for key in fitting_dict:
        f.write(f"{key[0]}, {key[1]}, {fitting_dict[key]}\n")

with open("Kron.md", "w") as f:
    fitting_dict = fitting_dict_by_optimizer["kron"]
    for key in fitting_dict:
        f.write(f"{key[0]}, {key[1]}, {fitting_dict[key]}\n")




for optimizer in optimizers:
    fitting_dict = fitting_dict_by_optimizer[optimizer]
    # fit a power law that is A * model_size^B  + C * data_size^D
    x = np.array(list(fitting_dict.keys()))
    # print(x)re
    y = np.array(list(fitting_dict.values()))
    # fit a power law that is A * model_size^B  + C * data_size^D + E
    initial_guess = base_popt[:4] # A, B, C, D, E
    popt, _ = curve_fit(lambda t, A, B, C, D: A * t[:, 0]**B + C * t[:, 1]**D + base_popt[4], x, y, p0=initial_guess, maxfev=10000)
    print(popt)
    # plot the fitting curve
    predicted_loss = popt[0] * x[:, 0]**popt[1] + popt[2] * x[:, 1]**popt[3] + base_popt[4]
    # plt.plot(x, y, 'o', label=optimizer)
    plt.plot(predicted_loss, y, 'o', label='Actual Data for ' + optimizer)
    plt.plot(np.arange(predicted_loss.min(), predicted_loss.max() + 0.01, 0.01), np.arange(predicted_loss.min(), predicted_loss.max() + 0.01, 0.01), label='y = x')
    plt.scatter(predicted_loss[-1], y[-1], color='red', label='Leave One Out Verification', s=100)
    plt.xlabel('Predicted Loss')
    plt.ylabel('Actual Loss')
    plt.legend()
    plt.savefig(f'optimizer_loss_scaling_fitting_{optimizer}.png')
    plt.close()
    # calculate regression error
    regression_error = np.sqrt(np.mean((predicted_loss - y)**2))
    print(f'Regression error for {optimizer}: {regression_error}')
    scaling_laws[optimizer] = popt
    
plt.figure(figsize=(10, 6))
# estimate the scaling law on a 1.2B model 
for optimizer in scaling_laws:
    model_size_in_params =  1207959552
    data_size_in_params_list = [model_size_in_params * 20 * chinchilla for chinchilla in [1, 2, 4, 8]]
    predicted_loss_list = [scaling_laws[optimizer][0] * data_size_in_params**scaling_laws[optimizer][1] + scaling_laws[optimizer][2] * data_size_in_params**scaling_laws[optimizer][3] + base_popt[4] for data_size_in_params in data_size_in_params_list]
    plt.plot([1, 2, 4, 8], predicted_loss_list, label=optimizer)
plt.legend()
plt.xlabel('Chinchilla')
plt.ylabel('Loss')
plt.title('Estimated Scaling Law for 1.2B Model')
plt.savefig(f'optimizer_loss_scaling_fitting_1.2B.png')
plt.close()

print(scaling_laws)


predicted_configs = {}
# optimal hyperparameters for AdamW
for optimizer_name in ['kron']:
    hyperparameters_dict = {}

    for model_size in model_sizes:
        for chinchilla in [1, 2, 4, 8]:
            if (optimizer_name, model_size, chinchilla) in actual_list:
                hyperparameters_dict[(model_size, chinchilla)] = actual_list[(optimizer_name, model_size, chinchilla)]["best_config"]
                if model_size == '520m' and chinchilla == 8:
                    hyperparameters_dict[(model_size, chinchilla)]['learning_rate'] = 0.004
    keys = list(hyperparameters_dict[(model_sizes[0], 1)].keys())

    with open(f'hyperparameters_fit_{optimizer_name}.md', 'w') as f:
        for key in keys:
            # fit a power law that is A * model_size^B * chinchilla^C + D
            x = [(expected_params[model_size], chinchilla) for model_size in model_sizes for chinchilla in [1, 2, 4, 8]]
            y = [hyperparameters_dict[(model_size, chinchilla)][key] for model_size in model_sizes for chinchilla in [1, 2, 4, 8]]
            # fit a power law and print error
            if type(y[-1]) == float or type(y[-1]) == int:
                if key == "muon_to_adam_lr":
                    continue
                baseline = np.mean(y[:-1])
                popt, _ = curve_fit(lambda t, A, B, C, D: A * t[:, 0]**B * t[:, 1]**C + D, x[1:-1], y[1:-1], p0=[0.0, -0.5, -0.5, baseline], maxfev=80000)
                # print error on the last point
                predicted_loss = popt[0] * x[-1][0]**popt[1] * x[-1][1]**popt[2] + popt[3]
                error = np.sqrt(np.mean((predicted_loss - y[-1])**2))
                f.write(f"Relative error for {key}: {error / (y[-1] + 1e-6)}\n")
                parameter = expected_params['1.2b']
                for chinchilla in [1, 2, 4, 8]:
                    f.write(f"For 1.2B with {chinchilla} chinchilla, {key} = {popt[0] * parameter**popt[1] * chinchilla**popt[2] + popt[3]}\n")
                    if (optimizer_name, '1.2b', chinchilla) not in predicted_configs:
                        predicted_configs[(optimizer_name, '1.2b', chinchilla)] = {}
                    predicted_configs[(optimizer_name, '1.2b', chinchilla)][key] = popt[0] * parameter**popt[1] * chinchilla**popt[2] + popt[3]

import pickle 
with open('predicted_configs.pkl', 'wb') as f:
    pickle.dump(predicted_configs, f)



