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

optimizers = ["mini", "lion", "sophia", "adamw", "nadamw", "mars", "cautious", "soap","muon", "scion", "kron", "soape", "soapb"]

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
        # else:
        #     print(f"{optimizer} {model_size} {target_chinchilla} not found")


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
plt.figure(figsize=(10, 6))
plt.rcParams.update({'font.size': 14})
sns.set_style("whitegrid")

model_sizes = ['130m', '300m', '520m']
model_sizes_in_params = [expected_params[model_size] for model_size in model_sizes]


# actual_list.pop(('adamw', '520m', 8))
actual_list.pop(('kron', '300m', 8))
# # optimizers = ['nadamw', 'kron', 'adamw', 'muon', 'mars']
# # optimi/
optimizers.remove("sophia")


for model_size in model_sizes:
    for optimizer in optimizers:
        optimizer_loss_list = []
        for chinchilla in [1, 2, 4, 8]:
            if (optimizer, model_size, chinchilla) in actual_list:
                optimizer_loss_list.append(actual_list[(optimizer, model_size, chinchilla)]["min_loss"])
            else:
                optimizer_loss_list.append(None)
        # linewidth = 3 if optimizer in ['adamw', 'nadamw', 'muon', 'soap'] else 1
        linewidth = 1
        if model_size == '130m':
            plt.plot([1, 2, 4, 8], optimizer_loss_list, label=optimizer, 
                    linewidth=linewidth, alpha=0.9, color=color_map[optimizer])
        else:
            plt.plot([1, 2, 4, 8], optimizer_loss_list, label=optimizer,
                    linewidth=linewidth, alpha=0.9, color=color_map[optimizer])
    # fix the size of the plot
    plt.gcf().set_size_inches(10, 6)
    plt.title(f'Data Size Scaling in {model_size}', fontsize=16)
    plt.xlabel('Chinchilla', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.legend(fontsize=12, loc='upper right', framealpha=0.9)
    plt.yticks(fontsize=12)
    plt.savefig(f'optimizer_loss_scaling_{model_size}.png')
    plt.close()

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
popt, _ = curve_fit(lambda t, A, B: A * t + B, x_loss_list, y_loss_list)
plt.plot(sorted(x_loss_list), popt[0] * np.array(sorted(x_loss_list)) + popt[1], label=f"linear fit: {popt[0]:.2f} * x + {popt[1]:.2f}")
# fix the size of the plot
plt.gcf().set_size_inches(10, 6)
plt.title(f'Using Loss of 300m to Predict Loss of 520m', fontsize=16)
plt.xlabel('Loss of 300m', fontsize=14)
plt.ylabel('Loss of 520m', fontsize=14)
plt.legend(fontsize=12, loc='upper left', framealpha=0.9)
plt.yticks(fontsize=12)
plt.savefig(f'optimizer_loss_scaling_300m_520m.png')
plt.close()


# provide a heatmap of the difference between muon and adamw
difference_list_of_list = []

for model_size in model_sizes:
    difference_list = []
    difference_list.append(actual_list[('nadamw', model_size, 1)]["min_loss"] - actual_list[('adamw', model_size, 1)]["min_loss"])
    difference_list.append(actual_list[('nadamw', model_size, 2)]["min_loss"] - actual_list[('adamw', model_size, 2)]["min_loss"])
    difference_list.append(actual_list[('nadamw', model_size, 4)]["min_loss"] - actual_list[('adamw', model_size, 4)]["min_loss"])
    difference_list.append(actual_list[('nadamw', model_size, 8)]["min_loss"] - actual_list[('adamw', model_size, 8)]["min_loss"])
    difference_list_of_list.append(difference_list)

import seaborn as sns

# Create a heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(difference_list_of_list, annot=True, cmap='coolwarm', center=0)
plt.title('Difference between NadamW and AdamW')
plt.xticks([0.5, 1.5, 2.5, 3.5], [1, 2, 4, 8])
plt.yticks([0.5, 1.5, 2.5], model_sizes)
plt.xlabel('Chinchilla')
plt.ylabel('Model Size')
plt.savefig(f'optimizer_loss_scaling_nadamw_adamw_heatmap.png')
plt.close()

# plot the curve in the same plot 
# plt.figure(figsize=(10, 6))
# for i in range(len(model_sizes)):
#     plt.scatter([1, 2, 4, 8], difference_list_of_list[i] - min(difference_list_of_list[i]), label=model_sizes[i])
# plt.plot(np.arange(-0.07, 0.01, 0.01), np.arange(-0.07, 0.01, 0.01), label='y = x')
# plt.legend()
# plt.savefig(f'optimizer_loss_scaling_muon_adamw_fit.png')
# plt.close()



# fit a power law that is A * model_size^B  + C * data_size^D + E to simulate the difference between muon and adamw

x = [(expected_params[model_size], chinchilla) for model_size in model_sizes for chinchilla in [1, 2, 4, 8]]
y = [actual_list[('muon', model_size, chinchilla)]["min_loss"] - actual_list[('adamw', model_size, chinchilla)]["min_loss"] for model_size in model_sizes for chinchilla in [1, 2, 4, 8]]

# provide a guess for the parameters
initial_guess = [0.0, 0.0, 0.0, -1.0, 0.0]  # A, B, C, D, E
popt, _ = curve_fit(lambda t, A, B, C, D, E: A * t[:, 0]**B + C * t[:, 1]**D + E, x, y, p0=initial_guess, maxfev=10000)



simulated_difference_list_of_list = []
for model_size in model_sizes:
    simulated_difference_list = []
    for chinchilla in [1, 2, 4, 8]:
        simulated_difference_list.append(popt[0] * expected_params[model_size]**popt[1] + popt[2] * (chinchilla)**popt[3] + popt[4])
    simulated_difference_list_of_list.append(simulated_difference_list)
plt.xlabel("Simulated Difference")
plt.ylabel("Actual Difference")

# only show two numbers in total
plt.scatter(simulated_difference_list_of_list, difference_list_of_list, label=f"difference fit:\n {popt[0]:.2f} * D^{popt[1]:.2f} + {popt[2]:.2e} * C^{popt[3]:.2f} + {popt[4]:.2f}")
plt.plot(np.arange(-0.07, 0.01, 0.01), np.arange(-0.07, 0.01, 0.01), label='y = x')
plt.legend()
plt.savefig(f'optimizer_loss_scaling_muon_adamw_fit.png')
plt.close()







# # scaling_laws = {}

# # # fit a scaling law for optimizer in optimizers
# # for optimizer in ['nadamw', 'kron']:
# #     fitting_dict = {}
# #     for model_size in model_sizes:
# #         for chinchilla in [1, 2, 4, 8]:
# #             if (optimizer, model_size, chinchilla) in actual_list:
# #                 fitting_dict[(expected_params[model_size], expected_params[model_size] * 20 * chinchilla)] = actual_list[(optimizer, model_size, chinchilla)]["min_loss"]

# #     # fit a power law that is A * model_size^B  + C * data_size^D
# #     x = np.array(list(fitting_dict.keys()))
# #     # print(x)re
# #     y = np.array(list(fitting_dict.values()))
# #     print()
# #     # print(y)
# #     # fit a power law that is A * model_size^B  + C * data_size^D + E
# #     initial_guess = [1.0, -0.5, 1.0, -0.5, 1.0]  # A, B, C, D, E
# #     popt, _ = curve_fit(lambda t, A, B, C, D, E: A * t[:, 0]**B + C * t[:, 1]**D + E, x, y, p0=initial_guess, maxfev=10000)
# #     print(popt)
# #     # plot the fitting curve
# #     predicted_loss = popt[0] * x[:, 0]**popt[1] + popt[2] * x[:, 1]**popt[3] + popt[4]
# #     # plt.plot(x, y, 'o', label=optimizer)
# #     plt.plot(predicted_loss, y, 'o', label='Actual Data for ' + optimizer)
# #     plt.plot(np.arange(predicted_loss.min(), predicted_loss.max() + 0.01, 0.01), np.arange(predicted_loss.min(), predicted_loss.max() + 0.01, 0.01), label='y = x')
# #     plt.xlabel('Predicted Loss')
# #     plt.ylabel('Actual Loss')
# #     plt.legend()
# #     plt.savefig(f'optimizer_loss_scaling_fitting_{optimizer}.png')
# #     plt.close()
# #     # calculate regression error
# #     regression_error = np.sqrt(np.mean((predicted_loss - y)**2))
# #     print(f'Regression error for {optimizer}: {regression_error}')
# #     scaling_laws[optimizer] = popt
    
# # plt.figure(figsize=(10, 6))
# # # estimate the scaling law on a 1.2B model 
# # for optimizer in scaling_laws:
# #     model_size_in_params =  1207959552
# #     data_size_in_params_list = [model_size_in_params * 20 * chinchilla for chinchilla in [1, 2, 4, 8]]
# #     predicted_loss_list = [scaling_laws[optimizer][0] * data_size_in_params**scaling_laws[optimizer][1] + scaling_laws[optimizer][2] * data_size_in_params**scaling_laws[optimizer][3] + scaling_laws[optimizer][4] for data_size_in_params in data_size_in_params_list]
# #     plt.plot([1, 2, 4, 8], predicted_loss_list, label=optimizer)
# # plt.legend()
# # plt.xlabel('Chinchilla')
# # plt.ylabel('Loss')
# # plt.title('Estimated Scaling Law for 1.2B Model')
# # plt.savefig(f'optimizer_loss_scaling_fitting_1.2B.png')
# # plt.close()

# # print(scaling_laws)



# optimal hyperparameters for AdamW
optimizer_name = 'muon'
hyperparameters_dict = {}

for model_size in model_sizes:
    for chinchilla in [1, 2, 4, 8]:
        if (optimizer_name, model_size, chinchilla) in actual_list:
            hyperparameters_dict[(model_size, chinchilla)] = actual_list[(optimizer_name, model_size, chinchilla)]["best_config"]

keys = list(hyperparameters_dict[(model_sizes[0], 1)].keys())

# with open(f'hyperparameters_fit_{optimizer_name}.md', 'w') as f:
#     for key in keys:
#         # fit a power law that is A * model_size^B * chinchilla^C + D
#         x = [(expected_params[model_size], chinchilla) for model_size in model_sizes for chinchilla in [1, 2, 4, 8]]
#         y = [hyperparameters_dict[(model_size, chinchilla)][key] for model_size in model_sizes for chinchilla in [1, 2, 4, 8]]
#         # fit a power law and print error
#         if type(y[-1]) == float or type(y[-1]) == int:
#             print(key)
#             print(y)
#             if key == "muon_to_adam_lr":
#                 continue
#             baseline = np.mean(y[:-1])
#             popt, _ = curve_fit(lambda t, A, B, C, D: A * t[:, 0]**B * t[:, 1]**C + D, x[1:-1], y[1:-1], p0=[0.0, -0.5, -0.5, baseline], maxfev=80000)
#             # print error on the last point
#             predicted_loss = popt[0] * x[-1][0]**popt[1] * x[-1][1]**popt[2] + popt[3]
#             error = np.sqrt(np.mean((predicted_loss - y[-1])**2))
#             f.write(f"Relative error for {key}: {error / (y[-1] + 1e-6)}\n")


