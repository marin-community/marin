import matplotlib.pyplot as plt
import wandb
from pprint import pprint
import pandas as pd
from tqdm import tqdm
import argparse
import pickle
from scipy.interpolate import griddata
import numpy as np
import json
from scipy.optimize import curve_fit

plt.rcParams.update({
    "font.family": "Palatino Linotype"
})

parser = argparse.ArgumentParser()
parser.add_argument("--rare_data_name", type=str, required=True)
parser.add_argument("--common_data_name", type=str, default="c4")
parser.add_argument("--build_cache", action="store_true")
args = parser.parse_args()

rare_data_name = args.rare_data_name
common_data_name = args.common_data_name

x_axis_key = "rare_data_epochs"
y_axis_key = "replay_ratio"

# key = f"{rare_data_name}-{common_data_name}-lr-data-schedule"
key = f"{rare_data_name}-{common_data_name}-repetition-trial-v7"
pretty_name_dict = {
    "finemath": "FineMath",
    "starcoder": "StarCoder",
    "flan": "Flan",
    "spj": "SlimPajama",
    "c4": "C4",
}

key_pretty_name_dict = {
    "rare_data_epochs": "Rare Repetitions",
    "replay_ratio": "Replay Ratio",
    "stage2_allocation": "Stage 2 Allocation",
}
def parse_run(run):
    if key not in run.tags:
        return None

    run_dict = {}
    run_id = run.id
    run_dict["run_id"] = run_id
    assert run_id.count("x") == 2

    run_dict["replay_ratio"] = float(run_id.split("-rr")[-1][:4])
    run_dict["stage2_allocation"] = float(run_id.split("-rs")[-1][:4])
    run_dict["lr_schedule"] = "wsd" if "wsd" in run_id else "cos"
    run_dict["lr_cooldown_duration"] = float(run_id.split(f"{run_dict['lr_schedule']}-")[-1][:5])
    run_dict["rare_data_epochs"] = int(run_id.split("x")[-1].split("-")[0])

    run_history_loss_keys = [f"eval/{rare_data_name}/loss", f"eval/{common_data_name}/loss"]

    history_loss = run.history(keys=run_history_loss_keys)

    run_dict["loss_history"] = history_loss

    run_dict[f"final_{rare_data_name}_loss"] = history_loss[f"eval/{rare_data_name}/loss"].iloc[-1]

    return run_dict

if args.build_cache:
    run_list = []
    runs = wandb.Api().runs("stanford-mercury/suhas-two-stage")
    for run in tqdm(runs):
        run_dict = parse_run(run)
        if run_dict is not None:
            print(run_dict["run_id"])
            print(run_dict)
            run_list.append(run_dict)
    pickle.dump(run_list, open(f"cache/{key}_run_list.pkl", "wb"))
else:
    run_list = pickle.load(open(f"cache/{key}_run_list.pkl", "rb"))

print("Total runs: ", len(run_list))

unique_x_axis_values = list(set([run[x_axis_key] for run in run_list]))
unique_x_axis_values.sort()

unique_y_axis_values = list(set([run[y_axis_key] for run in run_list]))
unique_y_axis_values.sort()

# Create a grid for the heatmap
loss_grid = np.zeros((len(unique_x_axis_values), len(unique_y_axis_values)))
best_run_grid = np.full((len(unique_x_axis_values), len(unique_y_axis_values)), None, dtype=object)

# Fill the grid with best (lowest) loss values
for i, x in enumerate(unique_x_axis_values):
    for j, y in enumerate(unique_y_axis_values):
        matching_runs = [run for run in run_list 
                        if run[x_axis_key] == x 
                        and run[y_axis_key] == y]
        if matching_runs:
            best_run = min(matching_runs, key=lambda x: x[f'final_{rare_data_name}_loss'])
            loss_grid[i, j] = best_run[f'final_{rare_data_name}_loss']
            best_run_grid[i, j] = best_run

# Create heatmap plot
plt.figure(figsize=(10, 8), dpi=600)
plt.imshow(loss_grid, cmap='viridis', aspect='auto', interpolation='nearest')
plt.colorbar(label=f'{pretty_name_dict[rare_data_name]} Loss')

# Add text annotations to cells
for i in range(len(unique_x_axis_values)):
    for j in range(len(unique_y_axis_values)):
        if best_run_grid[i, j] is not None:
            plt.text(j, i, f'{loss_grid[i, j]:.3f}', 
                    ha='center', va='center', 
                    color='white' if loss_grid[i, j] > np.mean(loss_grid) else 'black')

plt.xlabel(key_pretty_name_dict[y_axis_key])
plt.ylabel(key_pretty_name_dict[x_axis_key])
plt.title(f'{pretty_name_dict[rare_data_name]} Loss Heatmap\nBest Run for Each Configuration')

# Set tick labels
plt.xticks(range(len(unique_y_axis_values)), unique_y_axis_values)
plt.yticks(range(len(unique_x_axis_values)), unique_x_axis_values)

plt.tight_layout()
plt.savefig(f'plotting/plots/{key}_loss_heatmap.png', bbox_inches='tight')
plt.close()

# Print the best overall configuration
best_i, best_j = np.unravel_index(np.argmin(loss_grid), loss_grid.shape)
best_run = best_run_grid[best_i, best_j]
print("\nBest overall configuration:")
print(f"{key_pretty_name_dict[x_axis_key]}: {unique_x_axis_values[best_i]}")
print(f"{key_pretty_name_dict[y_axis_key]}: {unique_y_axis_values[best_j]}")
print(f"Final {pretty_name_dict[rare_data_name]} Loss: {loss_grid[best_i, best_j]:.4f}")
print(f"Run ID: {best_run['run_id']}")




    
