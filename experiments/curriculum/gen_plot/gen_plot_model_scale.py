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
parser.add_argument("--build_cache", action="store_true")
parser.add_argument("--scaling_type", type=str, default="model")
args = parser.parse_args()

assert args.scaling_type in ["model", "token", "chinchilla"]

data1_name, data2_name = "flan", "c4"

pretty_name_dict = {
    "stack_dedup": "Python",
    "stack_cpp": "C++",
    "c4": "C4",
    "flan": "Flan",
}

wandb_key = {
    "model": "flan-c4-eu-model-scaling",
    "token": "flan-c4-eu-token-scaling",
    "chinchilla": "flan-c4-eu-chinchilla-model-scaling"
}[args.scaling_type]

param_counts = {
    "150m": 150,
    "300m": 300,
    "600m": 600,
    "600m_0.003": 6000000,
    "1_9b": 1900,
    "1_9b_0.003": 1900000000,
    "8b_1024": 8000,
}

def parse_run(run):
    run_dict = {}
    run_id = run.id

    run_json_config = json.loads(run.json_config)
    if "linear" in run_id:
        run_dict["schedule_type"] = "linear"
        run_dict["decay_ratio"] = run_json_config["optimizer"]["value"]["decay"]
    else:
        run_dict["schedule_type"] = "cosine"
        run_dict["decay_ratio"] = None

    run_dict["step_count"] = run_json_config["trainer"]["value"]["num_train_steps"]

    run_id_entries = run_id.split("-")
    # run_dict["fraction_data1_stage2"] = float(run_id_entries[run_id_entries.index("vs") + 1])

    run_dict["model_size"] = run_id_entries[run_id_entries.index("euw4") + 1]
    
    if run_dict["model_size"] == "linear":
        return None
    
    if args.scaling_type == "model":
        if run_dict["model_size"] == "600m" and "1e-3" not in run_id and "0.001" not in run_id:
            if "0.0003" in run_id:
                return None
            else:
                run_dict["model_size"] = "600m_0.003"
    elif args.scaling_type == "chinchilla":
        if run_dict["model_size"] == "600m" and "0.0001" in run_id:
            return None
    elif args.scaling_type == "token":
        if run_dict["model_size"] == "600m":
            if run_dict["step_count"] == 3000 and "0.001" not in run_id:
                return None

    if run_dict["model_size"] == "1_9b" and "3e-4" not in run_id and "0.0003" not in run_id:
        # run_dict["model_size"] = "1_9b_0.003"
        return None
    

    if run_dict["model_size"] == "8b_1024" and "0.0003" not in run_id:
        return None

    run_id_dur_entry = [entry for entry in run_id_entries if entry.startswith("dur")][0]
    run_dict["stage2_duration"] = float(run_id_dur_entry[3:])

    run_dict["param_count"] = param_counts[run_dict["model_size"]]

    history = run.history(keys=[f"eval/{data1_name}/loss", f"eval/{data2_name}/loss"])

    if len(history[f"eval/{data1_name}/loss"]) != 21:
        return None
    
    run_dict[f"final_{data1_name}_loss"] = history[f"eval/{data1_name}/loss"].iloc[-1]
    run_dict[f"final_{data2_name}_loss"] = history[f"eval/{data2_name}/loss"].iloc[-1]
    return run_dict

def run_eq(run1, run2):
    return run1[f"model_size"] == run2[f"model_size"] and run1[f"stage2_duration"] == run2[f"stage2_duration"] and run1["schedule_type"] == run2["schedule_type"] and run1["decay_ratio"] == run2["decay_ratio"] and run1["step_count"] == run2["step_count"]

if args.build_cache:
    run_list = []
    runs = wandb.Api().runs("stanford-mercury/suhas-curriculum")
    for run in tqdm(runs):
        if wandb_key in run.tags and "euw4" in run.tags:
            try:
                run_dict = parse_run(run)
                if run_dict is not None:
                    prev_len = len(run_list)
                    run_list = [_run_dict for _run_dict in run_list if not run_eq(_run_dict, run_dict)]
                    if len(run_list) != prev_len:
                        print("\tDuplicate run found and removed")
                    print("\033[94m" + run.id + "\033[0m")
                    run_list.append(run_dict)
                else:
                    print("\t^ Rejected above run", run.id)
            except Exception as e:
                print("\t^ Rejected above run", run.id, "because of", e)
    pickle.dump(run_list, open(f"cache/{wandb_key}_run_list.pkl", "wb"))
else:
    run_list = pickle.load(open(f"cache/{wandb_key}_run_list.pkl", "rb"))

model_pretty_name_dict = {
    "150m": "150M params",
    "300m": "300M params",
    "600m": "600M params",
    "600m_0.003": "600M params (suboptimal lr)",
    "1_9b": "1.9B params",
    "1_9b_0.003": "1.9B params (suboptimal lr)",
    "8b_1024": "8B params",
}

token_pretty_name_dict = {
    3000: "3B tokens",
    6000: "6B tokens",
    12000: "12B tokens",
    24000: "24B tokens",
}

model_color_map = {
    "150m": 'C1',
    "300m": 'C2',
    "600m": 'C3',
    "600m_0.003": 'C4',
    "1_9b": 'C5',
    "8b_1024": 'C6',
}

token_color_map = {
    3000: 'C1',
    6000: 'C2',
    12000: 'C3',
    24000: 'C4',
}

attribute_key = "step_count" if args.scaling_type == "token" else "model_size"
color_map = token_color_map if args.scaling_type == "token" else model_color_map
attribute_pretty_name_dict = token_pretty_name_dict if args.scaling_type == "token" else model_pretty_name_dict
sort_lambda = lambda x: x if args.scaling_type == "token" else param_counts[x]

plt.figure(figsize=(7, 5), dpi=600)
plt.grid(True, linestyle='--', alpha=0.4)
unique_attribute_values = list(set([run[attribute_key] for run in run_list]))
unique_attribute_values.sort(key=sort_lambda)
fit_params = {}

print('-'*100)
print(run_list)
print('-'*100)

for idx, attribute_value in enumerate(unique_attribute_values):
    if args.scaling_type == "chinchilla" and attribute_value == "1_9b":
        print(f"Skipping {attribute_value} because not enough data")
        continue
    print(color_map)
    color = color_map[attribute_value]
    label = f"{attribute_pretty_name_dict[attribute_value]}"
    runs_with_decay = [run for run in run_list if run[attribute_key] == attribute_value]
    runs_with_decay.sort(key=lambda x: x['stage2_duration'])

    # print successive differences
    print(f"Successive differences for {label}:")
    for i in range(len(runs_with_decay) - 1):
        print(runs_with_decay[i+1][f'final_{data1_name}_loss'] - runs_with_decay[i][f'final_{data1_name}_loss'])
    
    x = np.array([run['stage2_duration'] for run in runs_with_decay])
    y = np.array([run[f"final_{data1_name}_loss"] for run in runs_with_decay])
    x_log = np.log(x)

    # Fit a cubic polynomial
    z = np.polyfit(x_log, y, 2)
    p = np.poly1d(z)
    
    # Generate smooth points for plotting
    x_smooth = np.logspace(np.log10(x.min()), np.log10(x.max()), 100)
    x_smooth_log = np.log(x_smooth)
    y_smooth = p(x_smooth_log)
    # Find minimum of the fitted curve
    min_idx = np.argmin(y_smooth)
    min_x = x_smooth[min_idx]
    min_y = y_smooth[min_idx]
    print(f"Minimum for {label}: x={min_x:.4f}, y={min_y:.4f}")
    
    # Plot star at minimum
    plt.plot(min_x, min_y, '*', markersize=10, color=color)
    plt.plot(x_smooth, y_smooth, '--', alpha=0.5, color=color)
    
    plt.scatter(x, y, label=label, marker='o', color=color)

plt.xlabel('Stage 2 Duration')
plt.ylabel(f'Final {data1_name} loss')
plt.xscale('log')
# plt.yscale('log')
plt.title(f'Scaling {args.scaling_type.capitalize()}: rare loss vs stage 2 duration\n\n{pretty_name_dict[data1_name]} (0.005) vs {pretty_name_dict[data2_name]} (0.995) with 0.05 learning rate decay ratio')
plt.xticks([0.8, 0.4, 0.2, 0.1, 0.05, 0.025, 0.00625], 
           [0.8, 0.4, 0.2, 0.1, 0.05, 0.025, 0.00625])
# plt.xlim(xmin=0.02)
# plt.xlim(xmax=0.85)
plt.plot([], [], '*', color='black', label='Minima from quadratic fit', markersize=10)
plt.legend()
plt.tight_layout()
plt.savefig(f'/Users/Suhas/Desktop/SUHAS/Repos/marin/experiments/curriculum/plots/scaling/{data1_name}_{data2_name}_{args.scaling_type}_rare_loss_curve.png')
plt.close()
