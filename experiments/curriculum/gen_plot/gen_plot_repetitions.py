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
args = parser.parse_args()

data1_name, data2_name = "flan", "c4"

pretty_name_dict = {
    "stack_dedup": "Python",
    "stack_cpp": "C++",
    "c4": "C4",
    "flan": "Flan",
}

wandb_key = "flan-c4-repetition-all-stage2"

param_counts = {
    "150m": 150,
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

    if "euw4" in run_id:
        run_dict["region"] = "euw4"
    elif "usc2" in run_id:
        run_dict["region"] = "usc2"
    else:
        raise ValueError(f"Unknown region: {run_id}")

    run_id_entries = run_id.split("-")
    # run_dict["fraction_data1_stage2"] = float(run_id_entries[run_id_entries.index("vs") + 1])

    run_dict["model_size"] = run_id_entries[run_id_entries.index(run_dict["region"]) + 1]
    
    if run_dict["model_size"] == "linear":
        return None
    
    if run_dict["model_size"] != "150m":
        return None

    

    run_id_dur_entry = [entry for entry in run_id_entries if entry.startswith("dur")][0]
    run_dict["stage2_duration"] = float(run_id_dur_entry[3:])

    run_dict["param_count"] = param_counts[run_dict["model_size"]]

    run_dict["repetition_count"] = 1 if not run_id_entries[1].startswith("r") else int(run_id_entries[1][1:])

    history = run.history(keys=[f"eval/{data1_name}/loss", f"eval/{data2_name}/loss"])

    if f"eval/{data1_name}/loss" not in history or len(history[f"eval/{data1_name}/loss"]) != 21:
        return None
    
    run_dict[f"final_{data1_name}_loss"] = history[f"eval/{data1_name}/loss"].iloc[-1]
    run_dict[f"final_{data2_name}_loss"] = history[f"eval/{data2_name}/loss"].iloc[-1]
    return run_dict

def run_eq(run1, run2):
    return run1[f"model_size"] == run2[f"model_size"] and run1[f"stage2_duration"] == run2[f"stage2_duration"] and run1["schedule_type"] == run2["schedule_type"] and run1["decay_ratio"] == run2["decay_ratio"] and run1["step_count"] == run2["step_count"] and run1["repetition_count"] == run2["repetition_count"]

if args.build_cache:
    run_list = []
    runs = wandb.Api().runs("stanford-mercury/suhas-curriculum")
    for run in tqdm(runs):
        if wandb_key in run.tags:
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
    pickle.dump(run_list, open(f"cache/{wandb_key}_run_list.pkl", "wb"))
else:
    run_list = pickle.load(open(f"cache/{wandb_key}_run_list.pkl", "rb"))

model_pretty_name_dict = {
    "150m": "150M params",
    # "300m": "300M params",
    # "600m": "600M params",
    # "600m_0.003": "600M params (suboptimal lr)",
    # # "1_9b": "1.9B params",
    # # "1_9b_0.003": "1.9B params (suboptimal lr)",
    # "1_9b_1024": "1.9B params",
    # "8b_1024": "8B params",
}

token_pretty_name_dict = {
    3000: "3B tokens",
    6000: "6B tokens",
    12000: "12B tokens",
    24000: "24B tokens",
}

repetition_pretty_name_dict = {
    1: "1 repetition",
    2: "2 repetitions",
    4: "4 repetitions",
    6: "6 repetitions",
    8: "8 repetitions",
    16: "16 repetitions",
}

model_color_map = {
    "150m": 'C1',
    "300m": 'C2',
    "600m": 'C3',
    "600m_0.003": 'C4',
    # "1_9b": 'C5',
    "1_9b_1024": 'C5',
    "8b_1024": 'C6',
}

token_color_map = {
    3000: 'C7',
    6000: 'C8',
    12000: 'C9',
    24000: 'C10',
}

repetition_color_map = {
    1: 'C11',
    2: 'C12',
    4: 'C13',
    6: 'C14',
    8: 'C15',
    16: 'C16',
}

attribute_key = "repetition_count"
color_map = repetition_color_map
attribute_pretty_name_dict = repetition_pretty_name_dict

plt.figure(figsize=(7, 5), dpi=600)
plt.grid(True, linestyle='--', alpha=0.4)
unique_attribute_values = list(set([run[attribute_key] for run in run_list]))
unique_attribute_values.sort()
fit_params = {}

print('-'*100)
print(run_list)
print('-'*100)

for idx, attribute_value in enumerate(unique_attribute_values):
    if attribute_value == 16:
        continue
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
plt.title(f'Changing repetition count: Rare loss vs stage 2 duration\n\n150M params with 0.05 learning rate decay ratio')
plt.xticks([0.8, 0.4, 0.2, 0.1, 0.05, 0.025, 0.00625], 
           [0.8, 0.4, 0.2, 0.1, 0.05, 0.025, 0.00625])
# plt.xlim(xmin=0.02)
# plt.xlim(xmax=0.85)
plt.plot([], [], '*', color='black', label='Minima from quadratic fit', markersize=10)
plt.legend()
plt.tight_layout()
plt.savefig(f'/Users/Suhas/Desktop/SUHAS/Repos/marin/experiments/curriculum/plots/repetition/{data1_name}_{data2_name}_repetition_rare_loss_curve.png')
plt.close()
