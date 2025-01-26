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
parser.add_argument("--data1_name", type=str)
parser.add_argument("--data2_name", type=str)
parser.add_argument("--model_size", type=str, default="150m")
args = parser.parse_args()

data1_name, data2_name = args.data1_name, args.data2_name

wandb_key = {
    ("stack_dedup", "c4"): {
        "150m": "all-stage2-bruteforce",
        "600m": "python-c4-0.005-600m-allstage2-sweep",
    },
    ("stack_dedup", "stack_cpp"): {
        "150m": "python-cpp-0.005-allstage2-sweep",
    }
}[(data1_name, data2_name)][args.model_size]

def parse_run(run):
    run_dict = {}

    run_json_config = json.loads(run.json_config)
    if "linear" in run.id:
        run_dict["schedule_type"] = "linear"
        run_dict["decay_ratio"] = run_json_config["optimizer"]["value"]["decay"]
    else:
        run_dict["schedule_type"] = "cosine"
        run_dict["decay_ratio"] = None
    stage2_stack_portion = run_json_config["data"]["value"]["train_weights"]["stack_dedup"]
    run_dict["stage2_duration"] = 0.005 / stage2_stack_portion

    history = run.history(keys=[f"eval/{data1_name}/loss", f"eval/{data2_name}/loss"])
    run_dict[f"final_{data1_name}_loss"] = history[f"eval/{data1_name}/loss"].iloc[-1]
    run_dict[f"final_{data2_name}_loss"] = history[f"eval/{data2_name}/loss"].iloc[-1]
    return run_dict

if args.build_cache:
    run_list = []
    runs = wandb.Api().runs("stanford-mercury/suhas-curriculum")
    for run in tqdm(runs):
        if wandb_key in run.tags and "stage2" in run.tags:
            run_dict = parse_run(run)
            run_list.append(run_dict)
    pickle.dump(run_list, open(f"cache/{wandb_key}_run_list.pkl", "wb"))
else:
    run_list = pickle.load(open(f"cache/{wandb_key}_run_list.pkl", "rb"))

decay_ratio_color_map = {
    None: 'blue',
    0.0: 'red',
    0.01: 'orange',
    0.05: 'green',
    0.1: 'purple',
    0.2: 'brown',
}

plt.figure(figsize=(7, 5), dpi=600)
plt.grid(True, linestyle='--', alpha=0.4)
unique_decay_ratios = list(set([run['decay_ratio'] for run in run_list]))
unique_decay_ratios.sort(key=lambda x: x if x is not None else -1)
fit_params = {}

for idx, decay_ratio in enumerate(unique_decay_ratios):
    color = decay_ratio_color_map[decay_ratio]
    if decay_ratio is None:
        label = "Cosine"
    else:
        label = f"Linear, cooldown {decay_ratio}"
    runs_with_decay = [run for run in run_list if run['decay_ratio'] == decay_ratio]
    runs_with_decay.sort(key=lambda x: x['stage2_duration'])
    
    x = np.array([run['stage2_duration'] for run in runs_with_decay])
    y = np.array([run[f"final_{data1_name}_loss"] for run in runs_with_decay])
    x_log = np.log(x)

    # Fit a cubic polynomial
    z = np.polyfit(x_log, y, 4)
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
plt.title(f'Final {data1_name} loss vs Stage 2 duration (0.005, 0.995 {data2_name}, {args.model_size})')
plt.xticks([0.8, 0.4, 0.2, 0.1, 0.05, 0.025, 0.00625], 
           [0.8, 0.4, 0.2, 0.1, 0.05, 0.025, 0.00625])
if data1_name == "stack_dedup" and data2_name == "c4" and args.model_size == "150m":
    plt.ylim(ymax=3.6)
plt.xlim(xmin=0.02)
plt.xlim(xmax=0.85)
plt.plot([], [], '*', color='black', label='Minima from quartic fit', markersize=10)
plt.legend()
plt.tight_layout()
plt.savefig(f'/Users/Suhas/Desktop/SUHAS/Repos/marin/experiments/curriculum/plots/allstage2/allstage2_{data1_name}_{data2_name}_{args.model_size}_rare_loss_curve.png')
plt.close()

# make a new plot where for each training run, plot the final stack loss vs c4 loss. make each schedule type + decay ratio a different color.

plt.figure(figsize=(7, 5), dpi=600)

for idx, decay_ratio in enumerate(unique_decay_ratios):
    color = decay_ratio_color_map[decay_ratio]
    if decay_ratio is None:
        label = "Cosine"
    else:
        label = f"Linear, cooldown {decay_ratio}"
    runs_with_decay = [run for run in run_list if run['decay_ratio'] == decay_ratio]
    runs_with_decay.sort(key=lambda x: x['stage2_duration'])
    
    x = np.array([run[f"final_{data1_name}_loss"] for run in runs_with_decay])
    y = np.array([run[f"final_{data2_name}_loss"] for run in runs_with_decay])
    plt.plot(x, y, label=label, marker='o', color=color)

plt.xlabel(f'Final {data1_name} loss')
plt.ylabel(f'Final {data2_name} loss')
plt.title(f'Final {data1_name} loss vs Final {data2_name} loss (0.005, 0.995, {args.model_size})')
# plt.xlim(xmax=3.6)
# plt.ylim(ymax=3.85)
plt.legend()
plt.tight_layout()
plt.savefig(f'/Users/Suhas/Desktop/SUHAS/Repos/marin/experiments/curriculum/plots/allstage2/allstage2_{data1_name}_{data2_name}_{args.model_size}_loss_tradeoff.png')
plt.close()
