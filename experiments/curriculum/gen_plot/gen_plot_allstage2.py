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

    history = run.history(keys=["eval/stack_dedup/loss", "eval/c4/loss"])
    run_dict["final_stack_loss"] = history["eval/stack_dedup/loss"].iloc[-1]
    run_dict["final_c4_loss"] = history["eval/c4/loss"].iloc[-1]
    return run_dict

if args.build_cache:
    run_list = []
    runs = wandb.Api().runs("stanford-mercury/suhas-curriculum")
    for run in tqdm(runs):
        if "all-stage2-bruteforce" in run.tags and "stage2" in run.tags:
            run_dict = parse_run(run)
            run_list.append(run_dict)
    pickle.dump(run_list, open("cache/allstage2_run_list.pkl", "wb"))
else:
    run_list = pickle.load(open("cache/allstage2_run_list.pkl", "rb"))

plt.figure(figsize=(7, 5), dpi=600)
plt.grid(True, linestyle='--', alpha=0.4)
unique_decay_ratios = list(set([run['decay_ratio'] for run in run_list]))
unique_decay_ratios.sort(key=lambda x: x if x is not None else -1)
fit_params = {}

for idx, decay_ratio in enumerate(unique_decay_ratios):
    color = f'C{idx}'
    if decay_ratio is None:
        label = "Cosine"
    else:
        label = f"Linear, cooldown {decay_ratio}"
    runs_with_decay = [run for run in run_list if run['decay_ratio'] == decay_ratio]
    runs_with_decay.sort(key=lambda x: x['stage2_duration'])
    
    x = np.array([run['stage2_duration'] for run in runs_with_decay])
    y = np.array([run['final_stack_loss'] for run in runs_with_decay])
    x_log = np.log(x)

    # Fit a cubic polynomial
    z = np.polyfit(x_log, y, 5)
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
    
    plt.scatter(x, y, label=label, marker='o')

plt.xlabel('Stage 2 Duration')
plt.ylabel('Final Stack Loss')
plt.xscale('log')
plt.title('Final Stack Loss vs Stage 2 Duration (total code fraction = 0.005)')
plt.xticks([0.8, 0.4, 0.2, 0.1, 0.05, 0.025, 0.00625], 
           [0.8, 0.4, 0.2, 0.1, 0.05, 0.025, 0.00625])
plt.xlim(0.02, 1)  # Adjusted xlim to cover fit range
plt.ylim(ymax=3.6)
plt.plot([], [], '*', color='black', label='Minima from quintic fit', markersize=10)
plt.legend()
plt.tight_layout()
plt.savefig('/Users/Suhas/Desktop/SUHAS/Repos/marin/experiments/curriculum/plots/allstage2/allstage2_final_stack_loss_tradeoff.png')
plt.close()

# make a new plot where for each training run, plot the final stack loss vs c4 loss. make each schedule type + decay ratio a different color.

plt.figure(figsize=(7, 5), dpi=600)

for idx, decay_ratio in enumerate(unique_decay_ratios):
    color = f'C{idx}'
    if decay_ratio is None:
        label = "Cosine"
    else:
        label = f"Linear, cooldown {decay_ratio}"
    runs_with_decay = [run for run in run_list if run['decay_ratio'] == decay_ratio]
    runs_with_decay.sort(key=lambda x: x['stage2_duration'])
    
    x = np.array([run['final_stack_loss'] for run in runs_with_decay])
    y = np.array([run['final_c4_loss'] for run in runs_with_decay])
    plt.plot(x, y, label=label, marker='o', color=color)

plt.xlabel('Final Stack Loss')
plt.ylabel('Final C4 Loss')
plt.title('Final Stack Loss vs Final C4 Loss (total code fraction = 0.005)')
plt.xlim(xmax=3.6)
plt.ylim(ymax=3.85)
plt.legend()
plt.tight_layout()
plt.savefig('/Users/Suhas/Desktop/SUHAS/Repos/marin/experiments/curriculum/plots/allstage2/allstage2_final_stack_loss_vs_c4_loss.png')
plt.close()
