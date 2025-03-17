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

    if run_json_config["optimizer"]["value"]["learning_rate"] != 3e-3:
        return None

    if "linear" in run.id:
        run_dict["schedule_type"] = "linear"
        run_dict["decay_ratio"] = run_json_config["optimizer"]["value"]["decay"]
    else:
        run_dict["schedule_type"] = "cosine"
        run_dict["decay_ratio"] = None

    history = run.history(keys=["eval/stack_dedup/loss", "eval/c4/loss"])
    run_dict["final_stack_loss"] = history["eval/stack_dedup/loss"].iloc[-1]
    run_dict["final_c4_loss"] = history["eval/c4/loss"].iloc[-1]
    return run_dict

if args.build_cache:
    run_list = []
    runs = wandb.Api().runs("stanford-mercury/suhas-curriculum")
    for run in tqdm(runs):
        if "baseline-lr-sweep" in run.tags:
            run_dict = parse_run(run)
            if run_dict is not None:
                run_list.append(run_dict)
    pickle.dump(run_list, open("cache/baseline_lr_sweep_run_list.pkl", "wb"))
else:
    run_list = pickle.load(open("cache/baseline_lr_sweep_run_list.pkl", "rb"))

plt.figure(figsize=(7, 5), dpi=600)
plt.grid(True, linestyle='--', alpha=0.4)
run_list.sort(key=lambda x: x['decay_ratio'] if x['decay_ratio'] is not None else -1)
fit_params = {}

color = 'purple'

x = np.array([run['decay_ratio'] for run in run_list if run['decay_ratio'] is not None])
y = np.array([run['final_stack_loss'] for run in run_list if run['decay_ratio'] is not None])

# # Fit a cubic polynomial
# z = np.polyfit(x, y, 4)
# p = np.poly1d(z)

# # Generate smooth points for plotting
# x_smooth = np.linspace(x.min(), x.max(), 100)
# y_smooth = p(x_smooth)
# # Find minimum of the fitted curve
# min_idx = np.argmin(y_smooth)
# min_x = x_smooth[min_idx]
# min_y = y_smooth[min_idx]
# print(f"Minimum: x={min_x:.4f}, y={min_y:.4f}")

# # Plot star at minimum
# plt.plot(min_x, min_y, '*', markersize=10, color=color)
# plt.plot(x_smooth, y_smooth, '--', alpha=0.5, color=color)

plt.plot(x, y, label='Linear schedule, varying cooldown ratio', marker='o', color=color)

cosine_runs = [run for run in run_list if run['schedule_type'] == 'cosine']
assert len(cosine_runs) == 1
assert cosine_runs[0]['decay_ratio'] == None
plt.axhline(y=cosine_runs[0]['final_stack_loss'], color='black', linestyle='--', alpha=0.5, label='Cosine schedule')

plt.xlabel('Decay Ratio')
plt.ylabel('Final Stack Loss')
# plt.xscale('log')
plt.title('Final Stack Loss vs Stage 2 Duration (total code fraction = 0.005)')
# plt.xticks([0.8, 0.4, 0.2, 0.1, 0.05, 0.025, 0.00625], 
#            [0.8, 0.4, 0.2, 0.1, 0.05, 0.025, 0.00625])
# plt.xlim(0.02, 1)  # Adjusted xlim to cover fit range
# plt.ylim(ymax=3.6)
plt.legend()
plt.tight_layout()
plt.savefig('/Users/Suhas/Desktop/SUHAS/Repos/marin/experiments/curriculum/plots/baseline_cooldown_sweep/baseline_cooldown_sweep_final_stack_loss_tradeoff.png')
plt.close()