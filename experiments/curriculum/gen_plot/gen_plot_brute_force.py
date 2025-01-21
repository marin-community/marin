import matplotlib.pyplot as plt
import wandb
from pprint import pprint
import pandas as pd
from tqdm import tqdm
import argparse
import pickle
from scipy.interpolate import griddata
import numpy as np

plt.rcParams.update({
    "font.family": "Palatino Linotype"
})

parser = argparse.ArgumentParser()
parser.add_argument("--build_cache", action="store_true")
args = parser.parse_args()

if args.build_cache:
    runs = wandb.Api().runs("stanford-mercury/suhas-curriculum")

    def parse_run(run):
        run_flags = run.split("-")

        # if any item ends in "e", merge it right the item to the right
        for i in range(len(run_flags)):
            if run_flags[i][-1] == "e":
                run_flags[i] = run_flags[i] + run_flags[i+1]
                run_flags.pop(i+1)
                break


        run_flags_dict = {}
        run_flags_dict["name"] = run
        
        if run_flags[-1] == "stage1":
            run_flags_dict["stage"] = "stage1"
            run_flags_dict["portion_code_stage2"] = float(run_flags[-3])
            run_flags_dict["code_weight_stage1"] = float(run_flags[-2])
        elif run_flags[-1] == "stage2":
            run_flags_dict["stage"] = "stage2"
            run_flags_dict["portion_code_stage2"] = float(run_flags[-4])
            run_flags_dict["code_weight_stage1"] = float(run_flags[-3])
            run_flags_dict["code_weight_stage2"] = float(run_flags[-2])
        else:
            raise ValueError(f"Unknown stage: {run_flags[-1]}")
        
        return run_flags_dict

    run_list = []
    for run in runs:
        if "two-stage-bruteforce" not in run.tags:
            continue
        
        run_dict = parse_run(run.id)
        if run_dict["stage"] == "stage1":
            run_dict["code_weight_stage2"] = run_dict["code_weight_stage1"]

        run_history = run.history(keys=["eval/stack_dedup/loss", "eval/c4/loss"])
        run_dict["history"] = run_history

        if "_step" not in run_dict["history"] or run_dict["history"]["_step"].iloc[-1] != 2999:
            continue

        print(run.id)
        run_list.append(run_dict)
    pickle.dump(run_list, open("cache/brute_force_run_list.pkl", "wb"))
else:
    run_list = pickle.load(open("cache/brute_force_run_list.pkl", "rb"))   

run_list_stage1 = [run for run in run_list if run['stage'] == 'stage1']
run_list_stage2 = [run for run in run_list if run['stage'] == 'stage2']

print("Number of stage1 runs:", len(run_list_stage1))
print("Number of stage2 runs:", len(run_list_stage2))

for run_stage2 in run_list_stage2:
    # find matching run for stage1
    found_match = False
    for run_stage1 in run_list_stage1:
        if run_stage1['code_weight_stage1'] == run_stage2['code_weight_stage1']:
            if found_match:
                raise ValueError("Multiple matches found")
            # Get the dataframes
            df_merged = run_stage1['history'].copy()
            df_stage2 = run_stage2['history']

            # For each row in stage2, update or append to stage1
            for idx, row in df_stage2.iterrows():
                step = row['_step']
                # Update existing step or append new row
                mask = df_merged['_step'] == step
                if mask.any():
                    # Update all columns for the matching row
                    for col in df_merged.columns:
                        df_merged.loc[mask, col] = row[col]
                else:
                    df_merged = pd.concat([df_merged, pd.DataFrame([row])], ignore_index=True)
            
            run_stage2['merged_history'] = df_merged
            found_match = True
            break
    if not found_match:
        print(f"No matching run found for {run_stage2['name']}")

plt.figure(figsize=(7, 5), dpi=600)
plt.grid(True, linestyle='--', alpha=0.4)

for run_stage2 in run_list_stage2:
    duration_frac_stage2 = round(0.005 * run_stage2['portion_code_stage2'] / run_stage2['code_weight_stage2'], 7)
    run_stage2['duration_frac_stage2'] = duration_frac_stage2

run_list_stage2_sorted = sorted(run_list_stage2, key=lambda x: x['code_weight_stage2'])

for portion_code_stage2 in [0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99]:
    print(f"Portion code stage 2: {portion_code_stage2}")
    for run_stage2 in run_list_stage2_sorted:
        if run_stage2['portion_code_stage2'] == portion_code_stage2:
            print(run_stage2['duration_frac_stage2'], '->', run_stage2['merged_history']['eval/stack_dedup/loss'].iloc[-1])
            # print(run_stage2['code_weight_stage1'], '->', run_stage2['merged_history']['eval/stack_dedup/loss'].iloc[-1])
    print('-' * 100)

for run_stage2 in run_list_stage2_sorted:
    # Filter out step 0 and plot
    history_no_step0 = run_stage2['merged_history'][run_stage2['merged_history']['_step'] > 0]
    plt.plot(history_no_step0['_step'], history_no_step0['eval/stack_dedup/loss'], label=f'Stage one: {run_stage2["code_weight_stage1"]} code, Stage two: {run_stage2["code_weight_stage2"]} code')

plt.xlabel('Step')
plt.ylabel('Loss')
plt.title(f'Two equal stages: Total code 15M tokens')
plt.legend()
plt.tight_layout()
plt.savefig(f'/Users/Suhas/Desktop/SUHAS/Repos/marin/experiments/curriculum/plots/brute_force/brute_force_15M_stack_dedup_loss.png')
plt.close()

# Create heatmap
plt.figure(figsize=(10, 8), dpi=600)

# Extract data points
x_vals = np.array([run['portion_code_stage2'] for run in run_list_stage2])
y_vals = np.array([run['duration_frac_stage2'] for run in run_list_stage2])
z_vals = np.array([run['history']['eval/stack_dedup/loss'].iloc[-1] for run in run_list_stage2])

# Create scatter plot with adjusted color settings
scatter = plt.scatter(x_vals, y_vals, c=z_vals, 
                     cmap='RdYlBu_r',  # Changed colormap to Red-Yellow-Blue (reversed)
                     s=100, 
                     vmin=z_vals.min(),  # Set minimum value
                     vmax=min(3.2, z_vals.max()))  # Set maximum value
plt.colorbar(scatter, label='Final Stack Dedup Loss')

# Add labels and title
plt.xlabel('Portion Code Stage 2')
plt.ylabel('Duration Fraction Stage 2')
plt.yscale('log')
plt.title('Heatmap of Training Parameters vs Loss (Loss ≤ 3.2)')

plt.grid(True, linestyle='--', alpha=0.4)
plt.tight_layout()
plt.savefig('/Users/Suhas/Desktop/SUHAS/Repos/marin/experiments/curriculum/plots/brute_force/brute_force_heatmap.png')
plt.close()

# Plot tradeoff between stack dedup loss (x-axis) and c4 loss (y-axis)
plt.figure(figsize=(7, 5), dpi=600)
plt.grid(True, linestyle='--', alpha=0.4)

# Extract final losses for each run
stack_dedup_losses = [run['merged_history']['eval/stack_dedup/loss'].iloc[-1] for run in run_list_stage2]
c4_losses = [run['merged_history']['eval/c4/loss'].iloc[-1] for run in run_list_stage2]

# Create scatter plot with different colors based on portion_code_stage2
unique_portions = sorted(set(run['portion_code_stage2'] for run in run_list_stage2))
unique_portions.remove(1.0)
for portion in unique_portions:
    mask = [run['portion_code_stage2'] == portion for run in run_list_stage2]
    plt.scatter(
        [l for l, m in zip(stack_dedup_losses, mask) if m],
        [l for l, m in zip(c4_losses, mask) if m],
        label=f'Portion code stage 2: {portion}',
        alpha=0.7
    )

plt.xlabel('Stack Dedup Loss')
plt.ylabel('C4 Loss')
plt.title('Tradeoff between Stack Dedup and C4 Loss')
plt.legend()
plt.tight_layout()
plt.savefig('/Users/Suhas/Desktop/SUHAS/Repos/marin/experiments/curriculum/plots/brute_force/brute_force_loss_tradeoff.png')
plt.close()

# For each unique portion, plot the final stack dedup loss vs code weight stage 2
plt.figure(figsize=(7, 5), dpi=600)
plt.grid(True, linestyle='--', alpha=0.4)
for portion in unique_portions:
    runs_portion = [run for run in run_list_stage2 if run['portion_code_stage2'] == portion]
    runs_portion_sorted = sorted(runs_portion, key=lambda x: x['code_weight_stage2'])
    plt.scatter(
        [run['duration_frac_stage2'] for run in runs_portion_sorted],
        [run['merged_history']['eval/stack_dedup/loss'].iloc[-1] for run in runs_portion_sorted],
        alpha=0.7
    )
    plt.plot(
        [run['duration_frac_stage2'] for run in runs_portion_sorted],
        [run['merged_history']['eval/stack_dedup/loss'].iloc[-1] for run in runs_portion_sorted],
        label=f'Portion code stage 2: {portion}',
        alpha=0.7
    )

plt.xlabel('Code Weight Stage 2')
plt.ylabel('Stack Dedup Loss')
plt.xscale('log')
plt.ylim(2.9, 3.4)
plt.title(f'Stack Dedup Loss vs Code Weight Stage 2 for Portion Code Stage 2: {portion}')
plt.legend()
plt.tight_layout()
plt.savefig(f'/Users/Suhas/Desktop/SUHAS/Repos/marin/experiments/curriculum/plots/brute_force/brute_force_multiple_scatter.png')
plt.close()




