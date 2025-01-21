import matplotlib.pyplot as plt
import wandb
from pprint import pprint
import pandas as pd
from tqdm import tqdm
import argparse
import pickle

plt.rcParams.update({
    "font.family": "Palatino Linotype"
})

parser = argparse.ArgumentParser()
parser.add_argument("--build_cache", action="store_true")
args = parser.parse_args()

# retrieve all wandb runs
runs = wandb.Api().runs("stanford-mercury/suhas-curriculum")

def parse_run(run, run_of_interest):
    if not run.startswith("stack-dedup-c4-curriculum-3B-150m-") or not run_of_interest(run):
        return None
    run_flags = run[len("stack-dedup-c4-curriculum-3B-150m-"):].split("-")
    run_flags_dict = {}

    if 'v' in run_flags[-1] and run_flags[-1][-1].isdigit():
        run_flags_dict["version"] = int(run_flags[-1][-1])
        run_flags = run_flags[:-1]
    else:
        run_flags_dict["version"] = 0

    run_flags_dict["name"] = "-".join(run_flags)

    if run_flags[0] == "halfsched":
        run_flags_dict["schedule"] = "halfsched"
        run_flags = run_flags[1:]
    elif run_flags[0] == "varsched":
        run_flags_dict["schedule"] = f"varsched ({run_flags[1]})"
        run_flags = run_flags[2:]
    else:
        raise ValueError(f"Unknown schedule: {run_flags[0]}")
    
    if run_flags[-1] == "stage1":
        run_flags_dict["stage"] = "stage1"
        run_flags_dict["code_portion_stage1"] = float(run_flags[0])
    elif run_flags[-1] == "stage2":
        run_flags_dict["stage"] = "stage2"
        run_flags_dict["code_portion_stage1"] = float(run_flags[0])
        run_flags_dict["code_portion_stage2"] = float(run_flags[1])
    else:
        raise ValueError(f"Unknown stage: {run_flags[-1]}")
    
    return run_flags_dict

run_list = []

def runs_equivalent(run1, run2):
    return run1['name'] == run2['name']

if args.build_cache:
    for run in tqdm(runs):
        def run_of_interest(run_name):
            return 'halfsched' in run_name
        
        run_flags_dict = parse_run(run.name, run_of_interest)
        if run_flags_dict is None:
            continue

        print(run_flags_dict['name'])

        current_run_is_redundant = False
        for existing_flags_dict in run_list:
            if runs_equivalent(run_flags_dict, existing_flags_dict):
                if run_flags_dict['version'] > existing_flags_dict['version']:
                    # remove existing_flags_dict from run_list
                    run_list.remove(existing_flags_dict)
                    print("    Removed existing run ->", existing_flags_dict['name'])
                else:
                    current_run_is_redundant = True

        if not current_run_is_redundant:
            run_history = run.history(keys=["eval/stack_dedup/loss", "eval/c4/loss"])
            run_flags_dict["history"] = run_history
            run_list.append(run_flags_dict)
        else:
            print("    Skipping redundant run ->", run_flags_dict['name'])
            
    # Save the run list to cache
    pickle.dump(run_list, open("cache/halfsched_run_list.pkl", "wb"))
else:
    # Load from cache
    run_list = pickle.load(open("cache/halfsched_run_list.pkl", "rb"))

run_list_stage1 = [run for run in run_list if run['stage'] == 'stage1']
run_list_stage2 = [run for run in run_list if run['stage'] == 'stage2']

for run_stage2 in run_list_stage2:
    # find matching run for stage1
    for run_stage1 in run_list_stage1:
        if run_stage1['code_portion_stage1'] == run_stage2['code_portion_stage1']:
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
            
            # Sort by step to ensure proper ordering
            df_merged = df_merged.sort_values('_step').reset_index(drop=True)

            run_stage2["history_stage1"] = run_stage1['history'].copy()
            run_stage2['merged_history'] = df_merged

for total_fraction in [0.5, 0.05, 0.005]:
    run_list_stage2_filtered = [run for run in run_list_stage2 if abs(0.5 * run['code_portion_stage1'] + 0.5 * run['code_portion_stage2'] - total_fraction) < 0.001 and run['code_portion_stage2'] != 0.0]

    run_list_stage2_sorted = sorted(run_list_stage2_filtered, key=lambda x: x['code_portion_stage1'])

    run_stage1_100 = [run for run in run_list_stage1 if abs(run['code_portion_stage1'] - total_fraction * 1.0) < 0.0001][0]
    run_stage1_140 = [run for run in run_list_stage1 if abs(run['code_portion_stage1'] - total_fraction * 1.4) < 0.0001][0]
    run_stage1_180 = [run for run in run_list_stage1 if abs(run['code_portion_stage1'] - total_fraction * 1.8) < 0.0001][0]

    # Plot both stages

    plt.figure(figsize=(7, 5), dpi=600)
    plt.grid(True, linestyle='--', alpha=0.4)

    for run_stage2 in run_list_stage2_sorted:
        # Filter out step 0 and plot
        history_no_step0 = run_stage2['merged_history'][run_stage2['merged_history']['_step'] > 0]
        plt.plot(history_no_step0['_step'], history_no_step0['eval/stack_dedup/loss'], label=f'Stage one: {run_stage2["code_portion_stage1"]} code, Stage two: {run_stage2["code_portion_stage2"]} code')

    plt.xlabel('Step')
    plt.ylabel('Loss')
    # plt.xscale('log')
    # plt.yscale('log')
    plt.title(f'Two equal stages: Total Code Fraction = {total_fraction}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'/Users/Suhas/Desktop/SUHAS/Repos/marin/experiments/curriculum/plots/halfsched/halfsched_total_{total_fraction}_stack_dedup_loss.png')
    plt.close()


    # Plot only stage 2

    plt.figure(figsize=(7, 5), dpi=600)
    plt.grid(True, linestyle='--', alpha=0.4)
    for run_stage2 in run_list_stage2_sorted:
        plt.plot(run_stage2['history']['_step'], run_stage2['history']['eval/stack_dedup/loss'], label=f'Stage one: {run_stage2["code_portion_stage1"]} code, Stage two: {run_stage2["code_portion_stage2"]} code')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.xscale('log')
    plt.yscale('log')
    plt.title(f'Stage 2: Total Code Fraction = {total_fraction}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'/Users/Suhas/Desktop/SUHAS/Repos/marin/experiments/curriculum/plots/halfsched/halfsched_total_{total_fraction}_stage2_stack_dedup_loss.png')
    plt.close()

    # Plot final stack dedup loss tradeoff

    plt.figure(figsize=(7, 5), dpi=600)
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.plot([run['code_portion_stage2'] / (total_fraction * 2.0) for run in run_list_stage2_sorted], [run['merged_history']['eval/stack_dedup/loss'].iloc[-1] for run in run_list_stage2_sorted], color='blue', marker='o', label='Two equal phase curriculum')

    plt.axhline(y=run_stage1_100['history']['eval/stack_dedup/loss'].iloc[-1], color='red', linestyle='--', label='1.0x Total code (no curriculum)')
    plt.axhline(y=run_stage1_140['history']['eval/stack_dedup/loss'].iloc[-1], color='orange', linestyle='--', label='1.4x Total code (no curriculum)')
    plt.axhline(y=run_stage1_180['history']['eval/stack_dedup/loss'].iloc[-1], color='yellow', linestyle='--', label='1.8x Total code (no curriculum)')
    
    plt.xlabel('Code Portion Stage 2')
    plt.ylabel('Final Stack Dedup Loss')
    plt.yscale('log')
    plt.title(f'Two equal stages: Total Code Fraction = {total_fraction}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'/Users/Suhas/Desktop/SUHAS/Repos/marin/experiments/curriculum/plots/halfsched/halfsched_total_{total_fraction}_final_stack_dedup_tradeoff.png')
    plt.close()

    # Plot final c4 loss tradeoff

    plt.figure(figsize=(7, 5), dpi=600)
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.plot([run['code_portion_stage2'] / (total_fraction * 2.0) for run in run_list_stage2_sorted], [run['merged_history']['eval/c4/loss'].iloc[-1] for run in run_list_stage2_sorted], color='blue', marker='o', label='Two equal phase curriculum')
    plt.axhline(y=run_stage1_100['history']['eval/c4/loss'].iloc[-1], color='red', linestyle='--', label='1.0x Total code (no curriculum)')
    plt.axhline(y=run_stage1_140['history']['eval/c4/loss'].iloc[-1], color='orange', linestyle='--', label='1.4x Total code (no curriculum)')
    plt.axhline(y=run_stage1_180['history']['eval/c4/loss'].iloc[-1], color='yellow', linestyle='--', label='1.8x Total code (no curriculum)')
    plt.xlabel('Code Portion Stage 2')
    plt.ylabel('Final C4 Loss')
    plt.yscale('log')
    plt.title(f'Two equal stages: Total Code Fraction = {total_fraction}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'/Users/Suhas/Desktop/SUHAS/Repos/marin/experiments/curriculum/plots/halfsched/halfsched_total_{total_fraction}_final_c4_tradeoff.png')
    plt.close()
    