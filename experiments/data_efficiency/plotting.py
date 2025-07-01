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
import matplotlib.colors as mcolors

plt.rcParams.update({
    "font.family": "Palatino Linotype"
})

# Custom color scheme
LIGHT_BLUE = '#8CD9FF'
PURPLE = '#7030A0'
CUSTOM_CMAP = mcolors.LinearSegmentedColormap.from_list('custom', [LIGHT_BLUE, PURPLE])

pretty_name_dict = {
    "finemath": "FineMath",
    "starcoder": "StarCoder",
    "flan": "Flan",
    "spj": "SlimPajama",
    "c4": "C4",
}

key_pretty_name_dict = {
    "rare_data_epochs": "Rare Repetitions",
    "replay_ratio": "Replay Fraction $\\rho$",
    "stage2_allocation": "Stage 2 Allocation $\\alpha$",
    "model_name": "Parameter Count",
    "lr_cooldown_duration": "LR Cooldown Duration",
    "weight_decay": "Weight Decay",
}

value_pretty_name_dict = {
    "150m4k": "150M",
    "300m4k": "300M",
    "600m4k": "600M",
    "1_9b4k": "1.9B",
}

def format_sci(val):
    return f"{val:.0e}".replace("e-0", "e-").replace("e+0", "e+")

def value_to_pretty_name(value):
    if value in value_pretty_name_dict:
        return value_pretty_name_dict[value]
    return value

def parse_run(run):
    if key not in run.tags or run.state != "finished":
        return None

    run_dict = {}
    run_id = run.id
    run_dict["run_id"] = run_id
    run_json_config = json.loads(run.json_config)

    num_steps = run_json_config["trainer"]["value"]["num_train_steps"]
    batch_size = run_json_config["trainer"]["value"]["train_batch_size"]
    seq_len = run_json_config["model"]["value"]["seq_len"]

    if run_id.startswith("ppl-eval-ensemble-"):
        run_id = run_id[len("ppl-eval-ensemble-"):]
        run_dict["ensemble_member_count"] = int(run_id.split("x-")[0])
        run_id = run_id.split("x-")[1]

    assert run_id.count("x") == 1

    run_dict["model_name"] = run_id.split("-")[0]
    run_dict["epochs"] = int(run_id.split("-")[1].split("x")[1])
    run_dict["base_tokens"] = num_steps * batch_size * seq_len / run_dict["epochs"]
    run_dict["data_name"] = run_id.split("-")[2]
    run_dict["lr_schedule"] = run_id.split("-")[3]
    run_dict["lr"] = float(run_id.split("-")[4][2:])
    run_dict["weight_decay"] = float(run_id.split("-")[5][2:])
    run_dict["batch_size"] = batch_size

    run_history_loss_keys = [f"eval/{run_dict['data_name']}/loss"]

    history_loss = run.history(keys=run_history_loss_keys)

    if f"eval/{run_dict['data_name']}/loss" not in history_loss.columns:
        return None
    
    run_dict["loss_history"] = history_loss
    run_dict[f"final_{run_dict['data_name']}_loss"] = history_loss[f"eval/{run_dict['data_name']}/loss"].iloc[-1]
    
    print(run_dict)

    return run_dict

def create_multi_restriction_scatter(run_lists, labels, key, title="Loss vs Base Tokens"):
    """
    Creates a figure with two subplots:
    1. Scatter plot with power law fits of the form A/x^B + C
    2. Bar plot showing the asymptotes (C values) from the power law fits
    
    Args:
        run_lists: List of run lists, each containing run dictionaries for a restriction
        labels: List of labels for the legend
        key: String identifier for the experiment
        title: Title for the plot
    """
    def power_law(x, A, B, C):
        return A / (x ** B) + C

    # Create figure with two subplots side by side, sharing y-axis
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8), gridspec_kw={'width_ratios': [2, 1]}, dpi=300, sharey=True)
    plt.title(title)
    
    colors = [LIGHT_BLUE, PURPLE, '#2ecc71', '#e74c3c']  # Add more colors for variety
    scatter_handles = []
    fit_handles = []
    asymptotes = []
    valid_labels = []
    all_x_values = set()  # Track all x values that will be plotted
    
    # First subplot: Scatter plot with power law fits
    for runs, label, color in zip(run_lists, labels, colors):
        # Get unique base tokens and find best run for each
        unique_base_tokens = sorted(list(set([run["base_tokens"] for run in runs])))[1:]

        best_runs = []
        for tokens in unique_base_tokens:
            matching_runs = [run for run in runs if run["base_tokens"] == tokens]
            best_run = min(matching_runs, key=lambda r: r['final_dclm_loss'])
            best_runs.append(best_run)
        
        # Print the runs that will be plotted
        print(f"\nPlotted runs for {label}:")
        for run in best_runs:
            print(f"{run['final_dclm_loss']:.4f}: {run['run_id']}")
        
        x_values = [run["base_tokens"] / 1_000_000 for run in best_runs]  # Convert to millions
        all_x_values.update(x_values)  # Add these x values to our set
        losses = [run['final_dclm_loss'] for run in best_runs]
        
        # Plot points and connecting line
        scatter = ax1.scatter(x_values, losses, color=color, label=f"{label} (data)", zorder=5)
        ax1.plot(x_values, losses, color=color, alpha=0.3, zorder=4)
        scatter_handles.append(scatter)

        # Fit power law
        try:
            popt, _ = curve_fit(power_law, x_values, losses, p0=[1.0, 0.5, 2.0], bounds=([0, 0, 0], [np.inf, np.inf, np.inf]))
            x_fit = np.linspace(min(x_values), max(x_values), 100)
            y_fit = power_law(x_fit, *popt)
            fit_line = ax1.plot(x_fit, y_fit, '--', color=color, label=f"{label} (fit: {popt[0]:.2f}/x^{popt[1]:.2f} + {popt[2]:.2f})", zorder=6)
            fit_handles.append(fit_line[0])
            
            # Store asymptote for bar plot
            asymptotes.append(popt[2])
            valid_labels.append(label)
        except RuntimeError as e:
            print(f"\nWarning: Could not fit power law for {label}")
            print(e)
    
    ax1.set_xlabel('Base Tokens (millions)')
    ax1.set_ylabel('DCLM Loss')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    
    # Set x-ticks only for the actual data points
    ax1.set_xticks(sorted(list(all_x_values)))
    ax1.set_xticklabels([f'{int(round(x))}' for x in sorted(list(all_x_values))])
    ax1.minorticks_off()  # Remove minor ticks
    
    # Second subplot: Bar plot of asymptotes
    x_pos = np.arange(len(asymptotes))
    ax2.bar(x_pos, asymptotes, color=colors[:len(asymptotes)])
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(valid_labels, rotation=45, ha='right')
    ax2.set_xlabel('Restriction Type')
    ax2.set_yscale('log')  # Use log scale to match main plot
    
    # Add value labels on top of bars
    for i, v in enumerate(asymptotes):
        ax2.text(i, v * 1.05, f'{v:.3f}', ha='center', va='bottom')
    
    # Combine scatter and fit handles/labels for legend
    all_handles = scatter_handles + fit_handles
    
    # First adjust the subplots to be closer together
    plt.subplots_adjust(right=0.75)
    
    # Then place the legend immediately after the adjusted subplots
    fig.legend(handles=all_handles, bbox_to_anchor=(0.78, 0.5), loc='center left')
    
    output_path = f'plots/{key}_multi_restriction_loss.png'
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()

def fit_inverse_curve(x, y):
    """Fits data to a curve of form y = A/x + B"""
    def inverse_func(x, A, B):
        return A/x + B
    
    popt, pcov = curve_fit(inverse_func, x, y)
    return popt[0], popt[1]  # A and B parameters

def create_heatmap(ax, all_fits, target_lr):
    """Helper function to create a heatmap for a specific learning rate"""
    lr_fits = [fit for fit in all_fits if fit[0][1] == target_lr]
    
    # Get unique epochs and weight decay values
    epochs = sorted(list(set(fit[0][0] for fit in lr_fits)))
    wds = sorted(list(set(fit[0][2] for fit in lr_fits)))
    
    # Create matrix for heatmap
    heatmap_data = np.zeros((len(epochs), len(wds)))
    for i, epoch in enumerate(epochs):
        for j, wd in enumerate(wds):
            matching_fits = [fit for fit in lr_fits if fit[0][0] == epoch and fit[0][2] == wd]
            if matching_fits:
                heatmap_data[i, j] = matching_fits[0][4]  # B value (asymptote)
    
    # Plot heatmap
    im = ax.imshow(heatmap_data, aspect='auto', cmap=CUSTOM_CMAP)
    
    # Add colorbar
    plt.colorbar(im, ax=ax, label='Asymptotic Loss')
    
    # Set ticks and labels
    ax.set_xticks(range(len(wds)))
    ax.set_yticks(range(len(epochs)))
    ax.set_xticklabels([f'{wd:.1f}' for wd in wds])
    ax.set_yticklabels(epochs)
    
    # Add labels
    ax.set_xlabel('Weight Decay')
    ax.set_ylabel('Epochs')
    ax.set_title(f'Asymptotic Loss Heatmap\n(lr={target_lr})')
    
    # Add text annotations with values
    for i in range(len(epochs)):
        for j in range(len(wds)):
            text = ax.text(j, i, f'{heatmap_data[i, j]:.3f}',
                          ha="center", va="center", color="black")
    
    return im

def plot_ensemble_scaling():
    # Create figure with three subplots: one large on left, two smaller stacked on right
    fig = plt.figure(figsize=(15, 7), dpi=600)
    gs = plt.GridSpec(2, 2, width_ratios=[2, 1], height_ratios=[1, 1])
    
    ax1 = fig.add_subplot(gs[:, 0])  # Main plot takes full height on left
    ax2 = fig.add_subplot(gs[0, 1])  # Top right
    ax3 = fig.add_subplot(gs[1, 1])  # Bottom right
    
    # Get the data from the last unique key's runs
    unique_keys = set()
    for run in run_list:
        unique_keys.add((run["epochs"], run["lr"], run["weight_decay"]))
    
    # Store fits and data for sorting
    all_fits = []
    for unique_key in unique_keys:
        runs = [run for run in run_list if run["epochs"] == unique_key[0] and run["lr"] == unique_key[1] and run["weight_decay"] == unique_key[2]]
        runs = sorted(runs, key=lambda x: x["ensemble_member_count"])
        x_data = np.array([run["ensemble_member_count"] for run in runs])
        y_data = np.array([run["final_dclm_loss"] for run in runs])
        
        # Fit 1/x curve
        A, B = fit_inverse_curve(x_data, y_data)
        all_fits.append((unique_key, x_data, y_data, A, B))
    
    # Sort by asymptote (B value)
    all_fits.sort(key=lambda x: x[4])
    
    # Plot in order of asymptote on first subplot
    for unique_key, x_data, y_data, A, B in all_fits:
        # Generate points for smooth curve
        x_fit = np.linspace(min(x_data), max(x_data), 100)
        y_fit = A/x_fit + B
        
        # Check if this is the specific configuration we want to highlight
        is_target_config = (unique_key[0] == 16 and unique_key[1] == 0.003 and unique_key[2] == 1.6)
        
        # Use different marker and color for the target configuration
        if is_target_config:
            ax1.scatter(x_data, y_data, marker='*', s=200)
            ax1.plot(x_fit, y_fit, '--', label=f'Fit (epochs={unique_key[0]}, lr={unique_key[1]}, wd={unique_key[2]}): {A:.3f}/x + {B:.3f}', linewidth=2)
        else:
            ax1.scatter(x_data, y_data)
            ax1.plot(x_fit, y_fit, '--', label=f'Fit (epochs={unique_key[0]}, lr={unique_key[1]}, wd={unique_key[2]}): {A:.3f}/x + {B:.3f}')
        
        print(f"\nFit parameters for configuration {unique_key}:")
        print(f"A (scaling factor) = {A:.3f}")
        print(f"B (asymptote) = {B:.3f}")
    
    ax1.set_xlabel('Ensemble Member Count')
    ax1.set_ylabel('DCLM Loss')
    ax1.set_title('Loss vs Ensemble Size with 1/x Fits')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True)

    # Create heatmaps for both learning rates
    create_heatmap(ax2, all_fits, 0.003)
    create_heatmap(ax3, all_fits, 0.001)
    
    plt.tight_layout()
    plt.savefig('plots/ensemble_scaling.png', bbox_inches='tight')
    plt.close()

# Main execution
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str)
    parser.add_argument("--build_cache", action="store_true")
    args = parser.parse_args()
    mode = args.mode

    key, project_name = {
        "data-scaling-laws-6-10": ("data-scaling-laws-6-10", "stanford-mercury/suhas-data-efficiency"),
        "varying-hparams-experiment": ("varying-hparams-experiment", "stanford-mercury/suhas-eval-data-efficiency"),
    }[args.mode]

    if args.build_cache:
        run_list = []
        runs = wandb.Api().runs(project_name)
        for run in tqdm(runs):
            run_dict = parse_run(run)
            if run_dict is not None:
                print(run_dict["run_id"])
                run_list.append(run_dict)
        pickle.dump(run_list, open(f"cache/{key}_run_list.pkl", "wb"))
    else:
        run_list = pickle.load(open(f"cache/{key}_run_list.pkl", "rb"))

    if key == "data-scaling-laws-6-10":
        def no_restriction(run):
            return True

        def zero_weight_decay_restriction(run):
            return run["weight_decay"] == 0.0
        
        def one_epoch_restriction(run):
            return run["epochs"] == 1 \
                # and run["lr"] == 3e-3
        
        def one_epoch_zero_weight_decay_restriction(run):
            return run["weight_decay"] == 0.0 \
                and run["epochs"] == 1 \
                # and run["lr"] == 3e-3

        # Get unique model sizes
        unique_models = sorted(list(set([run["model_name"] for run in run_list])))
        
        # Create separate plots for each model size
        for model_size in unique_models:
            print(f"\nCreating plot for model size: {model_size}")
            
            # Filter runs for this model size
            model_runs = [run for run in run_list if run["model_name"] == model_size]
            
            # Create lists of runs for each restriction
            no_restriction_runs = [run for run in model_runs if no_restriction(run)]
            zero_wd_runs = [run for run in model_runs if zero_weight_decay_restriction(run)]
            one_epoch_runs = [run for run in model_runs if one_epoch_restriction(run)]
            one_epoch_zero_wd_runs = [run for run in model_runs if one_epoch_zero_weight_decay_restriction(run)]

            # Create multi-restriction scatter plot
            run_lists = [no_restriction_runs, zero_wd_runs, one_epoch_runs, one_epoch_zero_wd_runs]
            labels = ["many epoch, yes wd", "many epoch, no wd", "one epoch, yes wd", "one epoch, no wd"]
            
            # Only include restrictions that have data
            valid_run_lists = []
            valid_labels = []
            for runs, label in zip(run_lists, labels):
                if len(runs) > 0:
                    valid_run_lists.append(runs)
                    valid_labels.append(label)
            
            if len(valid_run_lists) > 0:
                create_multi_restriction_scatter(
                    valid_run_lists, 
                    valid_labels, 
                    f"{key}_{model_size}", 
                    title=f"Loss vs Base Tokens for {value_pretty_name_dict.get(model_size, model_size)}"
                )
            else:
                print(f"No valid runs found for model size {model_size}")

    elif key == "varying-hparams-experiment":
        plot_ensemble_scaling()
        
        # Keep the original analysis code
        unique_keys = set()
        for run in run_list:
            unique_keys.add((run["epochs"], run["lr"], run["weight_decay"]))
        print(unique_keys)

        for unique_key in unique_keys:
            print(unique_key)
            runs = [run for run in run_list if run["epochs"] == unique_key[0] and run["lr"] == unique_key[1] and run["weight_decay"] == unique_key[2]]
            runs = sorted(runs, key=lambda x: x["ensemble_member_count"])
            ensemble_member_counts = [run["ensemble_member_count"] for run in runs]
            losses = [run["final_dclm_loss"] for run in runs]
            