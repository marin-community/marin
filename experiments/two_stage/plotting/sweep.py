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

def value_to_pretty_name(value):
    if value in value_pretty_name_dict:
        return value_pretty_name_dict[value]
    return value

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

    lr_str = run_id.split(f"{run_dict['lr_schedule']}-")[-1][:5]
    lr_str = lr_str if lr_str[-1] != "-" else lr_str[:-1]
    run_dict["lr"] = float(lr_str)
    run_dict["lr_cooldown_duration"] = "na" if "-na-" in run_id else float(run_id.split(f"{run_dict['lr']}-")[-1][:4])
    run_dict["rare_data_epochs"] = int(run_id.split("x")[-1].split("-")[0])
    run_dict["model_name"] = run_id.split("-")[0]

    if "-w" in run_id:
        weight_decay_str = run_id.split("-w")[-1]
        if weight_decay_str == "":
            weight_decay_str = "16"
        run_dict["weight_decay"] = float(weight_decay_str) / 10.0

    run_history_loss_keys = [f"eval/{rare_data_name}/loss"]

    history_loss = run.history(keys=run_history_loss_keys)

    run_dict["loss_history"] = history_loss

    if f"eval/{rare_data_name}/loss" not in history_loss.columns:
        return None
    
    run_dict[f"final_{rare_data_name}_loss"] = history_loss[f"eval/{rare_data_name}/loss"].iloc[-1]

    return run_dict

def create_heatmap(run_list, x_axis_key, y_axis_key, rare_data_name, common_data_name, key):
    """
    Creates a heatmap visualization from a list of experimental runs.
    
    Args:
        run_list: List of run dictionaries containing experiment results
        x_axis_key: Key for x-axis values in run dictionaries
        y_axis_key: Key for y-axis values in run dictionaries
        rare_data_name: Name of the rare dataset
        common_data_name: Name of the common dataset
        key: String identifier for the experiment
    """

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
    plt.figure(figsize=(3, 5), dpi=600)
    plt.imshow(loss_grid, cmap=CUSTOM_CMAP, aspect='auto', interpolation='nearest')

    # Find the best (minimum) value coordinates
    best_idx = np.unravel_index(np.argmin(loss_grid), loss_grid.shape)
    rect = plt.Rectangle((best_idx[1] - 0.5, best_idx[0] - 0.5), 1, 1, fill=False, 
                        edgecolor=PURPLE, linewidth=2)
    plt.gca().add_patch(rect)

    # Move colorbar to bottom and make it horizontal
    plt.colorbar(label=f'{pretty_name_dict[rare_data_name]} Loss', 
                orientation='horizontal')

    # Add text annotations to cells
    for i in range(len(unique_x_axis_values)):
        for j in range(len(unique_y_axis_values)):
            if best_run_grid[i, j] is not None:
                plt.text(j, i, f'{loss_grid[i, j]:.2f}', 
                        ha='center', va='center', 
                        color='white' if loss_grid[i, j] > np.mean(loss_grid) else 'black')

    plt.xlabel(key_pretty_name_dict[y_axis_key])
    plt.ylabel(key_pretty_name_dict[x_axis_key])
    plt.title(f'{pretty_name_dict[rare_data_name]} Loss')

    # Set tick labels
    plt.xticks(range(len(unique_y_axis_values)), unique_y_axis_values)
    plt.yticks(range(len(unique_x_axis_values)), unique_x_axis_values)

    plt.tight_layout()
    output_path = f'plotting/plots/{key}_loss_heatmap.png'
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()

    # Print the best overall configuration
    best_i, best_j = np.unravel_index(np.argmin(loss_grid), loss_grid.shape)
    best_run = best_run_grid[best_i, best_j]
    print("\nBest overall configuration:")
    print(f"{key_pretty_name_dict[x_axis_key]}: {unique_x_axis_values[best_i]}")
    print(f"{key_pretty_name_dict[y_axis_key]}: {unique_y_axis_values[best_j]}")
    print(f"Final {pretty_name_dict[rare_data_name]} Loss: {loss_grid[best_i, best_j]:.4f}")
    print(f"Run ID: {best_run['run_id']}")

    return best_run

def create_scatter_plot(run_list, x_axis_key, y_axis_key, rare_data_name, common_data_name, key):
    """
    Creates a scatter plot visualization from a list of experimental runs.
    Points are colored by y_axis_key values, showing loss against x_axis values.
    For replay_ratio, uses log(1-x) transformation.
    Includes quadratic fits for each set of points.
    """
    def transform_x(x):
        if x_axis_key == "replay_ratio":
            return -np.log(1 - x)
        return x

    def get_tick_labels(x):
        return str(x)
    
    def fit_quadratic(x, y):
        coeffs = np.polyfit(x, y, 2)
        # Find minimum of quadratic: -b/(2a)
        min_x = -coeffs[1]/(2*coeffs[0])
        min_y = coeffs[0]*min_x**2 + coeffs[1]*min_x + coeffs[2]
        return coeffs, min_x, min_y
    
    unique_y_values = sorted(list(set([run[y_axis_key] for run in run_list])))
    color_positions = np.linspace(0, 1, len(unique_y_values))
    colors = [CUSTOM_CMAP(pos) for pos in color_positions]
    
    plt.figure(figsize=(6, 4), dpi=600)
    
    # Store handles for legend ordering
    scatter_handles = []
    min_points = []
    
    for y_val, color in zip(unique_y_values, colors):
        matching_runs = [run for run in run_list if run[y_axis_key] == y_val]
        x_values = [run[x_axis_key] for run in matching_runs]
        x_values_transformed = [transform_x(x) for x in x_values]
        losses = [run[f'final_{rare_data_name}_loss'] for run in matching_runs]
        
        # First plot quadratic fit
        coeffs, min_x, min_y = fit_quadratic(x_values_transformed, losses)
        x_fit = np.linspace(min(x_values_transformed), max(x_values_transformed), 100)
        y_fit = coeffs[0]*x_fit**2 + coeffs[1]*x_fit + coeffs[2]
        plt.plot(x_fit, y_fit, '--k', alpha=0.5)
        
        # Then plot minimum point
        x_range = max(x_values_transformed) - min(x_values_transformed)
        if (min_x >= min(x_values_transformed) - 0.2*x_range and 
            min_x <= max(x_values_transformed) + 0.2*x_range):
            min_point = plt.scatter(min_x, min_y, color=color, marker='*', s=200, zorder=10)
            min_points.append(min_point)

        # Finally plot the actual points on top
        scatter = plt.scatter(x_values_transformed, losses, color=color, 
                            label=f'{value_to_pretty_name(y_val)} {key_pretty_name_dict[y_axis_key]}',
                            zorder=10)  # Higher zorder to ensure points are on top
        scatter_handles.append(scatter)
    
    # Add the fit line to legend (single entry)
    fit_line = plt.plot([], [], '--k', alpha=0.5, label='Quadratic Fit')[0]
    
    # Add the minimum points to legend (single entry)
    min_point_legend = plt.scatter([], [], color='k', marker='*', s=200, alpha=0.7, 
                                 label='Predicted Minima')
    
    # Order legend: scatter points, quadratic fit, minima
    handles = scatter_handles + [fit_line, min_point_legend]
    labels = [h.get_label() for h in handles]
    plt.legend(handles=handles, labels=labels, bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Find and mark the best actual point
    best_run = min(run_list, key=lambda x: x[f'final_{rare_data_name}_loss'])
    
    plt.xlabel(f'{key_pretty_name_dict[x_axis_key]}')
    plt.ylabel(f'{pretty_name_dict[rare_data_name]} Loss')
    plt.title(f'{pretty_name_dict[rare_data_name]} Loss vs {key_pretty_name_dict[x_axis_key]} and {key_pretty_name_dict[y_axis_key]}')
    
    unique_x_values = sorted(list(set([run[x_axis_key] for run in run_list])))
    x_ticks = [transform_x(x) for x in unique_x_values]
    plt.xticks(x_ticks, [get_tick_labels(x) for x in unique_x_values])
    
    output_path = f'plotting/plots/{key}_loss_scatter.png'
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    
    print("\nBest configuration from scatter analysis:")
    print(f"{key_pretty_name_dict[x_axis_key]}: {best_run[x_axis_key]}")
    print(f"{key_pretty_name_dict[y_axis_key]}: {best_run[y_axis_key]}")
    print(f"Final {pretty_name_dict[rare_data_name]} Loss: {best_run[f'final_{rare_data_name}_loss']:.4f}")
    print(f"Run ID: {best_run['run_id']}")

def create_simple_scatter(run_list, x_axis_key, rare_data_name, common_data_name, key, fit=True, baseline=None, title_stub="", ylims=None, show_best=False):
    """
    Creates a simple scatter plot with x_axis_key on x-axis and loss on y-axis.
    Only keeps the best (lowest loss) run for each x-axis value.
    """
    # Get unique x values and find best run for each
    unique_x_values = sorted(list(set([run[x_axis_key] for run in run_list])))
    best_runs = []
    for x in unique_x_values:
        matching_runs = [run for run in run_list if run[x_axis_key] == x]
        best_run = min(matching_runs, key=lambda r: r[f'final_{rare_data_name}_loss'])
        best_runs.append(best_run)
    
    run_list = best_runs

    def transform_x(x):
        if x_axis_key == "replay_ratio":
            return -np.log(1 - x)
        elif x_axis_key == "lr_cooldown_duration":
            x = np.clip(x, 0.01, 0.99)
            return np.log(x)
        elif x_axis_key == "rare_data_epochs":
            return np.log(x)
        elif x_axis_key == "weight_decay":
            return np.log(x + 0.05)
        return x

    def get_tick_labels(x):
        return str(x)
    
    plt.figure(figsize=(5, 3), dpi=300)
    
    x_values = [run[x_axis_key] for run in run_list]
    x_values_transformed = [transform_x(x) for x in x_values]
    losses = [run[f'final_{rare_data_name}_loss'] for run in run_list]
    
    # Plot the actual points first
    scatter = plt.scatter(x_values_transformed, losses, color=LIGHT_BLUE, zorder=5)
    
    # Plot quadratic fit
    if fit:
        coeffs = np.polyfit(x_values_transformed, losses, 2)
        x_fit = np.linspace(min(x_values_transformed), max(x_values_transformed), 100)
        y_fit = coeffs[0]*x_fit**2 + coeffs[1]*x_fit + coeffs[2]
        fit_line = plt.plot(x_fit, y_fit, '--k', alpha=0.5, label='Quadratic Fit')[0]
    else:
        plt.plot(x_values_transformed, losses, color=LIGHT_BLUE, zorder=5)
        fit_line = None

    if baseline:
        baseline_point = plt.scatter([transform_x(1.0)], [baseline], color=PURPLE, zorder=10)
        plt.axhline(y=baseline, color='black', linestyle=':', alpha=0.7)
    else:
        baseline_point = None

    if show_best:
        # Add star at the best point
        best_idx = np.argmin(losses)
        plt.scatter(x_values_transformed[best_idx], losses[best_idx], 
                   marker='*', color=LIGHT_BLUE, s=200, zorder=6)
        
        # Print stats for best point
        print(f"\nBest point:")
        print(f"{key_pretty_name_dict[x_axis_key]}: {x_values[best_idx]}")
        print(f"Loss: {losses[best_idx]:.4f}")
        print(f"Run ID: {run_list[best_idx]['run_id']}")
    
    plt.xlabel(f'{key_pretty_name_dict[x_axis_key]}')
    plt.ylabel(f'{pretty_name_dict[rare_data_name]} Loss')
    plt.title(f'{pretty_name_dict[rare_data_name]} Loss vs {key_pretty_name_dict[x_axis_key]}{title_stub}')
    
    unique_x_values = sorted(list(set(x_values)))
    x_ticks = [transform_x(x) for x in unique_x_values]
    plt.xticks(x_ticks, [get_tick_labels(x) for x in unique_x_values])

    if ylims:
        plt.ylim(ylims)
    
    # Order legend with data points first
    # print(fit, baseline)
    # if fit or baseline:
    #     plt.legend([scatter, fit_line, baseline_point], ['Data Points', 'Quadratic Fit', 'Uniform Ordering'], 
    #             bbox_to_anchor=(1.05, 1), loc='upper left')
    
    file_stub = title_stub.replace(" ", "_").replace("\\", "_").replace("%", "")
    output_path = f'plotting/plots/{key}_loss_simple{file_stub}.png'
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    
    print("\nBest configuration from simple scatter analysis:")
    print(f"{key_pretty_name_dict[x_axis_key]}: {best_run[x_axis_key]}")
    print(f"Final {pretty_name_dict[rare_data_name]} Loss: {best_run[f'final_{rare_data_name}_loss']:.4f}")
    print(f"Run ID: {best_run['run_id']}")

def create_double_scatter(run_lists, x_axis_key, rare_data_name, common_data_name, key, labels, ylims=None):
    """
    Creates a scatter plot with two series of data points.
    
    Args:
        run_lists: List of two run lists, each containing run dictionaries
        x_axis_key: Key for x-axis values
        rare_data_name: Name of rare dataset
        common_data_name: Name of common dataset
        key: String identifier for the experiment
        labels: List of two labels for the legend
        ylims: Optional tuple of (ymin, ymax) for setting y-axis limits
    """
    def transform_x(x):
        if x_axis_key == "replay_ratio":
            return -np.log(1 - x)
        elif x_axis_key == "lr_cooldown_duration":
            x = np.clip(x, 0.01, 0.99)
            return np.log(x)
        elif x_axis_key == "rare_data_epochs":
            return np.log(x)
        elif x_axis_key == "weight_decay":
            return np.log(x + 0.05)
        return x

    def get_tick_labels(x):
        return str(x)
    
    plt.figure(figsize=(5, 3), dpi=300)
    
    colors = [LIGHT_BLUE, PURPLE]  # Use our custom colors for the two series
    scatter_handles = []
    
    for runs, label, color in zip(run_lists, labels, colors):
        # Get unique x values and find best run for each
        unique_x_values = sorted(list(set([run[x_axis_key] for run in runs])))
        best_runs = []
        for x in unique_x_values:
            matching_runs = [run for run in runs if run[x_axis_key] == x]
            best_run = min(matching_runs, key=lambda r: r[f'final_{rare_data_name}_loss'])
            best_runs.append(best_run)
        
        x_values = [run[x_axis_key] for run in best_runs]
        x_values_transformed = [transform_x(x) for x in x_values]
        losses = [run[f'final_{rare_data_name}_loss'] for run in best_runs]
        
        # Plot points and line
        scatter = plt.scatter(x_values_transformed, losses, color=color, label=label, zorder=5)
        plt.plot(x_values_transformed, losses, color=color, alpha=0.5, zorder=4)
        scatter_handles.append(scatter)

        # Add star at the best point
        best_idx = np.argmin(losses)
        plt.scatter(x_values_transformed[best_idx], losses[best_idx], 
                   marker='*', color=color, s=200, zorder=6)
        
        # Print stats for best point
        print(f"\nBest point for {label}:")
        print(f"{key_pretty_name_dict[x_axis_key]}: {x_values[best_idx]}")
        print(f"Loss: {losses[best_idx]:.4f}")
        print(f"Run ID: {best_runs[best_idx]['run_id']}")
    
    plt.xlabel(f'{key_pretty_name_dict[x_axis_key]}')
    plt.ylabel(f'{pretty_name_dict[rare_data_name]} Loss')
    plt.title(f'{pretty_name_dict[rare_data_name]} Loss vs {key_pretty_name_dict[x_axis_key]}')
    
    # Set x-axis ticks using all runs to get complete range
    all_runs = run_lists[0] + run_lists[1]
    unique_x_values = sorted(list(set([run[x_axis_key] for run in all_runs])))
    x_ticks = [transform_x(x) for x in unique_x_values]
    plt.xticks(x_ticks, [get_tick_labels(x) for x in unique_x_values])

    if ylims:
        plt.ylim(ylims)
    
    plt.legend(handles=scatter_handles, bbox_to_anchor=(1.05, 1), loc='upper left')
    
    output_path = f'plotting/plots/{key}_loss_double.png'
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()

# Main execution
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rare_data_name", type=str, required=True)
    parser.add_argument("--common_data_name", type=str, default="c4")
    parser.add_argument("--build_cache", action="store_true")
    parser.add_argument("--x_axis_key", type=str, default="replay_ratio")
    parser.add_argument("--y_axis_key", type=str, default="stage2_allocation")
    parser.add_argument("--mode", type=str, default="schedule")
    args = parser.parse_args()

    rare_data_name = args.rare_data_name
    common_data_name = args.common_data_name

    x_axis_key = args.x_axis_key
    y_axis_key = args.y_axis_key

    key = {
        "repetition": f"{rare_data_name}-{common_data_name}-finding-repetitions",
        "repetition_v2": f"{rare_data_name}-{common_data_name}-finding-repetitions-0.1",
        "repetition_ft": f"{rare_data_name}-{common_data_name}-fine-tuning-epochs-v3",
        "schedule": f"{rare_data_name}-{common_data_name}-repetition-trial-v9",
        "schedule_v2": f"{rare_data_name}-{common_data_name}-repetition-trial-v10",
        "schedule_v3": f"{rare_data_name}-{common_data_name}-repetition-trial-v11",
        "sft": f"{rare_data_name}-{common_data_name}-fine-tuning-v5",
        "model": f"{rare_data_name}-{common_data_name}-model-scaling",
        "lr_schedule": f"{rare_data_name}-{common_data_name}-finding-lr-schedule",
        "lr_schedule_v2": f"{rare_data_name}-{common_data_name}-finding-lr-schedule-v2",
        "weight_decay": f"{rare_data_name}-{common_data_name}-finding-weight-decay",
    }[args.mode]

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

    run_list = sorted(run_list, key=lambda x: x[x_axis_key])

    print("Total runs: ", len(run_list))
    if x_axis_key == "rare_data_epochs":
        create_simple_scatter(run_list, x_axis_key, rare_data_name, common_data_name, key, fit=False)
    elif x_axis_key == "lr_cooldown_duration":
        # baseline_run = [run for run in run_list if run["replay_ratio"] != 0.0 and run["lr_cooldown_duration"] == 0.99 and run["lr"] == 3e-3][0]
        run_list = [run for run in run_list if run["replay_ratio"] == 0.0 and run["lr_cooldown_duration"] != 1.0 and run["lr"] == 3e-3]
        create_simple_scatter(run_list, x_axis_key, rare_data_name, common_data_name, key, fit=False)
    elif args.mode == "sft":
        run_list = [run for run in run_list if run["lr"] == 3e-4 and run["rare_data_epochs"] == 64 and run["replay_ratio"] < 0.8]
        baseline_run = [run for run in run_list if run["replay_ratio"] == 0.0][0]
        assert x_axis_key == "replay_ratio"
        create_simple_scatter(run_list, x_axis_key, rare_data_name, common_data_name, key, fit=False, baseline=baseline_run[f"final_{rare_data_name}_loss"], show_best=True)
    elif args.mode == "weight_decay":
        assert x_axis_key == "weight_decay"
        run_list = [run for run in run_list if run["rare_data_epochs"] == 32]
        print(run_list)
        create_simple_scatter(run_list, x_axis_key, rare_data_name, common_data_name, key, fit=False)
    else:
        run_list = [run for run in run_list if run["replay_ratio"] < 0.9]
        create_heatmap(run_list, x_axis_key, y_axis_key, rare_data_name, common_data_name, key)
        create_scatter_plot(run_list, x_axis_key, y_axis_key, rare_data_name, common_data_name, key)

        if args.mode == "schedule_v3" and rare_data_name == "starcoder":
            run_list_no_early = [run for run in run_list if run["stage2_allocation"] == 1.0]
            run_list_most_early = [run for run in run_list if run["stage2_allocation"] == 0.25]
            create_double_scatter([run_list_no_early, run_list_most_early], 
                                 "replay_ratio", rare_data_name, common_data_name,
                                 key, ["All at end", "25% at end"], ylims=(2.9, 3.8))


    
