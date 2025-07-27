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
from convex_certificate_scaling import get_bounding_box

plt.rcParams.update({
    "font.family": "Palatino Linotype"
})

# Custom color scheme
LIGHT_BLUE = '#8CD9FF'
PURPLE = '#7030A0'
GREEN = '#2ECC71'
CUSTOM_CMAP = mcolors.LinearSegmentedColormap.from_list('custom', [LIGHT_BLUE, PURPLE])
CUSTOM_CMAP.set_bad(color='white', alpha=0)

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
    "1_4b4k": "1.4B",
    "1_9b4k": "1.9B",
}

model_params = {
        "150m4k": 0.15,
        "300m4k": 0.3,
        "600m4k": 0.6,
        "1_4b4k": 1.4,
        "1_9b4k": 1.9
    }

def format_sci(val):
    return f"{val:.0e}".replace("e-0", "e-").replace("e+0", "e+")

def value_to_pretty_name(value):
    if value in value_pretty_name_dict:
        return value_pretty_name_dict[value]
    return value

def parse_run(run):
    if (key not in run.tags and key != "none") or "ignore" in run.tags:
        print(f"\033[91mSkipping run {run.id} because it does not have the key {key} or is ignored\033[0m")
        return None
    
    if run.state != "finished":
        print(f"\033[91mSkipping run {run.id} because it is not finished\033[0m")
        return None

    run_dict = {}
    run_id = run.id
    run_dict["run_id"] = run_id
    run_json_config = json.loads(run.json_config)

    num_steps = run_json_config["trainer"]["value"]["num_train_steps"]
    batch_size = run_json_config["trainer"]["value"]["train_batch_size"]
    seq_len = run_json_config["model"]["value"]["seq_len"]
    base_tokens = None

    if run_id.startswith("ppl-eval-ensemble-") or run_id.startswith("ss-"):
        if run_id.startswith("ppl-eval-ensemble-"):
            run_id = run_id[len("ppl-eval-ensemble-"):]
        elif run_id.startswith("ss-"):
            if key != "seed-science-7-9":
                print(f"\033[91mSkipping run {run_id} because it is not a seed science run\033[0m")
                return None
            
            run_id = run_id[len("ss-"):]

            if "ts0-ds0" in run_id:
                run_dict["seed_types"] = ["train", "data", "both"]
            elif "ts0" in run_id:
                run_dict["seed_types"] = ["data"]
            elif "ds0" in run_id:
                run_dict["seed_types"] = ["train"]
            else:
                run_dict["seed_types"] = ["both"]
        
        run_dict["ensemble_member_count"] = int(run_id.split("x-")[0])
        run_id = run_id.split("x-")[1]

        if "209M" in run_id:
            num_base_steps = 800
        elif "419M" in run_id:
            num_base_steps = 1600
        elif "838M" in run_id:
            num_base_steps = 3200
        elif "1.7B" in run_id:
            num_base_steps = 6400
        else:
            raise ValueError(f"Unknown token count: {run_id}")
        batch_size = 64
        seq_len = 4096
        base_tokens = num_base_steps * batch_size * seq_len

    if run_id.count("x") != 1:
        print(f"\033[91mSkipping run {run_id} because it does not have exactly one x\033[0m")
        return None
    
    if "Mx" not in run_id and "Bx" not in run_id:
        print(f"\033[91mSkipping run {run_id} because it uses incorrect naming convention\033[0m")
        return None

    run_dict["model_name"] = run_id.split("-")[0]
    run_dict["epochs"] = int(run_id.split("-")[1].split("x")[1])
    run_dict["base_tokens"] = base_tokens if base_tokens is not None else num_steps * batch_size * seq_len / run_dict["epochs"]
    run_dict["data_name"] = run_id.split("-")[2]
    run_dict["lr_schedule"] = run_id.split("-")[3]
    run_dict["lr"] = float(run_id.split("-")[4][2:])
    run_dict["weight_decay"] = float(run_id.split("-")[5][2:])
    run_dict["batch_size"] = batch_size

    run_history_loss_keys = [f"eval/{run_dict['data_name']}/loss"]

    history_loss = run.history(keys=run_history_loss_keys)

    if f"eval/{run_dict['data_name']}/loss" not in history_loss.columns:
        print(f"\033[91mSkipping run {run_id} because it does not have the loss history\033[0m")
        return None
    
    run_dict["loss_history"] = history_loss
    run_dict[f"final_{run_dict['data_name']}_loss"] = history_loss[f"eval/{run_dict['data_name']}/loss"].iloc[-1]
    
    # print(run_dict)

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

def fit_curve(x, y, form="reciprocal"):
    """Fits data to a curve of form y = A/x + B"""
    if form == "reciprocal":
        def func(x, A, B):
            return A/x + B
    elif form == "power_law":
        def func(x, A, B, C):
            return A/(x**B) + C
    else:
        raise ValueError(f"Invalid form: {form}")
    
    popt, pcov = curve_fit(func, x, y)
    A = popt[0]
    B = 1.0 if form == "reciprocal" else popt[1]
    C = popt[2]
    return A, B, C

def create_heatmap(ax, all_fits, target_lr, use_asymptote=True):
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
                if use_asymptote:
                    heatmap_data[i, j] = matching_fits[0][5]  # C value (asymptote)
                else:
                    # Get loss for ensemble size 1
                    x_data = matching_fits[0][1]  # ensemble sizes
                    y_data = matching_fits[0][2]  # losses
                    idx = np.where(x_data == 1)[0][0]
                    heatmap_data[i, j] = y_data[idx]
    
    heatmap_data = np.ma.masked_where(heatmap_data == 0, heatmap_data)
    # Find minimum value and its indices
    min_val = np.ma.min(heatmap_data)
    min_idx = np.where(heatmap_data == min_val)
    i, j = min_idx[0][0], min_idx[1][0]
    
    # Draw rectangle around minimum value
    rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, color=PURPLE, linewidth=2)
    ax.add_patch(rect)
    # Plot heatmap
    im = ax.imshow(heatmap_data, aspect='auto', cmap=CUSTOM_CMAP)
    
    # Add colorbar
    plt.colorbar(im, ax=ax, label='Loss')
    
    # Set ticks and labels
    ax.set_xticks(range(len(wds)))
    ax.set_yticks(range(len(epochs)))
    ax.set_xticklabels([f'{wd:.1f}' for wd in wds])
    ax.set_yticklabels(epochs)
    
    # Add labels
    ax.set_xlabel('Weight Decay')
    ax.set_ylabel('Epochs')
    metric = "Asymptotic" if use_asymptote else "Single Member"
    ax.set_title(f'{metric} Loss\n(lr={target_lr})')
    
    # Add text annotations with values
    for i in range(len(epochs)):
        for j in range(len(wds)):
            if np.ma.is_masked(heatmap_data[i, j]):
                data_str = "N/A"
            else:
                data_str = f'{heatmap_data[i, j]:.3f}'
            text = ax.text(j, i, data_str, ha="center", va="center", color="black")
    
    return im

def plot_ensemble_scaling(model_name, base_tokens):
    # First figure: 1/x fits
    plt.figure(figsize=(12, 7), dpi=600)

    def valid_run(run):
        return run["model_name"] == model_name and run["base_tokens"] == base_tokens

    filtered_run_list = [run for run in run_list if valid_run(run)]

    if len(filtered_run_list) == 0:
        print(f"No runs found for model size: {model_name} and base tokens: {base_tokens}")
        return
    else:
        print(f"Creating plot for model size: {model_name} and base tokens: {base_tokens}")

    
    # Get the data from the last unique key's runs
    unique_keys = set()
    for run in filtered_run_list:
        unique_keys.add((run["epochs"], run["lr"], run["weight_decay"]))

    print(unique_keys)
    
    # Store fits and data for sorting
    all_fits = []
    for unique_key in unique_keys:
        runs = [run for run in filtered_run_list if run["epochs"] == unique_key[0] and run["lr"] == unique_key[1] and run["weight_decay"] == unique_key[2]]
        runs = sorted(runs, key=lambda x: x["ensemble_member_count"])
        x_data = np.array([run["ensemble_member_count"] for run in runs])
        y_data = np.array([run["final_dclm_loss"] for run in runs])

        if len(x_data) <= 2:
            continue
        
        # Fit 1/x curve
        # A, B, C = fit_curve(x_data, y_data, form="reciprocal")
        A, B, C = fit_curve(x_data, y_data, form="power_law")
        all_fits.append((unique_key, x_data, y_data, A, B, C))
    
    # Sort by asymptote (B value)
    all_fits.sort(key=lambda x: x[5])
    
    # Plot in order of asymptote
    for unique_key, x_data, y_data, A, B, C in all_fits:
        # Generate points for smooth curve
        x_fit = np.linspace(min(x_data), max(x_data), 100)
        y_fit = A/(x_fit**B) + C
        
        plt.scatter(x_data, y_data)
        plt.plot(x_fit, y_fit, '--', label=f'Fit (epochs={unique_key[0]}, lr={unique_key[1]}, wd={unique_key[2]}): {A:.3f}/(x^{B:.3f}) + {C:.3f}')

    plt.xlabel('Ensemble Member Count')
    plt.ylabel('DCLM Loss')
    plt.title('Loss vs Ensemble Size with 1/x Fits')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'plots/ensemble_scaling_fits_{model_name}_{base_tokens}.png', bbox_inches='tight')
    plt.close()

    unique_learning_rates = list(set([fit[0][1] for fit in all_fits]))

    # Second figure: heatmaps
    fig, axs = plt.subplots(len(unique_learning_rates), 2, figsize=(12, 5 * len(unique_learning_rates)), dpi=600)

    if len(unique_learning_rates) == 1:
        create_heatmap(axs[0], all_fits, unique_learning_rates[0], use_asymptote=False)
        create_heatmap(axs[1], all_fits, unique_learning_rates[0], use_asymptote=True)
    else:
        for i, lr in enumerate(unique_learning_rates):
            create_heatmap(axs[i, 0], all_fits, lr, use_asymptote=False)
            create_heatmap(axs[i, 1], all_fits, lr, use_asymptote=True)
    
    plt.tight_layout()
    plt.savefig(f'plots/ensemble_scaling_heatmaps_{model_name}_{base_tokens}.png', bbox_inches='tight')
    plt.close()

    if all_fits[0][5] < 1.8:
        print("SKIPPING FIRST RUN SINCE IT'S LIKELY BUGGED")
        return all_fits[1]
    return all_fits[0]

def plot_model_scaling(best_run_dict, fit_type="power_law"):
    """Creates subplots showing loss vs model size for each token count with power law fits."""
    def power_law(x, A, B, C):
        return A / (x ** B) + C

    def reciprocal_law(x, A, C):
        return A / x + C

    # Get unique model sizes and token counts
    token_counts = sorted(best_run_dict.keys())
    model_sizes = sorted(list(best_run_dict[token_counts[0]].keys()), key=lambda x: model_params[x])
    
    # Store fitted parameters for reuse
    fitted_params = {}
    valid_token_counts = []
    asymptotes = []
    
    # Create figure with subplots stacked vertically with extra space for tables
    fig, axs = plt.subplots(len(token_counts), 1, figsize=(8, 6 * len(token_counts)), dpi=300)
    
    # If only one token count, wrap axes in list for consistent indexing
    if len(token_counts) == 1:
        axs = [axs]
    
    for i, token_count in enumerate(token_counts):
        # Get data for this token count
        x_values = np.array([model_params[model] for model in model_sizes])
        y_values = np.array([best_run_dict[token_count][model]["final_dclm_loss"] for model in model_sizes])
        certified_values = np.array([best_run_dict[token_count][model]["convex_certificate"] for model in model_sizes])
        
        # Fit power law
        try:
            if fit_type == "power_law":
                popt, _ = curve_fit(power_law, x_values, y_values, p0=[1.0, 0.5, 2.0], bounds=([0, 0, 0], [np.inf, np.inf, np.inf]))
            elif fit_type == "reciprocal_law":
                popt, _ = curve_fit(reciprocal_law, x_values, y_values, p0=[1.0, 2.0], bounds=([0, 0], [np.inf, np.inf]))
            fitted_params[token_count] = popt
            valid_token_counts.append(token_count)
            asymptotes.append(popt[-1])  # Store asymptote for reuse
            
            # Generate points for smooth curve
            x_fit = np.logspace(np.log10(min(x_values)), np.log10(max(x_values)), 100)
            y_fit = power_law(x_fit, *popt) if fit_type == "power_law" else reciprocal_law(x_fit, *popt)
            
            # Plot data points and fitted curve
            axs[i].scatter(x_values, y_values, color=PURPLE, zorder=5)
            axs[i].plot(x_fit, y_fit, '--', color=PURPLE, 
                       label=f'Fit: {popt[0]:.2f}/x^{popt[1]:.2f} + {popt[2]:.2f}' if fit_type == "power_law" else f'Fit: {popt[0]:.2f}/x + {popt[1]:.2f}', 
                       zorder=4)
            
            # Set scales and labels
            axs[i].set_xscale('log')
            axs[i].set_yscale('log')
            axs[i].set_xlabel('Model parameters')
            axs[i].set_ylabel('DCLM Loss')
            
            # Set x-ticks to model sizes
            axs[i].set_xticks(x_values)
            axs[i].set_xticklabels([value_pretty_name_dict.get(model, model) for model in model_sizes])
            
            # Add title for this subplot
            token_count_m = token_count / 1_000_000  # Convert to millions
            axs[i].set_title(f'Seed token count: {int(token_count_m)}M')
            
            # Add grid
            axs[i].grid(True, which='major', linestyle='--', alpha=0.7)
            
            # Add legend
            axs[i].legend(loc='upper right')
            
            # Add value labels
            for x, y in zip(x_values, y_values):
                axs[i].text(x * 1.05, y, f'{y:.3f}', verticalalignment='bottom')
            
            # Add green bounding boxes around certified points
            for j, (x, y, is_certified) in enumerate(zip(x_values, y_values, certified_values)):
                if is_certified:
                    axs[i].scatter(x, y, s=70, facecolors='none', edgecolors='green', 
                                 linewidth=2, marker='o', zorder=6)
            
            # Prepare table data
            hyperparams = ['Weight Decay', 'Learning Rate', 'Epochs']
            table_headers = ['Hyperparameter'] + [value_pretty_name_dict.get(model, model) for model in model_sizes]
            
            table_data = []
            for hparam in hyperparams:
                row = [hparam]
                for model in model_sizes:
                    run = best_run_dict[token_count][model]
                    if hparam == 'Weight Decay':
                        row.append(f"{run['weight_decay']:.1f}")
                    elif hparam == 'Learning Rate':
                        row.append(f"{run['lr']:.0e}")
                    elif hparam == 'Epochs':
                        row.append(f"{run['epochs']}")
                table_data.append(row)
            
            # Create table using automatic positioning
            table = axs[i].table(cellText=table_data,
                               colLabels=table_headers,
                               cellLoc='center',
                               loc='bottom',
                               bbox=[0.0, -0.5, 1.0, 0.3])
            
            # Style the table
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1, 1.0)
            
            # Color the header row
            for j in range(len(table_headers)):
                table[(0, j)].set_facecolor('#E6E6FA')
                table[(0, j)].set_text_props(weight='bold')
            
            # Color the first column (hyperparameter names)
            for row_idx in range(1, len(table_data) + 1):
                table[(row_idx, 0)].set_facecolor('#F0F0F0')
                table[(row_idx, 0)].set_text_props(weight='bold')
                
        except RuntimeError as e:
            print(f"\nWarning: Could not fit power law for token count {token_count}")
            print(e)
    
    # Automatic spacing adjustments
    plt.subplots_adjust(hspace=0.4)  # Reduced from fixed large spacing
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])  # Leave space for tables at bottom
    plt.savefig('plots/model_scaling.png', bbox_inches='tight')
    plt.close()

    """Creates a plot showing asymptotes vs token count using pre-fitted results."""
    
    # Create plot
    plt.figure(figsize=(8, 6), dpi=300)
    
    # Convert token counts to millions for better readability
    token_counts_m = [tc / 1_000_000 for tc in token_counts]
    
    # Fit a power law to the asymptotes vs token_counts_m
    popt, _ = curve_fit(power_law, token_counts_m, asymptotes, p0=[1.0, 0.5, 2.0], bounds=([0, 0, 0], [np.inf, np.inf, np.inf]))
    x_fit = np.logspace(np.log10(min(token_counts_m)), np.log10(max(token_counts_m)), 100)
    y_fit = power_law(x_fit, *popt)

    plt.scatter(token_counts_m, asymptotes, color=PURPLE, s=100, zorder=5)
    plt.plot(x_fit, y_fit, '--', color=PURPLE, zorder=4, label=f'Fit: {popt[0]:.2f}/x^{popt[1]:.2f} + {popt[2]:.2f}')

    # #### hacky
    # # heuristically selected hparams
    # ensemble_runs_manual = {
    #     # "150m4k": (token_counts_m[:2], [3.494941417720857, 3.399930198165533]),
    #     "300m4k": (token_counts_m, [3.2729813419904183, 3.1503966937866053, 3.1018990697199413, 3.0790376554299104]),
    #     "600m4k": (token_counts_m, [3.2122277662817145, 3.0519037696079496, 2.9058312693222317, 2.8262327566492838]),
    #     "1_4b4k": (token_counts_m, [3.173966150932168, 3.002520844567612, 2.8680140499504847, 2.752375695327966]),
    # }

    # # # same hparams
    # # ensemble_runs_manual = {
    # #     "150m4k": (token_counts_m[:2], [3.519, 3.435]),
    # #     "300m4k": (token_counts_m, [3.336, 3.196, 3.102, 3.079]),
    # #     "600m4k": (token_counts_m, [3.297, 3.098, 2.906, 2.826]),
    # #     "1_4b4k": (token_counts_m, [3.341, 3.091, 2.888, 2.800]),
    # # }

    # for model_size, (token_counts_m_current, losses) in ensemble_runs_manual.items():
    #     plt.scatter(token_counts_m_current, losses, s=100, zorder=5)
    #     popt, _ = curve_fit(power_law, token_counts_m_current, losses, p0=[1.0, 0.5, 2.0], bounds=([0, 0, 0], [np.inf, np.inf, np.inf]))
    #     x_fit = np.logspace(np.log10(min(token_counts_m_current)), np.log10(max(token_counts_m_current)), 100)
    #     y_fit = power_law(x_fit, *popt)
    #     plt.plot(x_fit, y_fit, '--', zorder=4, label=f'{model_size} $\infty$ ensembles: {popt[0]:.2f}/x^{popt[1]:.2f} + {popt[2]:.2f}')

    # ensemble_tiered_asymptotes = []
    # for i in range(len(token_counts_m)):
    #     model_sizes = []
    #     all_losses = []
    #     for model_size, (token_counts_m_current, losses) in ensemble_runs_manual.items():
    #         if i < len(token_counts_m_current):
    #             model_sizes.append(model_params[model_size])
    #             all_losses.append(losses[i])
    #     print(model_sizes, all_losses)
    #     popt, _ = curve_fit(power_law, model_sizes, all_losses, p0=[1.0, 0.5, 2.0], bounds=([0, 0, 0], [np.inf, np.inf, np.inf]))
    #     ensemble_tiered_asymptotes.append(popt[-1])

    # plt.scatter(token_counts_m, ensemble_tiered_asymptotes, color=LIGHT_BLUE, s=100, zorder=5)
    # popt, _ = curve_fit(power_law, token_counts_m, ensemble_tiered_asymptotes, p0=[1.0, 0.5, 2.0], bounds=([0, 0, 0], [np.inf, np.inf, np.inf]))
    # x_fit = np.logspace(np.log10(min(token_counts_m)), np.log10(max(token_counts_m)), 100)
    # y_fit = power_law(x_fit, *popt)
    # plt.plot(x_fit, y_fit, '--', color=LIGHT_BLUE, zorder=4, label=f'Tiered fit: {popt[0]:.2f}/x^{popt[1]:.2f} + {popt[2]:.2f}')
    # ### end hacky


    # Set labels and title
    plt.xscale('log')
    plt.xlabel('Seed token count (millions)')
    plt.ylabel('Asymptotic Loss')
    plt.title('Asymptotic Loss vs Training Token Count')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Add value labels
    for x, y in zip(token_counts_m, asymptotes):
        plt.text(x * 1.1, y, f'{y:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('plots/asymptotes_vs_tokens.png', bbox_inches='tight')
    plt.close()

def plot_seed_science(train_seed_losses, data_seed_losses, both_seed_losses):
    power_law = lambda x, A, B, C: A / (x ** B) + C
    
    plt.figure(figsize=(8, 6), dpi=300)
    
    # Fit power laws to each curve
    x_data = np.array(range(1, 6))
    
    # Train seed curve fit
    popt_train, _ = curve_fit(power_law, x_data, train_seed_losses, p0=[1.0, 0.5, 2.0], bounds=([0, 0, 0], [np.inf, np.inf, np.inf]))
    x_fit = np.linspace(1, 5, 100)
    y_fit_train = power_law(x_fit, *popt_train)
    
    # Data seed curve fit  
    popt_data, _ = curve_fit(power_law, x_data, data_seed_losses, p0=[1.0, 0.5, 2.0], bounds=([0, 0, 0], [np.inf, np.inf, np.inf]))
    y_fit_data = power_law(x_fit, *popt_data)
    
    # Both seeds curve fit
    popt_both, _ = curve_fit(power_law, x_data, both_seed_losses, p0=[1.0, 0.5, 2.0], bounds=([0, 0, 0], [np.inf, np.inf, np.inf]))
    y_fit_both = power_law(x_fit, *popt_both)

    # Plot fitted curves
    plt.plot(x_fit, y_fit_train, '--', color=PURPLE, zorder=4, 
             label=f'Train seed fit: {popt_train[0]:.2f}/x^{popt_train[1]:.2f} + {popt_train[2]:.2f}')
    plt.plot(x_fit, y_fit_data, '--', color=GREEN, zorder=4,
             label=f'Data seed fit: {popt_data[0]:.2f}/x^{popt_data[1]:.2f} + {popt_data[2]:.2f}')
    plt.plot(x_fit, y_fit_both, '--', color=LIGHT_BLUE, zorder=4,
             label=f'Both seeds fit: {popt_both[0]:.2f}/x^{popt_both[1]:.2f} + {popt_both[2]:.2f}')

    # Plot data points
    plt.scatter(x_data, train_seed_losses, color=PURPLE, s=100, zorder=5)
    plt.scatter(x_data, data_seed_losses, color=GREEN, s=100, zorder=5)  
    plt.scatter(x_data, both_seed_losses, color=LIGHT_BLUE, s=100, zorder=5)

    plt.xlabel('Ensemble member count')
    plt.ylabel('Loss')
    plt.title('Varying seed (150M parameters, 200M tokens, optimal single model hparams)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('plots/seed_science.png', bbox_inches='tight')
    plt.close()

def plot_token_scaling(best_run_dict, fit_type="power_law"):
    """Creates subplots showing loss vs token count for each model size with power law fits."""
    def power_law(x, A, B, C):
        return A / (x ** B) + C

    def reciprocal_law(x, A, C):
        return A / x + C

    # Get unique model sizes and token counts
    token_counts = sorted(best_run_dict.keys())
    model_sizes = sorted(list(best_run_dict[token_counts[0]].keys()), key=lambda x: model_params[x])
    
    # Store asymptotes for reuse
    asymptotes = []
    valid_model_sizes = []
    
    # Create figure with subplots stacked vertically with extra space for tables
    fig, axs = plt.subplots(len(model_sizes), 1, figsize=(8, 6 * len(model_sizes)), dpi=300)
    
    # If only one model size, wrap axes in list for consistent indexing
    if len(model_sizes) == 1:
        axs = [axs]
    
    for i, model_size in enumerate(model_sizes):
        # Get data for this model size across all token counts
        x_values = np.array([tc / 1_000_000 for tc in token_counts])  # Convert to millions
        y_values = np.array([best_run_dict[tc][model_size]["final_dclm_loss"] for tc in token_counts])
        certified_values = np.array([best_run_dict[tc][model_size]["convex_certificate"] for tc in token_counts])
        
        # Fit power law
        try:
            if fit_type == "power_law":
                popt, _ = curve_fit(power_law, x_values, y_values, p0=[1.0, 0.5, 2.0], bounds=([0, 0, 0], [np.inf, np.inf, np.inf]))
            elif fit_type == "reciprocal_law":
                popt, _ = curve_fit(reciprocal_law, x_values, y_values, p0=[1.0, 2.0], bounds=([0, 0], [np.inf, np.inf]))
            
            # Store asymptote for reuse
            asymptotes.append(popt[-1])  # Last parameter is the asymptote
            valid_model_sizes.append(model_size)
            
            # Generate points for smooth curve
            x_fit = np.logspace(np.log10(min(x_values)), np.log10(max(x_values)), 100)
            y_fit = power_law(x_fit, *popt) if fit_type == "power_law" else reciprocal_law(x_fit, *popt)
            
            # Plot data points and fitted curve
            axs[i].scatter(x_values, y_values, color=PURPLE, zorder=5)
            axs[i].plot(x_fit, y_fit, '--', color=PURPLE, 
                       label=f'Fit: {popt[0]:.2f}/x^{popt[1]:.2f} + {popt[2]:.2f}' if fit_type == "power_law" else f'Fit: {popt[0]:.2f}/x + {popt[1]:.2f}', 
                       zorder=4)
            
            # Set scales and labels
            axs[i].set_xscale('log')
            axs[i].set_yscale('log')
            axs[i].set_xlabel('Token Count (millions)')
            axs[i].set_ylabel('DCLM Loss')
            
            # Set x-ticks to token counts
            axs[i].set_xticks(x_values)
            axs[i].set_xticklabels([f'{int(x)}M' for x in x_values])
            
            # Add title for this subplot
            axs[i].set_title(f'Model Size: {value_pretty_name_dict.get(model_size, model_size)}')
            
            # Add grid
            axs[i].grid(True, which='major', linestyle='--', alpha=0.7)
            
            # Add legend
            axs[i].legend(loc='upper right')
            
            # Add value labels
            for x, y in zip(x_values, y_values):
                axs[i].text(x * 1.05, y, f'{y:.3f}', verticalalignment='bottom')
            
            # Add green bounding boxes around certified points
            for j, (x, y, is_certified) in enumerate(zip(x_values, y_values, certified_values)):
                if is_certified:
                    axs[i].scatter(x, y, s=70, facecolors='none', edgecolors='green', 
                                 linewidth=2, marker='o', zorder=6)
            
            # Prepare table data - transpose to show token counts as columns
            hyperparams = ['Weight Decay', 'Learning Rate', 'Epochs']
            table_headers = ['Hyperparameter'] + [f'{int(tc/1_000_000)}M' for tc in token_counts]
            
            table_data = []
            for hparam in hyperparams:
                row = [hparam]
                for token_count in token_counts:
                    run = best_run_dict[token_count][model_size]
                    if hparam == 'Weight Decay':
                        row.append(f"{run['weight_decay']:.1f}")
                    elif hparam == 'Learning Rate':
                        row.append(f"{run['lr']:.0e}")
                    elif hparam == 'Epochs':
                        row.append(f"{run['epochs']}")
                table_data.append(row)
            
            # Create table using automatic positioning
            table = axs[i].table(cellText=table_data,
                               colLabels=table_headers,
                               cellLoc='center',
                               loc='bottom',
                               bbox=[0.0, -0.5, 1.0, 0.3])
            
            # Style the table
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1, 1.0)
            
            # Color the header row
            for j in range(len(table_headers)):
                table[(0, j)].set_facecolor('#E6E6FA')
                table[(0, j)].set_text_props(weight='bold')
            
            # Color the first column (hyperparameter names)
            for row_idx in range(1, len(table_data) + 1):
                table[(row_idx, 0)].set_facecolor('#F0F0F0')
                table[(row_idx, 0)].set_text_props(weight='bold')
                
        except RuntimeError as e:
            print(f"\nWarning: Could not fit power law for model size {model_size}")
            print(e)
    
    # Automatic spacing adjustments
    plt.subplots_adjust(hspace=0.4)
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig('plots/token_scaling.png', bbox_inches='tight')
    plt.close()
    
    # Return asymptotes for reuse
    return valid_model_sizes, asymptotes

def plot_asymptotes_vs_model_size(model_sizes, asymptotes):
    """Creates a plot showing asymptotes vs model size using pre-fitted results."""
    def power_law(x, A, B, C):
        return A / (x ** B) + C
    
    # Create plot
    plt.figure(figsize=(10, 6), dpi=300)
    
    # Convert model sizes to parameter counts
    model_params_values = [model_params[model] for model in model_sizes]
    
    # Fit a power law to the asymptotes vs model parameters
    popt, _ = curve_fit(power_law, model_params_values, asymptotes, p0=[1.0, 0.5, 2.0], bounds=([0, 0, 0], [np.inf, np.inf, np.inf]))
    x_fit = np.logspace(np.log10(min(model_params_values)), np.log10(max(model_params_values)), 100)
    y_fit = power_law(x_fit, *popt)

    plt.scatter(model_params_values, asymptotes, color=PURPLE, s=100, zorder=5)
    plt.plot(x_fit, y_fit, '--', color=PURPLE, zorder=4, label=f'Fit: {popt[0]:.2f}/x^{popt[1]:.2f} + {popt[2]:.2f}')
    
    # Set labels and title
    plt.xscale('log')
    plt.xlabel('Model Size (Billion Parameters)')
    plt.ylabel('Asymptotic Loss (Infinite Tokens)')
    plt.title('Asymptotic Loss vs Model Size')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Set x-ticks to model sizes with pretty names
    plt.xticks(model_params_values, [value_pretty_name_dict.get(model, model) for model in model_sizes])
    
    # Add value labels
    for x, y, model in zip(model_params_values, asymptotes, model_sizes):
        plt.text(x * 1.1, y, f'{y:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('plots/asymptotes_vs_model_size.png', bbox_inches='tight')
    plt.close()

def fit_joint_scaling_law(best_run_dict):
    """Fits a joint scaling law of the form: loss = E + A/model^alpha + B/data^beta"""
    
    def joint_scaling_law(params, A, alpha, B, beta, E):
        """Joint scaling law function"""
        model_size, data_size = params
        return E + A / (model_size ** alpha) + B / (data_size ** beta)
    
    # Extract all data points
    model_sizes = []
    data_sizes = []
    losses = []
    
    token_counts = sorted(best_run_dict.keys())
    model_names = sorted(list(best_run_dict[token_counts[0]].keys()), key=lambda x: model_params[x])
    
    for token_count in token_counts:
        for model_name in model_names:
            run = best_run_dict[token_count][model_name]
            model_sizes.append(model_params[model_name])  # In billions
            data_sizes.append(token_count / 1_000_000)  # Convert to millions
            losses.append(run["final_dclm_loss"])
    
    # Prepare data for fitting
    model_sizes = np.array(model_sizes)
    data_sizes = np.array(data_sizes)
    losses = np.array(losses)
    
    # Stack model and data sizes for the joint function
    x_data = np.column_stack((model_sizes, data_sizes))
    
    print(f"\nFitting joint scaling law to {len(losses)} data points...")
    print(f"Model sizes range: {model_sizes.min():.1f}B - {model_sizes.max():.1f}B parameters")
    print(f"Data sizes range: {data_sizes.min():.0f}M - {data_sizes.max():.0f}M tokens")
    print(f"Loss range: {losses.min():.3f} - {losses.max():.3f}")
    
    try:
        # Fit joint scaling law
        # Initial guess: A=1, alpha=0.5, B=1, beta=0.5, E=2.0
        popt, pcov = curve_fit(
            lambda params, A, alpha, B, beta, E: joint_scaling_law(params, A, alpha, B, beta, E),
            x_data.T,  # Transpose for the lambda function
            losses,
            p0=[1.0, 0.5, 1.0, 0.5, 2.0],
            bounds=([0, 0, 0, 0, 0], [np.inf, np.inf, np.inf, np.inf, np.inf])
        )
        
        A, alpha, B, beta, E = popt
        param_errors = np.sqrt(np.diag(pcov))
        
        print(f"\nJoint Scaling Law Fit Results:")
        print(f"loss = {E:.3f} + {A:.3f}/model^{alpha:.3f} + {B:.3f}/data^{beta:.3f}")
        print(f"\nDetailed Parameters:")
        print(f"E (irreducible loss): {E:.3f} ± {param_errors[4]:.3f}")
        print(f"A (model scaling coefficient): {A:.3f} ± {param_errors[0]:.3f}")
        print(f"alpha (model scaling exponent): {alpha:.3f} ± {param_errors[1]:.3f}")
        print(f"B (data scaling coefficient): {B:.3f} ± {param_errors[2]:.3f}")
        print(f"beta (data scaling exponent): {beta:.3f} ± {param_errors[3]:.3f}")
        
        # Calculate R-squared
        y_pred = joint_scaling_law((model_sizes, data_sizes), A, alpha, B, beta, E)
        ss_res = np.sum((losses - y_pred) ** 2)
        ss_tot = np.sum((losses - np.mean(losses)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)
        print(f"R-squared: {r_squared:.4f}")
        
        # Calculate mean absolute percentage error
        mape = np.mean(np.abs((losses - y_pred) / losses)) * 100
        print(f"Mean Absolute Percentage Error: {mape:.2f}%")
        
        return popt, pcov, (model_sizes, data_sizes, losses)
        
    except RuntimeError as e:
        print(f"Failed to fit joint scaling law: {e}")
        return None, None, None

def plot_model_scaling_simple(best_run_dict, fit_type="power_law"):
    """Creates a single plot showing loss vs model size for all token counts with power law fits."""
    def power_law(x, A, B, C):
        return A / (x ** B) + C

    def reciprocal_law(x, A, C):
        return A / x + C

    # Get unique model sizes and token counts
    token_counts = sorted(best_run_dict.keys())
    token_counts_m = [tc / 1_000_000 for tc in token_counts]
    model_sizes = sorted(list(best_run_dict[token_counts[0]].keys()), key=lambda x: model_params[x])
    
    # Create single plot
    plt.figure(figsize=(8, 5), dpi=300)
    
    # Colors for different token counts
    colors = [PURPLE, LIGHT_BLUE, GREEN, '#e74c3c']
    
    for i, token_count in enumerate(token_counts):
        # Get data for this token count
        x_values = np.array([model_params[model] for model in model_sizes])
        y_values = np.array([best_run_dict[token_count][model]["final_dclm_loss"] for model in model_sizes])
        
        color = colors[i % len(colors)]
        token_count_m = token_count / 1_000_000  # Convert to millions
        
        # Fit power law
        try:
            if fit_type == "power_law":
                popt, _ = curve_fit(power_law, x_values, y_values, p0=[1.0, 0.5, 2.0], bounds=([0, 0, 0], [np.inf, np.inf, np.inf]))
            elif fit_type == "reciprocal_law":
                popt, _ = curve_fit(reciprocal_law, x_values, y_values, p0=[1.0, 2.0], bounds=([0, 0], [np.inf, np.inf]))
            
            # Generate points for smooth curve
            x_fit = np.logspace(np.log10(min(x_values)), np.log10(max(x_values)), 100)
            y_fit = power_law(x_fit, *popt) if fit_type == "power_law" else reciprocal_law(x_fit, *popt)
            
            # Plot data points and fitted curve
            plt.scatter(x_values, y_values, color=color, zorder=5, s=60)
            plt.plot(x_fit, y_fit, '--', color=color, zorder=4, alpha=0.8,
                    label=f'{int(token_count_m)}M tokens (fit: {popt[0]:.2f}/x^{popt[1]:.2f} + {popt[2]:.2f})' if fit_type == "power_law" else f'{int(token_count_m)}M tokens (fit: {popt[0]:.2f}/x + {popt[1]:.2f})')
                
        except RuntimeError as e:
            print(f"\nWarning: Could not fit power law for token count {token_count}")
            print(e)
    
    # Set scales and labels
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Model Size (Billion Parameters)')
    plt.ylabel('DCLM Loss')
    plt.title('Model Scaling Laws Across Different Token Counts')
    
    # Set x-ticks to model sizes
    model_params_values = [model_params[model] for model in model_sizes]
    plt.xticks(model_params_values, [value_pretty_name_dict.get(model, model) for model in model_sizes])
    
    # Add grid and legend
    plt.grid(True, which='major', linestyle='--', alpha=0.7)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('plots/model_scaling_simple.png', bbox_inches='tight')
    plt.close()

def plot_token_scaling_simple(best_run_dict, fit_type="power_law"):
    """Creates a single plot showing loss vs token count for all model sizes with power law fits."""
    def power_law(x, A, B, C):
        return A / (x ** B) + C

    def reciprocal_law(x, A, C):
        return A / x + C

    # Get unique model sizes and token counts
    token_counts = sorted(best_run_dict.keys())
    token_counts_m = [tc / 1_000_000 for tc in token_counts]
    model_sizes = sorted(list(best_run_dict[token_counts[0]].keys()), key=lambda x: model_params[x])
    
    # Create single plot
    plt.figure(figsize=(8, 5), dpi=300)
    
    # Colors for different model sizes
    colors = [PURPLE, LIGHT_BLUE, GREEN, '#e74c3c']
    
    for i, model_size in enumerate(model_sizes):
        # Get data for this model size across all token counts
        x_values = np.array([tc / 1_000_000 for tc in token_counts])  # Convert to millions
        y_values = np.array([best_run_dict[tc][model_size]["final_dclm_loss"] for tc in token_counts])
        
        color = colors[i % len(colors)]
        
        # Fit power law
        try:
            if fit_type == "power_law":
                popt, _ = curve_fit(power_law, x_values, y_values, p0=[1.0, 0.5, 2.0], bounds=([0, 0, 0], [np.inf, np.inf, np.inf]))
            elif fit_type == "reciprocal_law":
                popt, _ = curve_fit(reciprocal_law, x_values, y_values, p0=[1.0, 2.0], bounds=([0, 0], [np.inf, np.inf]))
            
            # Generate points for smooth curve
            x_fit = np.logspace(np.log10(min(x_values)), np.log10(max(x_values)), 100)
            y_fit = power_law(x_fit, *popt) if fit_type == "power_law" else reciprocal_law(x_fit, *popt)
            
            # Plot data points and fitted curve
            model_name = value_pretty_name_dict.get(model_size, model_size)
            plt.scatter(x_values, y_values, color=color, zorder=5, s=60)
            plt.plot(x_fit, y_fit, '--', color=color, zorder=4, alpha=0.8,
                    label=f'{model_name} (fit: {popt[0]:.2f}/x^{popt[1]:.2f} + {popt[2]:.2f})' if fit_type == "power_law" else f'{model_name} (fit: {popt[0]:.2f}/x + {popt[1]:.2f})')
            
        except RuntimeError as e:
            print(f"\nWarning: Could not fit power law for model size {model_size}")
            print(e)

    #### hacky
    # heuristically selected hparams
    ensemble_runs_manual = {
        "150m4k": (token_counts_m, [3.494941417720857, 3.399930198165533, 3.357577868802383, 3.2995859709652104]),
        "300m4k": (token_counts_m, [3.2729813419904183, 3.1503966937866053, 3.1018990697199413, 3.0172424035912697]),
        "600m4k": (token_counts_m, [3.2122277662817145, 3.0519037696079496, 2.9058312693222317, 2.8262327566492838]),
        "1_4b4k": (token_counts_m, [3.173966150932168, 3.002520844567612, 2.8680140499504847, 2.752375695327966]),
    }

    # # same hparams
    # ensemble_runs_manual = {
    #     "150m4k": (token_counts_m[:2], [3.519, 3.435]),
    #     "300m4k": (token_counts_m, [3.336, 3.196, 3.102, 3.079]),
    #     "600m4k": (token_counts_m, [3.297, 3.098, 2.906, 2.826]),
    #     "1_4b4k": (token_counts_m, [3.341, 3.091, 2.888, 2.800]),
    # }

    for i, (model_size, (token_counts_m_current, losses)) in enumerate(ensemble_runs_manual.items()):
        color = colors[i % len(colors)]
        plt.scatter(token_counts_m_current, losses, color=color, s=100, zorder=5)
        popt, _ = curve_fit(power_law, token_counts_m_current, losses, p0=[1.0, 0.5, 2.0], bounds=([0, 0, 0], [np.inf, np.inf, np.inf]))
        x_fit = np.logspace(np.log10(min(token_counts_m_current)), np.log10(max(token_counts_m_current)), 100)
        y_fit = power_law(x_fit, *popt)
        plt.plot(x_fit, y_fit, '-', color=color, zorder=4, label=f'{model_size} $\infty$ ensembles: {popt[0]:.2f}/x^{popt[1]:.2f} + {popt[2]:.2f}')

    # ensemble_tiered_asymptotes = []
    # for i in range(len(token_counts_m)):
    #     model_sizes = []
    #     all_losses = []
    #     for model_size, (token_counts_m_current, losses) in ensemble_runs_manual.items():
    #         if i < len(token_counts_m_current):
    #             model_sizes.append(model_params[model_size])
    #             all_losses.append(losses[i])
    #     print(model_sizes, all_losses)
    #     popt, _ = curve_fit(power_law, model_sizes, all_losses, p0=[1.0, 0.5, 2.0], bounds=([0, 0, 0], [np.inf, np.inf, np.inf]))
    #     ensemble_tiered_asymptotes.append(popt[-1])

    # plt.scatter(token_counts_m, ensemble_tiered_asymptotes, color=LIGHT_BLUE, s=100, zorder=5)
    # popt, _ = curve_fit(power_law, token_counts_m, ensemble_tiered_asymptotes, p0=[1.0, 0.5, 2.0], bounds=([0, 0, 0], [np.inf, np.inf, np.inf]))
    # x_fit = np.logspace(np.log10(min(token_counts_m)), np.log10(max(token_counts_m)), 100)
    # y_fit = power_law(x_fit, *popt)
    # plt.plot(x_fit, y_fit, '--', color=LIGHT_BLUE, zorder=4, label=f'Tiered fit: {popt[0]:.2f}/x^{popt[1]:.2f} + {popt[2]:.2f}')
    ### end hacky
    
    # Set scales and labels
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Token Count (millions)')
    plt.ylabel('DCLM Loss')
    plt.title('Token Scaling Laws Across Different Model Sizes')
    
    # Set x-ticks to token counts
    token_counts_m = [tc / 1_000_000 for tc in token_counts]
    plt.xticks(token_counts_m, [f'{int(x)}M' for x in token_counts_m])
    
    # Add grid and legend
    plt.grid(True, which='major', linestyle='--', alpha=0.7)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('plots/token_scaling_simple.png', bbox_inches='tight')
    plt.close()

def plot_200M_sample(losses_for_200M):
    def power_law(x, A, B, C):
        return A / (x ** B) + C

    plt.figure(figsize=(8, 5), dpi=300)
    max_x = max([max(x_data * model_params[model_size]) for model_size, (x_data, _, _, _, _) in losses_for_200M.items()])

    model_x_values = [0.15, 0.3, 0.6, 1.4]
    model_y_values = [3.750, 3.587, 3.510, 3.462]
    plt.scatter(model_x_values, model_y_values, color=PURPLE, s=50)
    # fit a power law to the model scaling
    popt, _ = curve_fit(power_law, model_x_values, model_y_values, p0=[1.0, 0.5, 2.0], bounds=([0, 0, 0], [np.inf, np.inf, np.inf]))
    model_x_fit = np.linspace(min(model_x_values), max_x * 25, 5000)
    model_y_fit = power_law(model_x_fit, *popt)
    plt.plot(model_x_fit, model_y_fit, '--', color=PURPLE, label=f"Model scaling: (Fit: {popt[0]:.2f}/x^{popt[1]:.2f} + {popt[2]:.2f})")

    # add a shaded gray region for any y value above popt[2]
    plt.fill_between(model_x_fit, 3.1, popt[2], color=PURPLE, alpha=0.2, label="Impossible with standard scaling, infinite compute")

    for model_size, (x_data, y_data, A, B, C) in losses_for_200M.items():
        x_fit = np.linspace(min(x_data * model_params[model_size]), max_x * 25, 5000)
        y_fit = A / ((x_fit / model_params[model_size]) ** B) + C
        plt.scatter(x_data * model_params[model_size], y_data)
        plt.plot(x_fit, y_fit, '--', label=f'{value_pretty_name_dict[model_size]} ensembles (Fit: {A:.2f}/(x^{B:.2f}) + {C:.2f})')
    plt.legend()
    plt.xscale('log')
    plt.xlabel('Total model parameters (billions)')
    plt.ylabel('DCLM Loss')
    plt.title('Scaling models and ensembles for 200M tokens')
    plt.savefig('plots/200M_sample.png', bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(8, 5), dpi=300)
    # plot AB/(A+Ce^Bx) for the same x values
    x_fit = np.logspace(np.log10(0.00001), np.log10(max_x * 25), 5000)
    plt.plot(x_fit, [popt[0] * popt[1] / (popt[0] + popt[2] * np.exp(popt[1] * np.log(x))) for x in x_fit], '--', color=PURPLE, label=f"Model scaling: (Fit: {popt[0]:.2f}/x^{popt[1]:.2f} + {popt[2]:.2f})")
    for model_size, (x_data, y_data, A, B, C) in losses_for_200M.items():
        x_fit = np.logspace(np.log10(min(x_data * model_params[model_size]) * 0.0001), np.log10(max_x * 25), 5000)
        y_fit = A * B / (A + C * np.exp(B * np.log(x_fit / model_params[model_size])))
        plt.plot(x_fit, y_fit, '--', label=f'{value_pretty_name_dict[model_size]} ensembles (Fit: {A:.2f}/(x^{B:.2f}) + {C:.2f})')
    plt.legend()
    plt.xscale('log')
    plt.xlabel('Total model parameters (billions)')
    plt.ylabel('Returns on excess log loss')
    plt.title('Returns on excess log loss for 200M tokens')
    plt.savefig('plots/200M_sample_AB_A_Ce_Bx.png', bbox_inches='tight')
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
        "varying-hparams-experiment": ("none", "stanford-mercury/suhas-eval-data-efficiency"),
        "infinite-model-scaling": ("none", "stanford-mercury/suhas-data-efficiency"),
        "seed-science": ("seed-science-7-9", "stanford-mercury/suhas-eval-data-efficiency"),
    }[mode]

    if args.build_cache:
        run_list = []
        runs = wandb.Api().runs(project_name)
        for run in tqdm(runs):
            run_dict = parse_run(run)
            if run_dict is not None:
                print(run_dict["run_id"])
                run_list.append(run_dict)
        pickle.dump(run_list, open(f"cache/{mode}_run_list.pkl", "wb"))
    else:
        run_list = pickle.load(open(f"cache/{mode}_run_list.pkl", "rb"))

    if mode == "data-scaling-laws-6-10":
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

    elif mode == "varying-hparams-experiment":
        unique_models = sorted(list(set([run["model_name"] for run in run_list])), key=lambda x: model_params[x])
        unique_base_tokens = sorted(list(set([run["base_tokens"] for run in run_list])))

        min_base_tokens = min(unique_base_tokens)
        losses_for_200M = {}

        for model_size in unique_models:
            for base_tokens in unique_base_tokens:
                ret = plot_ensemble_scaling(model_size, base_tokens)
                if ret is not None:
                    _, x_data, y_data, A, B, C = ret
                    print(C)
                    if base_tokens == min_base_tokens:
                        losses_for_200M[model_size] = (x_data, y_data, A, B, C)
        
        plot_200M_sample(losses_for_200M)

    elif mode == "infinite-model-scaling":
        # base_token_counts = [104857600.0, 209715200.0, 419430400.0, 838860800.0, 1677721600.0, 6291456000.0]
        base_token_counts = [209715200.0, 419430400.0, 838860800.0, 1677721600.0]
        model_sizes = ['150m4k', '300m4k', '600m4k', '1_4b4k']

        run_list = [run for run in run_list if run["batch_size"] == 64 and "seed" not in run["run_id"] and "-ts" not in run["run_id"] and run["data_name"] == "dclm" and run["epochs"] <= 64 and run["weight_decay"] >= 0.1]
        
        best_run_dict = {}
        for token_count in base_token_counts:
            best_run_dict[token_count] = {}
            print(token_count)
            for model_size in model_sizes:
                filtered_runs = [run for run in run_list if run["base_tokens"] == token_count and run["model_name"] == model_size]
                best_run = min(filtered_runs, key=lambda x: x["final_dclm_loss"])

                base_train_steps = best_run["base_tokens"] * best_run["epochs"] / (best_run["batch_size"] * 4096)
                neighbors = get_bounding_box(base_train_steps, best_run["epochs"], best_run["lr"], best_run["weight_decay"], best_run["model_name"])[1:]

                best_neighbor_loss = float("inf")
                num_neighbors = 0
                for neighbor in neighbors:
                    neighbor_candidates = [run for run in run_list if run["base_tokens"] == best_run["base_tokens"] and run["epochs"] == neighbor[1] and run["lr"] == neighbor[2] and run["weight_decay"] == neighbor[3] and run["model_name"] == neighbor[4]]

                    if len(neighbor_candidates) == 1:
                        best_neighbor_loss = min(best_neighbor_loss, neighbor_candidates[0]["final_dclm_loss"])
                        num_neighbors += 1
                    elif len(neighbor_candidates) != 0:
                        print(neighbor_candidates)
                        raise ValueError(f"Found {len(neighbor_candidates)} neighbors for {neighbor}")
                
                print(model_size, best_run["run_id"], num_neighbors)
                potential_neighbor_count = 6
                if best_run["epochs"] == 64:
                    potential_neighbor_count -= 1
                if best_run["weight_decay"] == 0.1:
                    potential_neighbor_count -= 1
                best_run["convex_certificate"] = num_neighbors == potential_neighbor_count and best_neighbor_loss > best_run["final_dclm_loss"]
                best_run_dict[token_count][model_size] = best_run

        plot_model_scaling(best_run_dict, fit_type="power_law")
        
        # Also create token scaling plot
        valid_model_sizes, asymptotes = plot_token_scaling(best_run_dict, fit_type="power_law")
        plot_asymptotes_vs_model_size(valid_model_sizes, asymptotes)
        
        # Fit joint scaling law
        fit_joint_scaling_law(best_run_dict)
        
        # Create simplified single-plot versions
        plot_model_scaling_simple(best_run_dict, fit_type="power_law")
        plot_token_scaling_simple(best_run_dict, fit_type="power_law")

    elif mode == "seed-science":
        run_list = sorted(run_list, key=lambda x: x["ensemble_member_count"])
        train_seed_losses = [run["final_dclm_loss"] for run in run_list if "train" in run["seed_types"]]
        data_seed_losses = [run["final_dclm_loss"] for run in run_list if "data" in run["seed_types"]]
        both_seed_losses = [run["final_dclm_loss"] for run in run_list if "both" in run["seed_types"]]

        plot_seed_science(train_seed_losses, data_seed_losses, both_seed_losses)

