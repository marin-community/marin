import matplotlib.pyplot as plt
import wandb
from tqdm import tqdm
import argparse
import pickle
import numpy as np
import json
from scipy.optimize import curve_fit
import matplotlib.colors as mcolors
from convex_certificate_scaling import get_bounding_box

plt.rcParams.update({"font.family": "Palatino"})

# Custom color scheme
LIGHT_BLUE = "#8CD9FF"
PURPLE = "#7030A0"
GREEN = "#2ECC71"
CUSTOM_CMAP = mcolors.LinearSegmentedColormap.from_list("custom", [LIGHT_BLUE, PURPLE])
CUSTOM_CMAP.set_bad(color="white", alpha=0)

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

model_params = {"150m4k": 0.15, "300m4k": 0.3, "600m4k": 0.6, "1_4b4k": 1.4, "1_9b4k": 1.9}

token_count_to_steps = {
    "209M": 800,
    "419M": 1600,
    "838M": 3200,
    "1.7B": 6400,
}


def parse_run(run):
    if key != "none" and key not in run.tags:
        print(f"\033[91mSkipping run {run.id} because it does not have the key {key}\033[0m")
        return None

    if "ignore" in run.tags:
        print(f"\033[91mSkipping run {run.id} because it is ignored\033[0m")
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
            run_id = run_id[len("ppl-eval-ensemble-") :]
        elif run_id.startswith("ss-"):
            if not key.startswith("seed-science-"):
                print(f"\033[91mSkipping run {run_id} because it is not a seed science run\033[0m")
                return None

            run_id = run_id[len("ss-") :]

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

        for token_count in token_count_to_steps:
            if token_count in run_id:
                num_base_steps = token_count_to_steps[token_count]
                break
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
    run_dict["base_tokens"] = (
        base_tokens if base_tokens is not None else num_steps * batch_size * seq_len / run_dict["epochs"]
    )
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

    return run_dict


class ScalingLaw:
    def __init__(self):
        self.params = None

    def func(self, x, params):
        pass  # implemented by subclass

    def __str__(self):
        pass  # implemented by subclass

    def asymptote(self):
        pass  # implemented by subclass

    def evaluate(self, x):
        return self.func(x, *self.params)

    def fit(self, x, y, p0=None, bounds=(-np.inf, np.inf)):
        popt, pcov = curve_fit(self.func, x, y, p0=p0, bounds=bounds)
        self.params = popt


class ReciprocalScalingLaw(ScalingLaw):
    def __init__(self):
        super().__init__()

    def func(self, x, A, C):
        return A / x + C

    def __str__(self):
        return f"{self.params[0]:.2f}/x + {self.params[1]:.2f}"

    def asymptote(self):
        return self.params[-1]


class PowerScalingLaw(ScalingLaw):
    def __init__(self):
        super().__init__()

    def func(self, x, A, B, C):
        return A / (x**B) + C

    def __str__(self):
        return f"{self.params[0]:.2f}/x^{self.params[1]:.2f} + {self.params[2]:.2f}"

    def asymptote(self):
        return self.params[-1]


# have data_dict map (epoch, wd) -> loss?
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
                    heatmap_data[i, j] = matching_fits[0][-1].asymptote()  # C value (asymptote)
                else:
                    # Get loss for ensemble size 1
                    x_data = matching_fits[0][1]  # ensemble sizes
                    y_data = matching_fits[0][2]  # losses
                    idx = np.where(x_data == 1)[0][0]
                    heatmap_data[i, j] = y_data[idx]

    # Find minimum value and its indices
    heatmap_data = np.ma.masked_where(heatmap_data == 0, heatmap_data)
    min_val = np.ma.min(heatmap_data)
    min_idx = np.where(heatmap_data == min_val)
    i, j = min_idx[0][0], min_idx[1][0]

    # Draw rectangle around minimum value
    rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, color=PURPLE, linewidth=2)
    ax.add_patch(rect)
    # Plot heatmap
    im = ax.imshow(heatmap_data, aspect="auto", cmap=CUSTOM_CMAP)

    # Add colorbar
    plt.colorbar(im, ax=ax, label="Loss")

    # Set ticks and labels
    ax.set_xticks(range(len(wds)))
    ax.set_yticks(range(len(epochs)))
    ax.set_xticklabels([f"{wd:.1f}" for wd in wds])
    ax.set_yticklabels(epochs)

    # Add labels
    ax.set_xlabel("Weight Decay")
    ax.set_ylabel("Epochs")
    metric = "Asymptotic" if use_asymptote else "Single Member"
    ax.set_title(f"{metric} Loss\n(lr={target_lr})")

    # Add text annotations with values
    for i in range(len(epochs)):
        for j in range(len(wds)):
            if np.ma.is_masked(heatmap_data[i, j]):
                data_str = "N/A"
            else:
                data_str = f"{heatmap_data[i, j]:.3f}"
            _ = ax.text(j, i, data_str, ha="center", va="center", color="black")

    return im


def plot_ensemble_scaling(model_name, base_tokens):
    # First figure: 1/x fits
    plt.figure(figsize=(12, 7), dpi=600)

    def valid_run(run):
        return run["model_name"] == model_name and run["base_tokens"] == base_tokens

    filtered_run_list = [run for run in run_list if valid_run(run)]

    assert len(filtered_run_list) > 0, f"No runs found for model size: {model_name} and base tokens: {base_tokens}"

    print(f"\033[94mCreating plot for model size: {model_name} and base tokens: {base_tokens}\033[0m")

    # Get the data from the last unique key's runs
    unique_keys = set()
    for run in filtered_run_list:
        unique_keys.add((run["epochs"], run["lr"], run["weight_decay"]))

    print(unique_keys)

    # Store fits and data for sorting
    all_fits = []
    for unique_key in unique_keys:
        runs = [
            run
            for run in filtered_run_list
            if run["epochs"] == unique_key[0] and run["lr"] == unique_key[1] and run["weight_decay"] == unique_key[2]
        ]
        runs = sorted(runs, key=lambda x: x["ensemble_member_count"])
        x_data = np.array([run["ensemble_member_count"] for run in runs])
        y_data = np.array([run["final_dclm_loss"] for run in runs])

        if len(x_data) <= 2:
            continue

        # Fit 1/x curve
        power_law = PowerScalingLaw()
        power_law.fit(x_data, y_data)
        all_fits.append((unique_key, x_data, y_data, power_law))

    # Sort by asymptote (C value)
    all_fits.sort(key=lambda x: x[-1].asymptote())

    # Plot in order of asymptote
    for unique_key, x_data, y_data, power_law in all_fits:
        # Generate points for smooth curve
        x_fit = np.linspace(min(x_data), max(x_data), 100)
        y_fit = power_law.evaluate(x_fit)

        plt.scatter(x_data, y_data)
        plt.plot(
            x_fit,
            y_fit,
            "--",
            label=f"Fit (epochs={unique_key[0]}, lr={unique_key[1]}, wd={unique_key[2]}): {power_law}",
        )

    plt.xlabel("Ensemble Member Count")
    plt.ylabel("DCLM Loss")
    plt.title("Loss vs Ensemble Size with 1/x Fits")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"plots/ensemble_scaling_fits_{model_name}_{base_tokens}.png", bbox_inches="tight")
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
    plt.savefig(f"plots/ensemble_scaling_heatmaps_{model_name}_{base_tokens}.png", bbox_inches="tight")
    plt.close()

    if all_fits[0][-1].asymptote() < 1.8:
        print("SKIPPING FIRST RUN SINCE IT'S LIKELY BUGGED")
        return all_fits[1]
    return all_fits[0]


def plot_model_scaling(best_run_dict, fit_type="power_law"):
    """Creates subplots showing loss vs model size for each token count with power law fits."""

    # Get unique model sizes and token counts
    token_counts = sorted(best_run_dict.keys())
    model_sizes = sorted(list(best_run_dict[token_counts[0]].keys()), key=lambda x: model_params[x])

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
                power_law = PowerScalingLaw()
                power_law.fit(x_values, y_values, p0=[1.0, 0.5, 2.0], bounds=([0, 0, 0], [np.inf, np.inf, np.inf]))
            elif fit_type == "reciprocal_law":
                reciprocal_law = ReciprocalScalingLaw()
                reciprocal_law.fit(x_values, y_values, p0=[1.0, 2.0], bounds=([0, 0], [np.inf, np.inf]))
            # fitted_params[token_count] = power_law.params if fit_type == "power_law" else reciprocal_law.params
            valid_token_counts.append(token_count)
            asymptotes.append(power_law.asymptote())  # Store asymptote for reuse

            # Generate points for smooth curve
            x_fit = np.logspace(np.log10(min(x_values)), np.log10(max(x_values)), 100)
            y_fit = power_law.evaluate(x_fit)

            # Plot data points and fitted curve
            axs[i].scatter(x_values, y_values, color=PURPLE, zorder=5)
            axs[i].plot(x_fit, y_fit, "--", color=PURPLE, label=f"Fit: {power_law}", zorder=4)

            # Set scales and labels
            axs[i].set_xscale("log")
            axs[i].set_yscale("log")
            axs[i].set_xlabel("Model parameters")
            axs[i].set_ylabel("DCLM Loss")

            # Set x-ticks to model sizes
            axs[i].set_xticks(x_values)
            axs[i].set_xticklabels([value_pretty_name_dict.get(model, model) for model in model_sizes])

            # Add title for this subplot
            token_count_m = token_count / 1_000_000  # Convert to millions
            axs[i].set_title(f"Seed token count: {int(token_count_m)}M")

            # Add grid
            axs[i].grid(True, which="major", linestyle="--", alpha=0.7)

            # Add legend
            axs[i].legend(loc="upper right")

            # Add value labels
            for x, y in zip(x_values, y_values, strict=False):
                axs[i].text(x * 1.05, y, f"{y:.3f}", verticalalignment="bottom")

            # Add green bounding boxes around certified points
            for x, y, is_certified in zip(x_values, y_values, certified_values, strict=False):
                if is_certified:
                    axs[i].scatter(x, y, s=70, facecolors="none", edgecolors="green", linewidth=2, marker="o", zorder=6)

            # Prepare table data
            hyperparams = ["Weight Decay", "Learning Rate", "Epochs"]
            table_headers = ["Hyperparameter"] + [value_pretty_name_dict.get(model, model) for model in model_sizes]

            table_data = []
            for hparam in hyperparams:
                row = [hparam]
                for model in model_sizes:
                    run = best_run_dict[token_count][model]
                    if hparam == "Weight Decay":
                        row.append(f"{run['weight_decay']:.1f}")
                    elif hparam == "Learning Rate":
                        row.append(f"{run['lr']:.0e}")
                    elif hparam == "Epochs":
                        row.append(f"{run['epochs']}")
                table_data.append(row)

            # Create table using automatic positioning
            table = axs[i].table(
                cellText=table_data, colLabels=table_headers, cellLoc="center", loc="bottom", bbox=[0.0, -0.5, 1.0, 0.3]
            )

            # Style the table
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1, 1.0)

            # Color the header row
            for j in range(len(table_headers)):
                table[(0, j)].set_facecolor("#E6E6FA")
                table[(0, j)].set_text_props(weight="bold")

            # Color the first column (hyperparameter names)
            for row_idx in range(1, len(table_data) + 1):
                table[(row_idx, 0)].set_facecolor("#F0F0F0")
                table[(row_idx, 0)].set_text_props(weight="bold")

        except RuntimeError as e:
            print(f"\nWarning: Could not fit power law for token count {token_count}")
            print(e)

    # Automatic spacing adjustments
    plt.subplots_adjust(hspace=0.4)  # Reduced from fixed large spacing
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])  # Leave space for tables at bottom
    plt.savefig("plots/model_scaling.png", bbox_inches="tight")
    plt.close()

    """Creates a plot showing asymptotes vs token count using pre-fitted results."""

    # Create plot
    plt.figure(figsize=(8, 6), dpi=300)

    # Convert token counts to millions for better readability
    token_counts_m = [tc / 1_000_000 for tc in token_counts]

    # Fit a power law to the asymptotes vs token_counts_m
    power_law = PowerScalingLaw()
    power_law.fit(token_counts_m, asymptotes, p0=[1.0, 0.5, 2.0], bounds=([0, 0, 0], [np.inf, np.inf, np.inf]))
    x_fit = np.logspace(np.log10(min(token_counts_m)), np.log10(max(token_counts_m)), 100)
    y_fit = power_law.evaluate(x_fit)

    plt.scatter(token_counts_m, asymptotes, color=PURPLE, s=100, zorder=5)
    plt.plot(x_fit, y_fit, "--", color=PURPLE, zorder=4, label=f"Fit: {power_law}")

    #### hacky
    # heuristically selected hparams
    best_ensembles = pickle.load(open("cache/varying-hparams-experiment_best_ensembles.pkl", "rb"))
    ensemble_runs = {}
    for model_size, base_tokens_dict in best_ensembles.items():
        ensemble_runs[model_size] = []
        for _, (_, _, power_law) in base_tokens_dict.items():
            ensemble_runs[model_size].append(power_law.asymptote())

    for model_size, losses in ensemble_runs.items():
        plt.scatter(token_counts_m, losses, s=100, zorder=5)
        power_law = PowerScalingLaw()
        power_law.fit(token_counts_m, losses, p0=[1.0, 0.5, 2.0], bounds=([0, 0, 0], [np.inf, np.inf, np.inf]))
        x_fit = np.logspace(np.log10(min(token_counts_m)), np.log10(max(token_counts_m)), 100)
        y_fit = power_law.evaluate(x_fit)
        plt.plot(x_fit, y_fit, "--", zorder=4, label=rf"{model_size} $\infty$ ensembles: {power_law}")

    ensemble_tiered_asymptotes = []
    for i in range(len(token_counts_m)):
        model_sizes = []
        all_losses = []
        for model_size, losses in ensemble_runs.items():
            if i < len(token_counts_m):
                model_sizes.append(model_params[model_size])
                all_losses.append(losses[i])
        power_law = PowerScalingLaw()
        power_law.fit(model_sizes, all_losses, p0=[1.0, 0.5, 2.0], bounds=([0, 0, 0], [np.inf, np.inf, np.inf]))
        ensemble_tiered_asymptotes.append(power_law.asymptote())

    plt.scatter(token_counts_m, ensemble_tiered_asymptotes, color=LIGHT_BLUE, s=100, zorder=5)
    power_law = PowerScalingLaw()
    power_law.fit(
        token_counts_m, ensemble_tiered_asymptotes, p0=[1.0, 0.5, 2.0], bounds=([0, 0, 0], [np.inf, np.inf, np.inf])
    )
    x_fit = np.logspace(np.log10(min(token_counts_m)), np.log10(max(token_counts_m)), 100)
    y_fit = power_law.evaluate(x_fit)
    plt.plot(x_fit, y_fit, "--", color=LIGHT_BLUE, zorder=4, label=f"Tiered fit: {power_law}")
    ### end hacky

    # Set labels and title
    plt.xscale("log")
    plt.xlabel("Seed token count (millions)")
    plt.ylabel("Asymptotic Loss")
    plt.title("Asymptotic Loss vs Training Token Count")
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Add value labels
    for x, y in zip(token_counts_m, asymptotes, strict=False):
        plt.text(x * 1.1, y, f"{y:.3f}", ha="center", va="bottom")

    plt.tight_layout()
    plt.savefig("plots/asymptotes_vs_tokens.png", bbox_inches="tight")
    plt.close()


def plot_seed_science(train_seed_losses, data_seed_losses, both_seed_losses):
    plt.figure(figsize=(8, 5), dpi=300)

    # Fit power laws to each curve
    x_data = np.array(range(1, 6))
    x_fit = np.linspace(1, 5, 100)

    print(train_seed_losses)
    print(data_seed_losses)
    print(both_seed_losses)

    # Train seed curve fit
    train_seed_power_law = PowerScalingLaw()
    train_seed_power_law.fit(x_data, train_seed_losses, p0=[1.0, 0.5, 2.0], bounds=([0, 0, 0], [np.inf, np.inf, np.inf]))
    train_seed_y_fit = train_seed_power_law.evaluate(x_fit)

    # Data seed curve fit
    data_seed_power_law = PowerScalingLaw()
    data_seed_power_law.fit(x_data, data_seed_losses, p0=[1.0, 0.5, 2.0], bounds=([0, 0, 0], [np.inf, np.inf, np.inf]))
    data_seed_y_fit = data_seed_power_law.evaluate(x_fit)

    # Both seeds curve fit
    both_seed_power_law = PowerScalingLaw()
    both_seed_power_law.fit(x_data, both_seed_losses, p0=[1.0, 0.5, 2.0], bounds=([0, 0, 0], [np.inf, np.inf, np.inf]))
    both_seed_y_fit = both_seed_power_law.evaluate(x_fit)

    # Plot fitted curves
    plt.plot(x_fit, train_seed_y_fit, "--", color=PURPLE, zorder=4, label=f"Only train seed: {train_seed_power_law}")
    plt.plot(x_fit, data_seed_y_fit, "--", color=GREEN, zorder=4, label=f"Only data seed: {data_seed_power_law}")
    plt.plot(x_fit, both_seed_y_fit, "--", color=LIGHT_BLUE, zorder=4, label=f"Both seeds: {both_seed_power_law}")

    # Plot data points
    plt.scatter(x_data, train_seed_losses, color=PURPLE, s=100, zorder=5)
    plt.scatter(x_data, data_seed_losses, color=GREEN, s=100, zorder=5)
    plt.scatter(x_data, both_seed_losses, color=LIGHT_BLUE, s=100, zorder=5)

    plt.xlabel("Ensemble member count")
    plt.ylabel("Loss")
    plt.xticks(x_data, [f"{int(x)}" for x in x_data])
    plt.title("Ensemble loss for different sources of randomness (300M parameters, 200M tokens)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("plots/seed_science.png", bbox_inches="tight")
    plt.close()


def plot_token_scaling(best_run_dict, fit_type="power_law"):
    """Creates subplots showing loss vs token count for each model size with power law fits."""

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
                scaling_law = PowerScalingLaw()
                scaling_law.fit(x_values, y_values, p0=[1.0, 0.5, 2.0], bounds=([0, 0, 0], [np.inf, np.inf, np.inf]))
            elif fit_type == "reciprocal_law":
                reciprocal_law = ReciprocalScalingLaw()
                reciprocal_law.fit(x_values, y_values, p0=[1.0, 2.0], bounds=([0, 0], [np.inf, np.inf]))

            # Store asymptote for reuse
            asymptotes.append(scaling_law.asymptote())  # Last parameter is the asymptote
            valid_model_sizes.append(model_size)

            # Generate points for smooth curve
            x_fit = np.logspace(np.log10(min(x_values)), np.log10(max(x_values)), 100)
            y_fit = scaling_law.evaluate(x_fit)

            # Plot data points and fitted curve
            axs[i].scatter(x_values, y_values, color=PURPLE, zorder=5)
            axs[i].plot(x_fit, y_fit, "--", color=PURPLE, label=f"Fit: {scaling_law}", zorder=4)

            # Set scales and labels
            axs[i].set_xscale("log")
            axs[i].set_yscale("log")
            axs[i].set_xlabel("Token Count (millions)")
            axs[i].set_ylabel("DCLM Loss")

            # Set x-ticks to token counts
            axs[i].set_xticks(x_values)
            axs[i].set_xticklabels([f"{int(x)}M" for x in x_values])

            # Add title for this subplot
            axs[i].set_title(f"Model Size: {value_pretty_name_dict.get(model_size, model_size)}")

            # Add grid
            axs[i].grid(True, which="major", linestyle="--", alpha=0.7)

            # Add legend
            axs[i].legend(loc="upper right")

            # Add value labels
            for x, y in zip(x_values, y_values, strict=False):
                axs[i].text(x * 1.05, y, f"{y:.3f}", verticalalignment="bottom")

            # Add green bounding boxes around certified points
            for x, y, is_certified in zip(x_values, y_values, certified_values, strict=False):
                if is_certified:
                    axs[i].scatter(x, y, s=70, facecolors="none", edgecolors="green", linewidth=2, marker="o", zorder=6)

            # Prepare table data - transpose to show token counts as columns
            hyperparams = ["Weight Decay", "Learning Rate", "Epochs"]
            table_headers = ["Hyperparameter"] + [f"{int(tc/1_000_000)}M" for tc in token_counts]

            table_data = []
            for hparam in hyperparams:
                row = [hparam]
                for token_count in token_counts:
                    run = best_run_dict[token_count][model_size]
                    if hparam == "Weight Decay":
                        row.append(f"{run['weight_decay']:.1f}")
                    elif hparam == "Learning Rate":
                        row.append(f"{run['lr']:.0e}")
                    elif hparam == "Epochs":
                        row.append(f"{run['epochs']}")
                table_data.append(row)

            # Create table using automatic positioning
            table = axs[i].table(
                cellText=table_data, colLabels=table_headers, cellLoc="center", loc="bottom", bbox=[0.0, -0.5, 1.0, 0.3]
            )

            # Style the table
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1, 1.0)

            # Color the header row
            for j in range(len(table_headers)):
                table[(0, j)].set_facecolor("#E6E6FA")
                table[(0, j)].set_text_props(weight="bold")

            # Color the first column (hyperparameter names)
            for row_idx in range(1, len(table_data) + 1):
                table[(row_idx, 0)].set_facecolor("#F0F0F0")
                table[(row_idx, 0)].set_text_props(weight="bold")

        except RuntimeError as e:
            print(f"\nWarning: Could not fit power law for model size {model_size}")
            print(e)

    # Automatic spacing adjustments
    plt.subplots_adjust(hspace=0.4)
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig("plots/token_scaling.png", bbox_inches="tight")
    plt.close()

    # Return asymptotes for reuse
    return valid_model_sizes, asymptotes


def plot_asymptotes_vs_model_size(model_sizes, asymptotes):
    """Creates a plot showing asymptotes vs model size using pre-fitted results."""

    # Create plot
    plt.figure(figsize=(10, 6), dpi=300)

    # Convert model sizes to parameter counts
    model_params_values = [model_params[model] for model in model_sizes]

    # Fit a power law to the asymptotes vs model parameters
    power_law = PowerScalingLaw()
    power_law.fit(model_params_values, asymptotes, p0=[1.0, 0.5, 2.0], bounds=([0, 0, 0], [np.inf, np.inf, np.inf]))
    x_fit = np.logspace(np.log10(min(model_params_values)), np.log10(max(model_params_values)), 100)
    y_fit = power_law.evaluate(x_fit)

    plt.scatter(model_params_values, asymptotes, color=PURPLE, s=100, zorder=5)
    plt.plot(x_fit, y_fit, "--", color=PURPLE, zorder=4, label=f"Fit: {power_law}")

    # Set labels and title
    plt.xscale("log")
    plt.xlabel("Model Size (Billion Parameters)")
    plt.ylabel("Asymptotic Loss (Infinite Tokens)")
    plt.title("Asymptotic Loss vs Model Size")
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Set x-ticks to model sizes with pretty names
    plt.xticks(model_params_values, [value_pretty_name_dict.get(model, model) for model in model_sizes])

    # Add value labels
    for x, y in zip(model_params_values, asymptotes, strict=False):
        plt.text(x * 1.1, y, f"{y:.3f}", ha="center", va="bottom")

    plt.tight_layout()
    plt.savefig("plots/asymptotes_vs_model_size.png", bbox_inches="tight")
    plt.close()


def fit_joint_scaling_law(best_run_dict):
    """Fits a joint scaling law of the form: loss = E + A/model^alpha + B/data^beta"""

    def joint_scaling_law(params, A, alpha, B, beta, E):
        """Joint scaling law function"""
        model_size, data_size = params
        return E + A / (model_size**alpha) + B / (data_size**beta)

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
            bounds=([0, 0, 0, 0, 0], [np.inf, np.inf, np.inf, np.inf, np.inf]),
        )

        A, alpha, B, beta, E = popt
        param_errors = np.sqrt(np.diag(pcov))

        print("\nJoint Scaling Law Fit Results:")
        print(f"loss = {E:.3f} + {A:.3f}/model^{alpha:.3f} + {B:.3f}/data^{beta:.3f}")
        print("\nDetailed Parameters:")
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

    # Get unique model sizes and token counts
    token_counts = sorted(best_run_dict.keys())
    model_sizes = sorted(list(best_run_dict[token_counts[0]].keys()), key=lambda x: model_params[x])

    # Create single plot
    plt.figure(figsize=(8, 5), dpi=300)

    # Colors for different token counts
    colors = [PURPLE, LIGHT_BLUE, GREEN, "#e74c3c"]

    for i, token_count in enumerate(token_counts):
        # Get data for this token count
        x_values = np.array([model_params[model] for model in model_sizes])
        y_values = np.array([best_run_dict[token_count][model]["final_dclm_loss"] for model in model_sizes])

        color = colors[i % len(colors)]
        token_count_m = token_count / 1_000_000  # Convert to millions

        # Fit power law
        try:
            if fit_type == "power_law":
                scaling_law = PowerScalingLaw()
                scaling_law.fit(x_values, y_values, p0=[1.0, 0.5, 2.0], bounds=([0, 0, 0], [np.inf, np.inf, np.inf]))
            elif fit_type == "reciprocal_law":
                scaling_law = ReciprocalScalingLaw()
                scaling_law.fit(x_values, y_values, p0=[1.0, 2.0], bounds=([0, 0], [np.inf, np.inf]))

            # Generate points for smooth curve
            x_fit = np.logspace(np.log10(min(x_values)), np.log10(max(x_values)), 100)
            y_fit = scaling_law.evaluate(x_fit)

            # Plot data points and fitted curve
            plt.scatter(x_values, y_values, color=color, zorder=5, s=60)
            plt.plot(
                x_fit,
                y_fit,
                "--",
                color=color,
                zorder=4,
                alpha=0.8,
                label=f"{int(token_count_m)}M tokens (fit: {scaling_law})",
            )

        except RuntimeError as e:
            print(f"\nWarning: Could not fit power law for token count {token_count}")
            print(e)

    # Set scales and labels
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Model Size (Billion Parameters)")
    plt.ylabel("DCLM Loss")
    plt.title("Model Scaling Laws Across Different Token Counts")

    # Set x-ticks to model sizes
    model_params_values = [model_params[model] for model in model_sizes]
    plt.xticks(model_params_values, [value_pretty_name_dict.get(model, model) for model in model_sizes])

    # Add grid and legend
    plt.grid(True, which="major", linestyle="--", alpha=0.7)
    plt.legend()

    plt.tight_layout()
    plt.savefig("plots/model_scaling_simple.png", bbox_inches="tight")
    plt.close()


def plot_token_scaling_simple(best_run_dict, fit_type="power_law"):
    """Creates a single plot showing loss vs token count for all model sizes with power law fits."""

    # Get unique model sizes and token counts
    token_counts = sorted(best_run_dict.keys())
    token_counts_m = [tc / 1_000_000 for tc in token_counts]
    model_sizes = sorted(list(best_run_dict[token_counts[0]].keys()), key=lambda x: model_params[x])

    # Create single plot
    plt.figure(figsize=(8, 5), dpi=300)

    # Colors for different model sizes
    colors = [PURPLE, LIGHT_BLUE, GREEN, "#e74c3c"]

    for i, model_size in enumerate(model_sizes):
        # Get data for this model size across all token counts
        x_values = np.array([tc / 1_000_000 for tc in token_counts])  # Convert to millions
        y_values = np.array([best_run_dict[tc][model_size]["final_dclm_loss"] for tc in token_counts])

        color = colors[i % len(colors)]

        # Fit power law
        try:
            if fit_type == "power_law":
                scaling_law = PowerScalingLaw()
                scaling_law.fit(x_values, y_values, p0=[1.0, 0.5, 2.0], bounds=([0, 0, 0], [np.inf, np.inf, np.inf]))
            elif fit_type == "reciprocal_law":
                scaling_law = ReciprocalScalingLaw()
                scaling_law.fit(x_values, y_values, p0=[1.0, 2.0], bounds=([0, 0], [np.inf, np.inf]))

            # Generate points for smooth curve
            x_fit = np.logspace(np.log10(min(x_values)), np.log10(max(x_values)), 100)
            y_fit = scaling_law.evaluate(x_fit)

            # Plot data points and fitted curve
            model_name = value_pretty_name_dict.get(model_size, model_size)
            plt.scatter(x_values, y_values, color=color, zorder=5, s=60)
            plt.plot(x_fit, y_fit, "--", color=color, zorder=4, alpha=0.8, label=f"{model_name} (fit: {scaling_law})")

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
        scaling_law = PowerScalingLaw()
        scaling_law.fit(
            token_counts_m_current,
            losses,
            p0=[1.0, 0.5, 2.0],
            bounds=([0, 0, 0], [np.inf, np.inf, np.inf]),
        )
        x_fit = np.logspace(np.log10(min(token_counts_m_current)), np.log10(max(token_counts_m_current)), 100)
        y_fit = scaling_law.evaluate(x_fit)
        plt.plot(x_fit, y_fit, "-", color=color, zorder=4, label=rf"{model_size} $\infty$ ensembles: {scaling_law}")

    # ensemble_tiered_asymptotes = []
    # for i in range(len(token_counts_m)):
    #     model_sizes = []
    #     all_losses = []
    #     for model_size, (token_counts_m_current, losses) in ensemble_runs_manual.items():
    #         if i < len(token_counts_m_current):
    #             model_sizes.append(model_params[model_size])
    #             all_losses.append(losses[i])
    #     print(model_sizes, all_losses)
    #     popt, _ = curve_fit(
    #         power_law,
    #         model_sizes,
    #         all_losses,
    #         p0=[1.0, 0.5, 2.0],
    #         bounds=([0, 0, 0], [np.inf, np.inf, np.inf]),
    #     )
    #     ensemble_tiered_asymptotes.append(popt[-1])

    # plt.scatter(token_counts_m, ensemble_tiered_asymptotes, color=LIGHT_BLUE, s=100, zorder=5)
    # popt, _ = curve_fit(
    #     power_law,
    #     token_counts_m,
    #     ensemble_tiered_asymptotes,
    #     p0=[1.0, 0.5, 2.0],
    #     bounds=([0, 0, 0], [np.inf, np.inf, np.inf]),
    # )
    # x_fit = np.logspace(np.log10(min(token_counts_m)), np.log10(max(token_counts_m)), 100)
    # y_fit = power_law(x_fit, *popt)
    # plt.plot(
    #     x_fit,
    #     y_fit,
    #     '--',
    #     color=LIGHT_BLUE,
    #     zorder=4,
    #     label=f'Tiered fit: {popt[0]:.2f}/x^{popt[1]:.2f} + {popt[2]:.2f}',
    # )
    ### end hacky

    # Set scales and labels
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Token Count (millions)")
    plt.ylabel("DCLM Loss")
    plt.title("Token Scaling Laws Across Different Model Sizes")

    # Set x-ticks to token counts
    token_counts_m = [tc / 1_000_000 for tc in token_counts]
    plt.xticks(token_counts_m, [f"{int(x)}M" for x in token_counts_m])

    # Add grid and legend
    plt.grid(True, which="major", linestyle="--", alpha=0.7)
    plt.legend()

    plt.tight_layout()
    plt.savefig("plots/token_scaling_simple.png", bbox_inches="tight")
    plt.close()


def plot_benchmark_results():
    from eval_200m_models import VANILLA_MODEL_SCALING, ENSEMBLE_MODEL_SCALING, get_base_name
    import json

    with open("experiments/data_efficiency/200m_benchmark_results.json", "r") as f:
        benchmark_results = json.load(f)
    for model in VANILLA_MODEL_SCALING + ENSEMBLE_MODEL_SCALING:
        for seed in range(model.num_seeds):
            base_name = get_base_name(model)
            ensemble_results = benchmark_results[base_name][str(seed + 1)]
            benchmark_results[base_name][str(seed + 1)]["avg_acc"] = np.mean(
                [ensemble_results[task]["acc"] for task in ensemble_results]
            )

    plt.figure(figsize=(8, 5), dpi=300)
    model_x_values = [model_params[model.model_size] for model in VANILLA_MODEL_SCALING]
    model_y_values = [1.0 - benchmark_results[get_base_name(model)]["1"]["avg_acc"] for model in VANILLA_MODEL_SCALING]
    plt.plot(model_x_values, model_y_values, color=PURPLE, label="Model Scaling", marker="o")
    for model in ENSEMBLE_MODEL_SCALING:
        model_x_values = [model_params[model.model_size] * (seed + 1) for seed in range(model.num_seeds)]
        model_y_values = [
            1.0 - benchmark_results[get_base_name(model)][str(seed + 1)]["avg_acc"] for seed in range(model.num_seeds)
        ]
        plt.plot(
            model_x_values,
            model_y_values,
            label=f"Ensemble {value_pretty_name_dict[model.model_size]} Scaling",
            marker="o",
        )

    plt.legend()
    plt.grid(True, which="major", linestyle="--", alpha=0.7)
    plt.xscale("log")
    plt.xlabel("Total model parameters (billions)")
    plt.ylabel("Average Error")
    plt.title("Scaling Models and Ensembles for 200M Tokens (Downstream Benchmarks)")
    plt.savefig("plots/benchmark_results.png", bbox_inches="tight")
    plt.close()


def plot_distillation(losses_for_200M):
    plt.figure(figsize=(8, 5), dpi=300)
    max_x = max([max(x_data * model_params[model_size]) for model_size, (x_data, _, _) in losses_for_200M.items()])

    # TODO: standardize this
    model_x_values = [0.15, 0.3, 0.6, 1.4]
    model_y_values = [3.750, 3.587, 3.510, 3.462]
    plt.scatter(model_x_values, model_y_values, color=PURPLE, s=50)

    # fit a power law to the model scaling
    base_power_law = PowerScalingLaw()
    base_power_law.fit(model_x_values, model_y_values, p0=[1.0, 0.5, 2.0], bounds=([0, 0, 0], [np.inf, np.inf, np.inf]))
    model_x_fit = np.linspace(min(model_x_values), max_x * 25, 5000)
    model_y_fit = base_power_law.evaluate(model_x_fit)
    plt.plot(model_x_fit, model_y_fit, "--", color=PURPLE, label=f"Model scaling: (Fit: {base_power_law})")

    # add a shaded gray region for any y value above popt[2]
    plt.fill_between(
        model_x_fit,
        3.1,
        base_power_law.asymptote(),
        color=PURPLE,
        alpha=0.2,
        label="Impossible with standard scaling, infinite compute",
    )

    for model_size, (x_data, y_data, power_law) in losses_for_200M.items():
        if model_size == "300m4k":
            x_fit = np.linspace(min(x_data * model_params[model_size]), max_x * 25, 5000)
            y_fit = power_law.evaluate(x_fit / model_params[model_size])
            plt.scatter(x_data * model_params[model_size], y_data, color="tab:orange")
            plt.plot(
                x_fit,
                y_fit,
                "--",
                color="tab:orange",
                label=f"{value_pretty_name_dict[model_size]} ensembles (Fit: {power_law})",
            )

            # Best self-distill run: 300m4k-209Mx16-dclm+sd0805^0.75-cos-lr0.0030-wd0.10-bs64 (3.43243)
            plt.scatter(
                0.3,
                3.43243,
                color="tab:red",
                edgecolors="black",
                linewidths=0.8,
                marker="*",
                s=120,
                zorder=6,
                label="Self-Distill: 3.43",
            )
            plt.annotate(
                "",
                xytext=(0.3 * x_data[0], y_data[0]),
                xy=(0.3, 3.43243),
                arrowprops=dict(
                    arrowstyle="->", color="black", linestyle="--", alpha=0.5, connectionstyle="arc3,rad=0.25"
                ),
                zorder=0,
            )

            # Best 8-mixture distill run: 300m4k-209Mx16-dclm+ens8x0730^0.95-cos-lr0.0030-wd0.01-bs64 (3.3612)
            plt.scatter(
                0.3,
                3.3612,
                color=LIGHT_BLUE,
                edgecolors="black",
                linewidths=0.8,
                marker="*",
                s=120,
                zorder=6,
                label="8-Ensemble Distill: 3.36",
            )
            plt.annotate(
                "",
                xytext=(0.3 * x_data[-1], y_data[-1]),
                xy=(0.3, 3.3612),
                arrowprops=dict(
                    arrowstyle="->", color="black", linestyle="--", alpha=0.5, connectionstyle="arc3,rad=-0.25"
                ),
                zorder=0,
            )

            plt.axvline(x=0.3, color="grey", linestyle="--", alpha=0.3, zorder=0)
    plt.legend()
    plt.xscale("log")
    plt.xlabel("Total model parameters (billions)")
    plt.ylabel("DCLM Loss")
    plt.title("Distilling a 300M model (200M seed tokens)")
    plt.savefig("plots/distillation.png", bbox_inches="tight")
    plt.close()


def plot_200M_sample(losses_for_200M):
    plt.figure(figsize=(8, 5), dpi=300)
    max_x = max([max(x_data * model_params[model_size]) for model_size, (x_data, _, _) in losses_for_200M.items()])

    # TODO: standardize this
    model_x_values = [0.15, 0.3, 0.6, 1.4]
    model_y_values = [3.750, 3.587, 3.510, 3.462]
    plt.scatter(model_x_values, model_y_values, color=PURPLE, s=50)
    # fit a power law to the model scaling
    base_power_law = PowerScalingLaw()
    base_power_law.fit(model_x_values, model_y_values, p0=[1.0, 0.5, 2.0], bounds=([0, 0, 0], [np.inf, np.inf, np.inf]))
    model_x_fit = np.linspace(min(model_x_values), max_x * 25, 5000)
    model_y_fit = base_power_law.evaluate(model_x_fit)
    plt.plot(model_x_fit, model_y_fit, "--", color=PURPLE, label=f"Model scaling: (Fit: {base_power_law})")

    # add a shaded gray region for any y value above popt[2]
    plt.fill_between(
        model_x_fit,
        3.1,
        base_power_law.asymptote(),
        color=PURPLE,
        alpha=0.2,
        label="Impossible with standard scaling, infinite compute",
    )

    for model_size, (x_data, y_data, power_law) in losses_for_200M.items():
        x_fit = np.linspace(min(x_data * model_params[model_size]), max_x * 25, 5000)
        y_fit = power_law.evaluate(x_fit / model_params[model_size])
        plt.scatter(x_data * model_params[model_size], y_data)
        plt.plot(x_fit, y_fit, "--", label=f"{value_pretty_name_dict[model_size]} ensembles (Fit: {power_law})")
    plt.legend()
    plt.xscale("log")
    plt.xlabel("Total model parameters (billions)")
    plt.ylabel("DCLM Loss")
    plt.title("Scaling models and ensembles for 200M tokens")
    plt.savefig("plots/200M_sample.png", bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(8, 5), dpi=300)
    # plot AB/(A+Ce^Bx) for the same x values
    x_fit = np.logspace(np.log10(0.00001), np.log10(max_x * 25), 5000)
    A, B, C = base_power_law.params
    plt.plot(
        x_fit,
        [A * B / (A + C * np.exp(B * np.log(x))) for x in x_fit],
        "--",
        color=PURPLE,
        label=f"Model scaling: (Fit: {A:.2f}/x^{B:.2f} + {C:.2f})",
    )
    for model_size, (x_data, _, power_law) in losses_for_200M.items():
        A, B, C = power_law.params
        x_fit = np.logspace(np.log10(min(x_data * model_params[model_size]) * 0.0001), np.log10(max_x * 25), 5000)
        y_fit = A * B / (A + C * np.exp(B * np.log(x_fit / model_params[model_size])))
        plt.plot(
            x_fit,
            y_fit,
            "--",
            label=f"{value_pretty_name_dict[model_size]} ensembles (Fit: {A:.2f}/(x^{B:.2f}) + {C:.2f})",
        )
    plt.legend()
    plt.xscale("log")
    plt.xlabel("Total model parameters (billions)")
    plt.ylabel("Returns on excess log loss")
    plt.title("Returns on excess log loss for 200M tokens")
    plt.savefig("plots/200M_sample_AB_A_Ce_Bx.png", bbox_inches="tight")
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
        "seed-science": ("seed-science-8-7", "stanford-mercury/suhas-eval-data-efficiency"),
        "benchmark-results": ("none", "none"),
        "distillation": ("none", "stanford-mercury/suhas-eval-data-efficiency"),
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
        if mode != "benchmark-results":
            run_list = pickle.load(open(f"cache/{mode}_run_list.pkl", "rb"))

    if mode == "varying-hparams-experiment":
        unique_models = sorted(list(set([run["model_name"] for run in run_list])), key=lambda x: model_params[x])
        unique_base_tokens = sorted(list(set([run["base_tokens"] for run in run_list])))

        best_ensembles = {}

        for model_size in unique_models:
            best_ensembles[model_size] = {}
            for base_tokens in unique_base_tokens:
                ret = plot_ensemble_scaling(model_size, base_tokens)
                if ret is not None:
                    _, x_data, y_data, power_law = ret
                    print(power_law.asymptote())
                    best_ensembles[model_size][base_tokens] = (x_data, y_data, power_law)

        for base_tokens in unique_base_tokens[:1]:
            plot_200M_sample({model_size: best_ensembles[model_size][base_tokens] for model_size in unique_models})
        # plot_200M_sample(best_ensembles[min(unique_models)])
        print(best_ensembles)
        pickle.dump(best_ensembles, open(f"cache/{mode}_best_ensembles.pkl", "wb"))

    elif mode == "infinite-model-scaling":
        base_token_counts = [209715200.0, 419430400.0, 838860800.0, 1677721600.0]
        model_sizes = ["150m4k", "300m4k", "600m4k", "1_4b4k"]

        epoch_ub = 64
        weight_decay_lb = 0.1
        run_list = [
            run
            for run in run_list
            if run["batch_size"] == 64
            and "seed" not in run["run_id"]
            and "-ts" not in run["run_id"]
            and run["data_name"] == "dclm"
            and run["epochs"] <= epoch_ub
            and run["weight_decay"] >= weight_decay_lb
        ]

        best_run_dict = {}
        for token_count in base_token_counts:
            best_run_dict[token_count] = {}
            for model_size in model_sizes:
                filtered_runs = [
                    run for run in run_list if run["base_tokens"] == token_count and run["model_name"] == model_size
                ]
                best_run = min(filtered_runs, key=lambda x: x["final_dclm_loss"])

                base_train_steps = best_run["base_tokens"] * best_run["epochs"] / (best_run["batch_size"] * 4096)
                neighbors = get_bounding_box(
                    base_train_steps,
                    best_run["epochs"],
                    best_run["lr"],
                    best_run["weight_decay"],
                    best_run["model_name"],
                )[1:]

                best_neighbor_loss = float("inf")
                num_neighbors = 0
                for neighbor in neighbors:
                    neighbor_candidates = [
                        run
                        for run in run_list
                        if run["base_tokens"] == best_run["base_tokens"]
                        and run["epochs"] == neighbor[1]
                        and run["lr"] == neighbor[2]
                        and run["weight_decay"] == neighbor[3]
                        and run["model_name"] == neighbor[4]
                    ]

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
                best_run["convex_certificate"] = (
                    num_neighbors == potential_neighbor_count and best_neighbor_loss > best_run["final_dclm_loss"]
                )
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

    elif mode == "benchmark-results":
        plot_benchmark_results()

    elif mode == "distillation":
        unique_models = sorted(list(set([run["model_name"] for run in run_list])), key=lambda x: model_params[x])
        unique_base_tokens = sorted(list(set([run["base_tokens"] for run in run_list])))

        best_ensembles = {}

        for model_size in unique_models:
            best_ensembles[model_size] = {}
            for base_tokens in unique_base_tokens:
                ret = plot_ensemble_scaling(model_size, base_tokens)
                if ret is not None:
                    _, x_data, y_data, power_law = ret
                    print(power_law.asymptote())
                    best_ensembles[model_size][base_tokens] = (x_data, y_data, power_law)

        for base_tokens in unique_base_tokens[:1]:
            plot_distillation({model_size: best_ensembles[model_size][base_tokens] for model_size in unique_models})
