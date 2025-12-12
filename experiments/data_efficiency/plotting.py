# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import matplotlib.pyplot as plt
import wandb
from tqdm import tqdm
import argparse
import pickle
import numpy as np
import json
from scipy.optimize import curve_fit
import matplotlib.colors as mcolors
from experiments.data_efficiency.utils import get_bounding_box

plt.rcParams.update(
    {
        "font.family": "Palatino",
        "mathtext.fontset": "custom",
        "mathtext.rm": "Palatino",
        "mathtext.it": "Palatino:italic",
        "mathtext.bf": "Palatino:bold",
    }
)

# plt.rcParams.update({"font.size": 12})

# Custom color scheme
LIGHT_BLUE = "#8CD9FF"
PURPLE = "#7030A0"
GREEN = "#55A630"
RED = "#e74c3c"
CRAYOLA_BLUE = "#5178D5"
ROBIN_EGG_BLUE = "#45BFC6"
APPLE_GREEN = "#89B537"
ORANGE = "#F8961E"
GOLD = "#C7A020"
PINK = "#FD8BCE"
VIRIDIAN = "#2B8D6E"
BLACK = "#000000"
DARK_ORANGE = "#FF8C00"


CUSTOM_CMAP = mcolors.LinearSegmentedColormap.from_list("custom", [LIGHT_BLUE, PURPLE])
CUSTOM_CMAP.set_bad(color="white", alpha=0)

BASE_TOKEN_COUNTS = [209715200.0, 419430400.0, 838860800.0, 1677721600.0]
PARAM_STRS = ["150m4k", "300m4k", "600m4k", "1_4b4k"]

value_pretty_name_dict = {
    "150m4k": "150M",
    "300m4k": "300M",
    "600m4k": "600M",
    "1_4b4k": "1.4B",
    0.15: "150M",
    0.3: "300M",
    0.6: "600M",
    1.4: "1.4B",
    209715200.0: "209M",
    419430400.0: "419M",
    838860800.0: "838M",
    1677721600.0: "1.7B",
}

param_str_to_count = {"150m4k": 0.15, "300m4k": 0.3, "600m4k": 0.6, "1_4b4k": 1.4, "1_5b4k": 1.5, "3_2b4k": 3.2}
# param_str_to_count = {"150m4k": 0.15, "300m4k": 0.3, "600m4k": 0.6, "1_4b4k": 1.4}
PARAM_COUNTS = list(param_str_to_count.values())

token_str_to_steps = {
    "209M": 800,
    "419M": 1600,
    "838M": 3200,
    "1.7B": 6400,
}

token_count_marker_dict = {
    209715200.0: "o",
    419430400.0: "^",
    838860800.0: "s",
    1677721600.0: "X",
}

param_str_color_dict = {
    "150m4k": CRAYOLA_BLUE,
    "300m4k": ROBIN_EGG_BLUE,
    "600m4k": APPLE_GREEN,
    "1_4b4k": ORANGE,
    "1_5b4k": PINK,
    "3_2b4k": VIRIDIAN,
}

BASELINE_COLOR = "red"
REGULARIZED_COLOR = PURPLE
TIERED_COLOR = GOLD

ENSEMBLE_COLOR = PINK
SELF_DISTILL_COLOR = VIRIDIAN
WRAP_COLOR = BLACK
MIX_COLOR = DARK_ORANGE

ABLATION_FIGSIZE = (5, 4)
SEED_SCALING_FIGSIZE = (5, 5)
MEDIUM_RECTANGLE_FIGSIZE = (6, 5)
WIDE_RECTANGLE_FIGSIZE = (7, 5)


def parse_run(run):
    if key == "dclm-default-sweep" or key == "suhas-dclm-chinchilla-lr-tune":
        return parse_chinchilla_run(run)

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

        for token_str in token_str_to_steps:
            if token_str in run_id:
                num_base_steps = token_str_to_steps[token_str]
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

    run_history_loss_keys = [f"eval/{run_dict['data_name']}/loss", "train/loss"]

    if run_id.startswith("ppl-eval-ensemble-") or run_id.startswith("ss-"):
        run_history_loss_keys = [f"eval/{run_dict['data_name']}/loss"]

    history_loss = run.history(keys=run_history_loss_keys)

    if f"eval/{run_dict['data_name']}/loss" not in history_loss.columns:
        print(f"\033[91mSkipping run {run_id} because it does not have the loss history\033[0m")
        return None

    run_dict["loss_history"] = history_loss
    run_dict[f"final_{run_dict['data_name']}_loss"] = history_loss[f"eval/{run_dict['data_name']}/loss"].iloc[-1]

    if "train/loss" in history_loss.columns:
        run_dict["final_train_loss"] = history_loss["train/loss"].iloc[-1]

    return run_dict


def parse_chinchilla_run(run):
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
    run_summary_dict = run._attrs["summaryMetrics"]

    run_dict["parameter_count"] = run_summary_dict["parameter_count"]
    run_dict["token_count"] = run_summary_dict["throughput/total_tokens"]
    run_dict["data_name"] = "dclm"

    flop_str = run_id.split("-")[1]
    run_dict["flops"] = int(flop_str.split("e+")[0]) * (10 ** int(flop_str.split("e+")[1]))

    run_dict["batch_size"] = run_json_config["trainer"]["value"]["train_batch_size"]

    run_dict["num_layers"] = run_json_config["model"]["value"]["num_layers"]
    run_dict["intermediate_dim"] = run_json_config["model"]["value"]["intermediate_dim"]
    run_dict["hidden_dim"] = run_json_config["model"]["value"]["hidden_dim"]
    run_dict["num_heads"] = run_json_config["model"]["value"]["num_heads"]
    run_dict["num_kv_heads"] = run_json_config["model"]["value"]["num_kv_heads"]
    run_dict["seq_len"] = run_json_config["model"]["value"]["seq_len"]
    run_dict["vocab_size"] = 128_256

    # run_dict["flops_per_token"] = lm_flops_per_token(
    #     run_dict["hidden_dim"],
    #     run_dict["intermediate_dim"],
    #     run_dict["num_layers"],
    #     run_dict["num_kv_heads"],
    #     run_dict["num_heads"],
    #     run_dict["seq_len"],
    #     run_dict["vocab_size"],
    #     glu=True,
    # )

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
    def __init__(self, var_name="x"):
        super().__init__()
        self.var_name = var_name

    def func(self, x, A, C):
        return A / x + C

    def __str__(self):
        return f"{self.params[0]:.2f}/{self.var_name} + {self.params[1]:.2f}"

    def asymptote(self):
        return self.params[-1]


class PowerScalingLaw(ScalingLaw):
    def __init__(self, var_name="x"):
        super().__init__()
        self.var_name = var_name

    def func(self, x, A, B, C):
        return A / (x**B) + C

    def __str__(self):
        return f"{self.params[0]:.2f}/{self.var_name}^{self.params[1]:.2f} + {self.params[2]:.2f}"

    def asymptote(self):
        return self.params[-1]


class ChinchillaScalingLaw(ScalingLaw):
    def __init__(self, var1_name="N", var2_name="D"):
        super().__init__()
        self.var1_name = var1_name
        self.var2_name = var2_name

    def func(self, x, A, alpha, B, beta, E):
        n, d = x
        return A / (n**alpha) + B / (d**beta) + E

    def __str__(self):
        return (
            f"{self.params[0]:.2f}/{self.var1_name}^{self.params[1]:.2f} + "
            f"{self.params[2]:.2f}/{self.var2_name}^{self.params[3]:.2f} + "
            f"{self.params[4]:.2f}"
        )

    def asymptote(self):
        return self.params[-1]


def back_out_data(loss, law):
    # loss = A / d^B + C
    # d = (A / (loss - C)) ** (1/B)
    A, B, C = law.params
    return (A / (loss - C)) ** (1 / B)


def construct_best_run_dict(run_list):
    epoch_ub = 64
    weight_decay_lb = 0.0
    lr_ub = 3e-3
    run_list = [
        run
        for run in run_list
        if run["batch_size"] == 64
        and "seed" not in run["run_id"]
        and "-ts" not in run["run_id"]
        and run["data_name"] == "dclm"
        and run["epochs"] <= epoch_ub
        and run["weight_decay"] >= weight_decay_lb
        and run["lr"] <= lr_ub
    ]

    best_run_dict = {}
    for token_count in BASE_TOKEN_COUNTS:
        best_run_dict[token_count] = {}
        for model_size in PARAM_STRS:
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
            best_run["convex_certificate"] = num_neighbors == 6 and best_neighbor_loss >= best_run["final_dclm_loss"]
            best_run_dict[token_count][model_size] = best_run
    return best_run_dict


# have data_dict map (epoch, wd) -> loss?
def create_heatmap(ax, all_fits, target_lr, use_asymptote=True, main_body=False, use_box=True):
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
    best_color = "black" if not use_asymptote else PINK
    if use_box:
        rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, color=best_color, linewidth=2)
        ax.add_patch(rect)
    # Plot heatmap
    im = ax.imshow(heatmap_data, aspect="auto", cmap=CUSTOM_CMAP)

    # Set ticks and labels
    ax.set_xticks(range(len(wds)))
    ax.set_yticks(range(len(epochs)))
    ax.set_xticklabels([f"{wd:.1f}" for wd in wds])
    ax.set_yticklabels(epochs)

    # Add labels
    if not use_asymptote:
        ax.set_ylabel("Epochs")
        plt.colorbar(im, ax=ax)
    else:
        plt.colorbar(im, ax=ax, label="Loss")
    ax.set_xlabel("Weight Decay")
    metric = "Infinite member asymptote" if use_asymptote else "Single member loss"

    title = f"{metric}\n(LR = {target_lr})"
    if main_body:
        title = f"{metric}"
    ax.set_title(title)

    # Add text annotations with values
    for i in range(len(epochs)):
        for j in range(len(wds)):
            if np.ma.is_masked(heatmap_data[i, j]):
                data_str = "N/A"
            else:
                data_str = f"{heatmap_data[i, j]:.3f}"
            _ = ax.text(j, i, data_str, ha="center", va="center", color="black", fontsize=11)

    return im


def plot_ensemble_scaling(model_name, base_tokens):
    def is_hparams_shift(hparams, hparams_shift) -> bool:

        if hparams[2] == 1:
            return True

        epochs_valid = hparams_shift[0] == hparams[0] * 2
        lr_valid = hparams_shift[1] == hparams[1]
        wd_valid = hparams_shift[2] == hparams[2] * 0.5

        if hparams_shift[0] == 128:
            epochs_valid = True
        if hparams_shift[2] == 0.1:
            wd_valid = True

        return epochs_valid and lr_valid and wd_valid

    # First figure: 1/x fits
    plt.figure(figsize=(4, 4), dpi=600)

    def valid_run(run):
        valid = run["model_name"] == model_name and run["base_tokens"] == base_tokens

        if model_name == "600m4k" and base_tokens == 209715200:
            valid = valid and run["lr"] == 1e-3

        return valid

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

        if model_name == "300m4k" and base_tokens == 209715200 and 2 not in x_data:
            # insert x=2, y=3.45994 manually since marin name got corrupted by failed run
            x_data = np.insert(x_data, 1, 2)
            y_data = np.insert(y_data, 1, 3.45994)
            print(x_data, y_data)

        # Fit 1/x curve
        power_law = PowerScalingLaw(var_name="K")
        power_law.fit(x_data, y_data)
        all_fits.append((unique_key, x_data, y_data, power_law))

    # Sort by single member loss
    all_fits.sort(key=lambda x: x[2][0])
    assert all_fits[0][1][0] == 1
    best_single_model_hparams = all_fits[0][0]
    best_single_model_hparams_ensemble = all_fits[0]
    print("Best single model hyperparams: ", best_single_model_hparams)

    # Sort by asymptote
    all_fits.sort(key=lambda x: x[-1].asymptote())

    # Plot in order of asymptote
    for unique_key, x_data, y_data, power_law in all_fits:
        # Generate points for smooth curve
        x_fit = np.linspace(min(x_data), max(x_data) + 5, 100)
        y_fit = power_law.evaluate(x_fit)

        alpha = 1.0
        label = f"{unique_key[0]} epochs, {unique_key[2]} WD (Fit: {power_law})"
        color = None

        if model_name == "300m4k" and base_tokens == 209715200:
            if unique_key[1] != 3e-3:
                continue

            alpha = 0.15
            if unique_key[0] == 32 and unique_key[2] == 0.8:
                alpha = 1.0
                color = PINK
                label = f"Best asymptote: {unique_key[0]} epochs, {unique_key[2]} WD\n(Fit: {power_law})"
            elif unique_key[0] == 16 and unique_key[2] == 1.6:
                alpha = 1.0
                label = f"Best single model: {unique_key[0]} epochs, {unique_key[2]} WD\n(Fit: {power_law})"
                color = "black"
            else:
                label = None

        plt.scatter(x_data[:5], y_data[:5], alpha=alpha, color=color)
        plt.plot(
            x_fit,
            y_fit,
            "--",
            alpha=alpha,
            color=color,
            label=label,
        )

    plt.xlabel("Ensemble member count $K$")
    plt.ylabel("DCLM Loss")
    plt.title("Varying ensemble hyper-parameters")
    plt.legend()
    # plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    plt.xscale("log")
    plt.xticks([1, 2, 3, 4, 5], ["1", "2", "3", "4", "5"])
    plt.yticks([], [], minor=True)
    plt.xlim(right=5.5)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        f"experiments/data_efficiency/plots/ensemble_hparams/ensemble_scaling_fits_{value_pretty_name_dict[base_tokens]}_{value_pretty_name_dict[model_name]}.png",
        bbox_inches="tight",
    )
    plt.close()

    unique_learning_rates = list(set([fit[0][1] for fit in all_fits]))
    unique_learning_rates.sort()

    # Second figure: heatmaps
    fig, axs = plt.subplots(len(unique_learning_rates), 2, figsize=(7, 4 * len(unique_learning_rates)), dpi=300)

    if len(unique_learning_rates) == 1:
        create_heatmap(axs[0], all_fits, unique_learning_rates[0], use_asymptote=False)
        create_heatmap(axs[1], all_fits, unique_learning_rates[0], use_asymptote=True)
    else:
        for i, lr in enumerate(unique_learning_rates):
            use_box = True
            if model_name == "300m4k" and base_tokens == 209715200 and lr == 1e-3:
                use_box = False
            create_heatmap(axs[i, 0], all_fits, lr, use_asymptote=False, use_box=use_box)
            create_heatmap(axs[i, 1], all_fits, lr, use_asymptote=True, use_box=use_box)

    fig.suptitle(
        f"Ensemble tuning for $N=${value_pretty_name_dict[model_name]}, $D=${value_pretty_name_dict[base_tokens]}",
        fontsize=18,
    )
    plt.tight_layout()
    plt.savefig(
        f"experiments/data_efficiency/plots/ensemble_hparams/ensemble_scaling_heatmaps_{value_pretty_name_dict[base_tokens]}_{value_pretty_name_dict[model_name]}.png",
        bbox_inches="tight",
    )
    plt.close()

    if model_name == "300m4k" and base_tokens == 209715200:
        # Main body sweeping hyper-parameters
        fig, axs = plt.subplots(1, 2, figsize=(7, 4), dpi=600)

        unique_learning_rates = unique_learning_rates[1:]
        assert len(unique_learning_rates) == 1

        create_heatmap(axs[0], all_fits, unique_learning_rates[0], use_asymptote=False, main_body=True)
        create_heatmap(axs[1], all_fits, unique_learning_rates[0], use_asymptote=True, main_body=True)

        plt.tight_layout()
        plt.savefig(
            "experiments/data_efficiency/plots/ensemble_hparams/example_200M_varying_hparams.png", bbox_inches="tight"
        )
        plt.close()

    best_fit = all_fits[0]

    if all_fits[0][-1].asymptote() < 1.8:
        print("SKIPPING FIRST RUN SINCE IT'S LIKELY BUGGED")
        best_fit = all_fits[1]

    # if model_name == "1_4b4k" and base_tokens == 209715200:
    #     best_fit = all_fits[2]

    best_fit_hparams = best_fit[0]

    print("Best asymptote hyperparams: ", best_fit_hparams)

    if model_name == "1_4b4k" and base_tokens == 209715200:
        return best_fit, best_single_model_hparams_ensemble

    assert is_hparams_shift(
        best_single_model_hparams, best_fit_hparams
    ), "Best fit hyperparams are not a shift of the single model hyperparams"

    return best_fit, best_single_model_hparams_ensemble


def plot_model_scaling(best_run_dict, fit_type="power_law"):
    """Creates subplots showing loss vs model size for each token count with power law fits."""

    # Get unique model sizes and token counts
    token_counts = sorted(best_run_dict.keys())
    model_sizes = sorted(list(best_run_dict[token_counts[0]].keys()), key=lambda x: param_str_to_count[x])

    valid_token_counts = []
    power_laws = []
    run_losses = []

    # Create figure with subplots arranged horizontally with extra space for tables
    fig, axs = plt.subplots(1, len(token_counts), figsize=(4.2 * len(token_counts), 8), dpi=300)

    # If only one token count, wrap axes in list for consistent indexing
    if len(token_counts) == 1:
        axs = [axs]

    for i, token_count in enumerate(token_counts):
        # Get data for this token count
        x_values = np.array([param_str_to_count[model] for model in model_sizes])
        y_values = np.array([best_run_dict[token_count][model]["final_dclm_loss"] for model in model_sizes])
        certified_values = np.array([best_run_dict[token_count][model]["convex_certificate"] for model in model_sizes])

        # Fit power law
        try:
            if fit_type == "power_law":
                power_law = PowerScalingLaw(var_name="N")
                power_law.fit(x_values, y_values, p0=[1.0, 0.5, 2.0], bounds=([0, 0, 0], [np.inf, np.inf, np.inf]))
            elif fit_type == "reciprocal_law":
                reciprocal_law = ReciprocalScalingLaw(var_name="N")
                reciprocal_law.fit(x_values, y_values, p0=[1.0, 2.0], bounds=([0, 0], [np.inf, np.inf]))
            # fitted_params[token_count] = power_law.params if fit_type == "power_law" else reciprocal_law.params
            valid_token_counts.append(token_count)
            power_laws.append(power_law)  # Store asymptote for reuse
            run_losses.append(y_values)

            # Generate points for smooth curve
            x_fit = np.logspace(np.log10(min(x_values)), np.log10(max(x_values)), 100)
            y_fit = power_law.evaluate(x_fit)

            # Plot data points and fitted curve
            axs[i].scatter(x_values, y_values, color=PURPLE, zorder=5)
            axs[i].plot(x_fit, y_fit, "--", color=PURPLE, label=f"{power_law}", zorder=4)

            # Set scales and labels
            axs[i].set_xscale("log")
            axs[i].set_yscale("log")
            axs[i].set_xlabel("Model parameters")
            axs[i].set_ylabel("DCLM Loss")

            # Set x-ticks to model sizes
            axs[i].set_xticks(x_values)
            axs[i].set_xticklabels([value_pretty_name_dict.get(model, model) for model in model_sizes])

            # Add title for this subplot
            axs[i].set_title(f"{value_pretty_name_dict[token_count]} seed tokens")

            # Add grid
            axs[i].grid(True, which="major", linestyle="--", alpha=0.7)

            # Add legend
            axs[i].legend(loc="upper right")

            # # Add value labels
            # for x, y in zip(x_values, y_values, strict=False):
            #     axs[i].text(x * 1.05, y, f"{y:.3f}", verticalalignment="bottom")

            # Add green bounding boxes around certified points
            for x, y, is_certified in zip(x_values, y_values, certified_values, strict=False):
                if not is_certified:
                    axs[i].scatter(x, y, s=70, facecolors="none", edgecolors=RED, linewidth=2, marker="o", zorder=6)

            # Prepare table data
            hyperparams = ["WD", "LR", "E"]
            table_headers = [""] + [value_pretty_name_dict.get(model, model) for model in model_sizes]

            table_data = []
            for hparam in hyperparams:
                row = [hparam]
                for model in model_sizes:
                    run = best_run_dict[token_count][model]
                    if hparam == "WD":
                        row.append(f"{run['weight_decay']:.1f}")
                    elif hparam == "LR":
                        row.append(f"{run['lr']:.0e}")
                    elif hparam == "E":
                        row.append(f"{run['epochs']}")
                table_data.append(row)

            # Create table using automatic positioning
            table = axs[i].table(
                cellText=table_data, colLabels=table_headers, cellLoc="center", loc="bottom", bbox=[0.0, -0.6, 1.0, 0.4]
            )

            # Style the table
            # table.auto_set_font_size(False)
            # table.set_fontsize(11)
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
    plt.subplots_adjust(wspace=0.3)  # Horizontal spacing between subplots
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])  # Leave space for tables at bottom
    plt.savefig("experiments/data_efficiency/plots/model_scaling.png", bbox_inches="tight")
    plt.close()

    return power_laws, run_losses


def plot_seed_science(train_seed_losses, data_seed_losses, both_seed_losses):
    plt.figure(figsize=(8, 5), dpi=300)

    # Fit power laws to each curve
    x_data = np.array(range(1, 6))
    x_fit = np.linspace(1, 5, 100)

    print(train_seed_losses)
    print(data_seed_losses)
    print(both_seed_losses)

    # Train seed curve fit
    train_seed_power_law = PowerScalingLaw(var_name="K")
    train_seed_power_law.fit(x_data, train_seed_losses, p0=[1.0, 0.5, 2.0], bounds=([0, 0, 0], [np.inf, np.inf, np.inf]))
    train_seed_y_fit = train_seed_power_law.evaluate(x_fit)

    # Data seed curve fit
    data_seed_power_law = PowerScalingLaw(var_name="K")
    data_seed_power_law.fit(x_data, data_seed_losses, p0=[1.0, 0.5, 2.0], bounds=([0, 0, 0], [np.inf, np.inf, np.inf]))
    data_seed_y_fit = data_seed_power_law.evaluate(x_fit)

    # Both seeds curve fit
    both_seed_power_law = PowerScalingLaw(var_name="K")
    both_seed_power_law.fit(x_data, both_seed_losses, p0=[1.0, 0.5, 2.0], bounds=([0, 0, 0], [np.inf, np.inf, np.inf]))
    both_seed_y_fit = both_seed_power_law.evaluate(x_fit)

    # Plot fitted curves
    plt.plot(x_fit, x_fit * 0 + train_seed_losses[0], "--", color="black", zorder=4, label="Varying no seed")
    plt.plot(
        x_fit, train_seed_y_fit, "--", color=PURPLE, zorder=4, label=f"Only train seed (fit: {train_seed_power_law})"
    )
    plt.plot(x_fit, data_seed_y_fit, "--", color=GREEN, zorder=4, label=f"Only data seed (fit: {data_seed_power_law})")
    plt.plot(x_fit, both_seed_y_fit, "--", color=LIGHT_BLUE, zorder=4, label=f"Both seeds (fit: {both_seed_power_law})")

    # Plot data points
    plt.scatter(x_data, train_seed_losses, color=PURPLE, s=100, zorder=5)
    plt.scatter(x_data, data_seed_losses, color=GREEN, s=100, zorder=5)
    plt.scatter(x_data, both_seed_losses, color=LIGHT_BLUE, s=100, zorder=5)

    plt.xlabel("Ensemble member count $K$")
    plt.ylabel("Loss")
    plt.xticks(x_data, [f"{int(x)}" for x in x_data])
    plt.title("Ensembles with different sources of randomness")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("experiments/data_efficiency/plots/seed_science.png", bbox_inches="tight")
    plt.close()


def plot_token_scaling(best_run_dict, fit_type="power_law"):
    """Creates subplots showing loss vs token count for each model size with power law fits."""

    # Get unique model sizes and token counts
    token_counts = sorted(best_run_dict.keys())
    model_sizes = sorted(list(best_run_dict[token_counts[0]].keys()), key=lambda x: param_str_to_count[x])

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
                scaling_law = PowerScalingLaw(var_name="D")
                scaling_law.fit(x_values, y_values, p0=[1.0, 0.5, 2.0], bounds=([0, 0, 0], [np.inf, np.inf, np.inf]))
            elif fit_type == "reciprocal_law":
                reciprocal_law = ReciprocalScalingLaw(var_name="D")
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
    plt.savefig("experiments/data_efficiency/plots/token_scaling.png", bbox_inches="tight")
    plt.close()

    # Return asymptotes for reuse
    return valid_model_sizes, asymptotes


def plot_asymptotes_vs_model_size(model_sizes, asymptotes):
    """Creates a plot showing asymptotes vs model size using pre-fitted results."""

    # Create plot
    plt.figure(figsize=(10, 6), dpi=300)

    # Convert model sizes to parameter counts
    model_params_values = [param_str_to_count[model] for model in model_sizes]

    # Fit a power law to the asymptotes vs model parameters
    power_law = PowerScalingLaw(var_name="N")
    power_law.fit(model_params_values, asymptotes, p0=[1.0, 0.5, 2.0], bounds=([0, 0, 0], [np.inf, np.inf, np.inf]))
    x_fit = np.logspace(np.log10(min(model_params_values)), np.log10(max(model_params_values)), 100)
    y_fit = power_law.evaluate(x_fit)

    plt.scatter(model_params_values, asymptotes, color=PURPLE, s=100, zorder=5)
    plt.plot(x_fit, y_fit, "--", color=PURPLE, zorder=4, label=f"Fit: {power_law}")

    # Set labels and title
    plt.xscale("log")
    plt.xlabel("Parameter count (B)")
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
    plt.savefig("experiments/data_efficiency/plots/asymptotes_vs_model_size.png", bbox_inches="tight")
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
    model_names = sorted(list(best_run_dict[token_counts[0]].keys()), key=lambda x: param_str_to_count[x])

    for token_count in token_counts:
        for model_name in model_names:
            run = best_run_dict[token_count][model_name]
            model_sizes.append(param_str_to_count[model_name])  # In billions
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


def plot_standard_model_seed_scaling(parameter_scaling_losses, best_run_dict, fit_type="power_law"):
    y_lower = 2.6
    y_upper = 3.85
    """Plot 1: Varying model size, fixed token count"""

    # Get unique model sizes and token counts
    token_counts = sorted(best_run_dict.keys())
    model_sizes = sorted(list(best_run_dict[token_counts[0]].keys()), key=lambda x: param_str_to_count[x])

    # Create single plot
    plt.figure(figsize=SEED_SCALING_FIGSIZE, dpi=300)

    asymptotes = []
    for _, token_count in enumerate(token_counts):
        # Get data for this token count
        x_values = np.array([param_str_to_count[model] for model in model_sizes])
        y_values = np.array([best_run_dict[token_count][model]["final_dclm_loss"] for model in model_sizes])

        marker = token_count_marker_dict[token_count]
        token_counts_b = [token_count / 1_000_000_000 for token_count in token_counts]

        # Fit power law
        if fit_type == "power_law":
            scaling_law = PowerScalingLaw(var_name="N")
            scaling_law.fit(x_values, y_values, p0=[1.0, 0.5, 2.0], bounds=([0, 0, 0], [np.inf, np.inf, np.inf]))
        elif fit_type == "reciprocal_law":
            scaling_law = ReciprocalScalingLaw(var_name="N")
            scaling_law.fit(x_values, y_values, p0=[1.0, 2.0], bounds=([0, 0], [np.inf, np.inf]))

        # Generate points for smooth curve
        x_fit = np.logspace(np.log10(min(x_values)), np.log10(max(x_values)), 100)
        y_fit = scaling_law.evaluate(x_fit)

        # Plot data points and fitted curve
        for model_size, loss in zip(model_sizes, y_values, strict=False):
            plt.scatter(
                param_str_to_count[model_size],
                loss,
                color=param_str_color_dict[model_size],
                zorder=5,
                s=60,
                marker=marker,
            )

            if model_size == "1_4b4k" and token_count == 209715200:
                single_1_4b_loss = loss

        plt.plot(
            x_fit,
            y_fit,
            "--",
            color="gray",
            zorder=4,
            alpha=0.8,
            # label=f"{int(token_count_m)}M tokens (fit: {scaling_law})",
        )

        plt.scatter(
            [],
            [],
            s=100,
            zorder=5,
            color="gray",
            alpha=0.8,
            marker=marker,
            label=f"{value_pretty_name_dict[token_count]} tokens (fit: {scaling_law})",
        )

        asymptotes.append(scaling_law.asymptote())

    # Set scales and labels
    plt.xscale("log")
    plt.xlabel("Parameter count $N$")
    plt.ylabel("Loss")
    plt.title("Taking parameter count $N\\to\\infty$")
    plt.ylim(y_lower, y_upper)
    # Set x-ticks to model sizes
    model_params_values = [param_str_to_count[model] for model in model_sizes]
    plt.xticks(model_params_values, [value_pretty_name_dict.get(model, model) for model in model_sizes])

    # Add grid and legend
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.savefig("experiments/data_efficiency/plots/standard_seed_scaling_model_size.png", bbox_inches="tight")
    plt.close()

    """Plot 1b: Same but for standard tuning without regularization"""
    plt.figure(figsize=SEED_SCALING_FIGSIZE, dpi=300)
    for token_count, losses in parameter_scaling_losses.items():
        for param_str, loss in zip(PARAM_STRS, losses, strict=True):
            print(token_count, loss)
            plt.scatter(
                param_str_to_count[param_str],
                loss,
                marker=token_count_marker_dict[token_count],
                s=100,
                zorder=5,
                color=param_str_color_dict[param_str],
            )
        plt.plot(
            PARAM_COUNTS,
            losses,
            "--",
            color="gray",
            zorder=4,
            alpha=0.8,
        )
        plt.scatter(
            [],
            [],
            s=100,
            zorder=5,
            color="gray",
            alpha=0.8,
            marker=token_count_marker_dict[token_count],
            label=f"{value_pretty_name_dict[token_count]} tokens",
        )
    plt.xscale("log")
    plt.xlabel("Parameter count $N$")
    plt.ylabel("Loss")
    plt.title("Tuning parameter and epoch count")
    plt.legend()
    plt.xticks(PARAM_COUNTS, ["150M", "300M", "600M", "1.4B"])
    plt.xticks([], [], minor=True)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("experiments/data_efficiency/plots/baseline_seed_scaling.png", bbox_inches="tight")
    plt.close()

    """Plot 2: Varying seed token count, infinite model size"""

    plt.figure(figsize=SEED_SCALING_FIGSIZE, dpi=300)

    best_parameter_scaling_losses = pickle.load(
        open("experiments/data_efficiency/cache/best_parameter_scaling_losses.pkl", "rb")
    )
    epoched_losses = [min(losses) for losses in best_parameter_scaling_losses.values()]
    epoched_power_law = PowerScalingLaw(var_name="D")
    epoched_power_law.fit(
        token_counts_b, epoched_losses, p0=[1.0, 0.5, 2.0], bounds=([0, 0, 0], [np.inf, np.inf, np.inf])
    )
    x_fit = np.logspace(np.log10(min(token_counts_b)), np.log10(max(token_counts_b)), 100)
    y_fit = epoched_power_law.evaluate(x_fit)
    for i, token_count in enumerate(BASE_TOKEN_COUNTS):
        plt.scatter(
            token_count / 1_000_000_000,
            epoched_losses[i],
            marker=token_count_marker_dict[token_count],
            s=100,
            zorder=5,
            color=BASELINE_COLOR,
        )
    plt.plot(
        x_fit,
        y_fit,
        "--",
        color=BASELINE_COLOR,
        zorder=4,
        label=f"Standard recipe\n(Fit: {epoched_power_law})",
        alpha=0.8,
    )

    seed_token_power_law = PowerScalingLaw(var_name="D")
    seed_token_power_law.fit(
        token_counts_b, asymptotes, p0=[1.0, 0.5, 2.0], bounds=([0, 0, 0], [np.inf, np.inf, np.inf])
    )
    x_fit = np.logspace(np.log10(min(token_counts_b)), np.log10(max(token_counts_b)), 100)
    y_fit = seed_token_power_law.evaluate(x_fit)
    for i, token_count in enumerate(BASE_TOKEN_COUNTS):
        effective_data_epoching = back_out_data(epoched_losses[i], epoched_power_law)
        effective_data_regularized = back_out_data(asymptotes[i], epoched_power_law)
        print(effective_data_regularized / effective_data_epoching)
        plt.scatter(
            token_count / 1_000_000_000,
            asymptotes[i],
            marker=token_count_marker_dict[token_count],
            s=100,
            zorder=5,
            color=REGULARIZED_COLOR,
        )
    plt.plot(
        x_fit,
        y_fit,
        "--",
        color=REGULARIZED_COLOR,
        zorder=4,
        label=f"Regularized asymptotes\n(Fit: {seed_token_power_law})",
        alpha=0.8,
    )

    print(
        "Data efficiency of single 1.4B model:",
        back_out_data(single_1_4b_loss, epoched_power_law) / back_out_data(epoched_losses[0], epoched_power_law),
    )

    plt.xscale("log")
    plt.xlabel("Seed token count $D$")
    plt.ylabel("Loss")
    plt.title("Varying seed token count $D$")
    plt.legend()
    plt.ylim(y_lower, y_upper)
    plt.xticks(token_counts_b, ["209M", "419M", "839M", "1.67B"])
    plt.xticks([], [], minor=True)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("experiments/data_efficiency/plots/standard_seed_scaling_token_count.png", bbox_inches="tight")
    plt.close()

    return asymptotes, seed_token_power_law, epoched_power_law


def plot_ensemble_model_seed_scaling(standard_asymptotes, standard_power_law):
    y_lower = 2.6
    y_upper = 3.85
    y_lower_200M = 3.1
    y_upper_200M = 3.85
    """Plot 1: Varying ensemble member count, fixed model size and token count"""
    """1a: All token counts"""
    # Create plot
    plt.figure(figsize=SEED_SCALING_FIGSIZE, dpi=300)

    best_ensembles = pickle.load(
        open("experiments/data_efficiency/cache/varying-hparams-experiment_best_ensembles.pkl", "rb")
    )
    token_counts = list(best_ensembles[next(iter(best_ensembles.keys()))].keys())
    token_counts_b = [token_count / 1_000_000_000 for token_count in token_counts]

    for param_str, token_count_losses in best_ensembles.items():
        for token_count, (x_data, y_data, power_law) in token_count_losses.items():
            color = param_str_color_dict[param_str]
            marker = token_count_marker_dict[token_count]
            plt.scatter(x_data, y_data, s=100, color=color, marker=marker)
            x_fit = np.logspace(np.log10(min(x_data)), np.log10(max(x_data)), 100)
            y_fit = power_law.evaluate(x_fit)
            plt.plot(x_fit, y_fit, "--", color=color)

            if param_str == "1_4b4k" and token_count == 209715200:
                assert len(y_data) == 5
                five_1_4b_loss = y_data[4]

    for token_count in token_counts:
        marker = token_count_marker_dict[token_count]
        plt.scatter(
            [],
            [],
            s=100,
            zorder=5,
            color="gray",
            alpha=0.8,
            marker=marker,
            label=f"{value_pretty_name_dict[token_count]} tokens",
        )

    for param_str in best_ensembles.keys():
        color = param_str_color_dict[param_str]
        plt.scatter([], [], s=100, zorder=5, color=color, alpha=0.8, label=f"{value_pretty_name_dict[param_str]} params")

    plt.xscale("log")
    plt.xlabel("Ensemble member count $K$")
    plt.ylabel("Loss")
    plt.title("Taking member count $K\\to\\infty$")
    plt.xticks([1, 2, 3, 4, 5], [1, 2, 3, 4, 5])
    plt.xticks([], [], minor=True)
    plt.ylim(y_lower, y_upper)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("experiments/data_efficiency/plots/ensemble_seed_scaling_member_count.png", bbox_inches="tight")
    plt.close()

    """1b: 200M tokens"""
    plt.figure(figsize=SEED_SCALING_FIGSIZE, dpi=300)
    for param_str, token_count_losses in best_ensembles.items():
        for token_count, (x_data, y_data, power_law) in token_count_losses.items():
            color = param_str_color_dict[param_str]
            marker = token_count_marker_dict[token_count]
            plt.scatter(x_data, y_data, s=100, color=color, marker=marker, zorder=5)
            x_fit = np.logspace(np.log10(min(x_data)), np.log10(max(x_data)), 100)
            y_fit = power_law.evaluate(x_fit)
            plt.plot(
                x_fit,
                y_fit,
                "--",
                color=color,
                zorder=4,
                alpha=0.8,
                label=f"{value_pretty_name_dict[param_str]} params (Fit: {power_law})",
            )
            break  # don't plot past 200M tokens

    plt.xscale("log")
    plt.xlabel("Ensemble member count $K$")
    plt.ylabel("Loss")
    plt.title("Taking member count $K\\to\\infty$, 200M tokens")
    plt.xticks([1, 2, 3, 4, 5], [1, 2, 3, 4, 5])
    plt.xticks([], [], minor=True)
    plt.ylim(y_lower_200M, y_upper_200M)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("experiments/data_efficiency/plots/ensemble_seed_scaling_member_count_200M.png", bbox_inches="tight")
    plt.close()

    """Plot 2: Varying model size, infinite ensemble member count, fixed token count"""
    """2a: All token counts"""
    plt.figure(figsize=SEED_SCALING_FIGSIZE, dpi=300)

    tiered_asymptotes = []
    for token_count in token_counts:
        param_counts = []
        asymptotes = []
        for param_str, token_count_losses in best_ensembles.items():
            param_counts.append(param_str_to_count[param_str])
            asymptote = token_count_losses[token_count][2].asymptote()
            asymptotes.append(asymptote)

            plt.scatter(
                param_str_to_count[param_str],
                asymptote,
                s=100,
                color=param_str_color_dict[param_str],
                marker=token_count_marker_dict[token_count],
                zorder=5,
            )

        power_law = PowerScalingLaw(var_name="N")
        power_law.fit(param_counts, asymptotes, p0=[1.0, 0.5, 2.0], bounds=([0, 0, 0], [np.inf, np.inf, np.inf]))
        x_fit = np.logspace(np.log10(min(param_counts)), np.log10(max(param_counts)), 100)
        y_fit = power_law.evaluate(x_fit)
        plt.plot(
            x_fit,
            y_fit,
            "--",
            color="gray",
            zorder=4,
            alpha=0.8,
            # label=f"{value_pretty_name_dict[token_count]} tokens: {power_law}",
        )

        plt.scatter(
            [],
            [],
            s=100,
            zorder=5,
            color="gray",
            alpha=0.8,
            marker=token_count_marker_dict[token_count],
            label=f"{value_pretty_name_dict[token_count]} tokens (Fit: {power_law})",
        )

        tiered_asymptotes.append(power_law.asymptote())

    plt.xscale("log")
    plt.xlabel("Parameter count $N$ for $\\infty$ ensembles")
    plt.ylabel("Loss")
    plt.title("Taking parameter count $N\\to\\infty$")
    plt.xticks(param_counts, [value_pretty_name_dict[param_count] for param_count in param_counts])
    plt.xticks([], [], minor=True)
    plt.ylim(y_lower, y_upper)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("experiments/data_efficiency/plots/ensemble_seed_scaling_model_size.png", bbox_inches="tight")
    plt.close()

    """2b: 200M tokens"""
    plt.figure(figsize=SEED_SCALING_FIGSIZE, dpi=300)

    plt.axhline(y=3.74974, color=BASELINE_COLOR, linestyle="--", zorder=4, alpha=0.8, label="Standard recipe")
    plt.axhline(y=3.43, color=REGULARIZED_COLOR, linestyle="--", zorder=4, alpha=0.8, label="Regularized asymptote")

    for token_count in token_counts[:1]:
        param_counts = []
        asymptotes = []
        for param_str, token_count_losses in best_ensembles.items():
            param_counts.append(param_str_to_count[param_str])
            asymptote = token_count_losses[token_count][2].asymptote()
            asymptotes.append(asymptote)

            plt.scatter(
                param_str_to_count[param_str],
                asymptote,
                s=100,
                color=param_str_color_dict[param_str],
                marker=token_count_marker_dict[token_count],
                zorder=5,
            )

        power_law = PowerScalingLaw(var_name="N")
        power_law.fit(param_counts, asymptotes, p0=[1.0, 0.5, 2.0], bounds=([0, 0, 0], [np.inf, np.inf, np.inf]))
        x_fit = np.logspace(np.log10(min(param_counts)), np.log10(max(param_counts)), 100)
        y_fit = power_law.evaluate(x_fit)
        plt.plot(
            x_fit,
            y_fit,
            "--",
            color="gray",
            zorder=4,
            alpha=0.8,
            # label=f"{value_pretty_name_dict[token_count]} tokens: {power_law}",
        )

        plt.scatter(
            [],
            [],
            s=100,
            zorder=5,
            color="gray",
            alpha=0.8,
            marker=token_count_marker_dict[token_count],
            label=f"{value_pretty_name_dict[token_count]} tokens (Fit: {power_law})",
        )

    plt.xscale("log")
    plt.xlabel("Parameter count $N$ for $\\infty$ ensembles")
    plt.ylabel("Loss")
    plt.title("Taking parameter count $N\\to\\infty$, 200M tokens")
    plt.xticks(param_counts, [value_pretty_name_dict[param_count] for param_count in param_counts])
    plt.xticks([], [], minor=True)
    plt.ylim(y_lower_200M, y_upper_200M)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("experiments/data_efficiency/plots/ensemble_seed_scaling_model_size_200M.png", bbox_inches="tight")
    plt.close()

    """Plot 3: Varying seed token count, infinite model size and infinite ensemble member count"""

    plt.figure(figsize=SEED_SCALING_FIGSIZE, dpi=300)

    best_parameter_scaling_losses = pickle.load(
        open("experiments/data_efficiency/cache/best_parameter_scaling_losses.pkl", "rb")
    )
    epoched_losses = [min(losses) for losses in best_parameter_scaling_losses.values()]
    print("BEST EPOCHED LOSSES", epoched_losses)
    epoched_power_law = PowerScalingLaw(var_name="D")
    epoched_power_law.fit(
        token_counts_b, epoched_losses, p0=[1.0, 0.5, 2.0], bounds=([0, 0, 0], [np.inf, np.inf, np.inf])
    )
    x_fit = np.logspace(np.log10(min(token_counts_b)), np.log10(max(token_counts_b)), 100)
    y_fit = epoched_power_law.evaluate(x_fit)
    for i, token_count in enumerate(BASE_TOKEN_COUNTS):
        plt.scatter(
            token_count / 1_000_000_000,
            epoched_losses[i],
            marker=token_count_marker_dict[token_count],
            s=100,
            zorder=5,
            color=BASELINE_COLOR,
        )
    plt.plot(
        x_fit,
        y_fit,
        "--",
        color=BASELINE_COLOR,
        zorder=4,
        label=f"Standard recipe\n(Fit: {epoched_power_law})",
        alpha=0.8,
    )

    print("Standard asymptotes", standard_asymptotes)

    for i, (token_count, standard_asymptote) in enumerate(zip(token_counts, standard_asymptotes, strict=False)):
        effective_data_epoching = back_out_data(epoched_losses[i], epoched_power_law)
        effective_data_ensemble = back_out_data(standard_asymptote, epoched_power_law)
        print(effective_data_ensemble / effective_data_epoching)
        plt.scatter(
            token_count / 1_000_000_000,
            standard_asymptote,
            s=100,
            marker=token_count_marker_dict[token_count],
            zorder=5,
            color=REGULARIZED_COLOR,
        )

    x_fit = np.logspace(np.log10(min(token_counts_b)), np.log10(max(token_counts_b)), 100)
    y_fit = standard_power_law.evaluate(x_fit)
    plt.plot(
        x_fit,
        y_fit,
        "--",
        color=REGULARIZED_COLOR,
        zorder=4,
        label=f"Regularized asymptotes\n(Fit: {standard_power_law})",
    )

    for i, (token_count, asymptote) in enumerate(zip(token_counts, tiered_asymptotes, strict=False)):
        effective_data_epoching = back_out_data(epoched_losses[i], epoched_power_law)
        effective_data_ensemble = back_out_data(asymptote, epoched_power_law)
        print(effective_data_ensemble / effective_data_epoching)
        plt.scatter(
            token_count / 1_000_000_000,
            asymptote,
            s=100,
            marker=token_count_marker_dict[token_count],
            zorder=5,
            color=TIERED_COLOR,
        )

    tiered_power_law = PowerScalingLaw(var_name="D")
    tiered_power_law.fit(
        token_counts_b, tiered_asymptotes, p0=[1.0, 0.5, 2.0], bounds=([0, 0, 0], [np.inf, np.inf, np.inf])
    )
    x_fit = np.logspace(np.log10(min(token_counts_b)), np.log10(max(token_counts_b)), 100)
    y_fit = tiered_power_law.evaluate(x_fit)
    plt.plot(x_fit, y_fit, "--", color=TIERED_COLOR, zorder=4, label=f"Ensemble asymptotes\n(Fit: {tiered_power_law})")

    print(
        "Data efficiency of five 1.4B model:",
        back_out_data(five_1_4b_loss, epoched_power_law) / back_out_data(epoched_losses[0], epoched_power_law),
    )

    plt.xscale("log")
    plt.xlabel("Seed token count $D$")
    plt.ylabel("Loss")
    plt.title("Varying seed token count $D$")
    plt.xticks(token_counts_b, ["209M", "419M", "839M", "1.67B"])
    plt.xticks([], [], minor=True)
    plt.ylim(y_lower, y_upper)
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.savefig("experiments/data_efficiency/plots/ensemble_seed_scaling_token_count.png", bbox_inches="tight")
    plt.close()


def plot_token_scaling_simple(best_run_dict, fit_type="power_law"):
    """Creates a single plot showing loss vs token count for all model sizes with power law fits."""

    # Get unique model sizes and token counts
    token_counts = sorted(best_run_dict.keys())
    token_counts_m = [tc / 1_000_000 for tc in token_counts]
    model_sizes = sorted(list(best_run_dict[token_counts[0]].keys()), key=lambda x: param_str_to_count[x])

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
                scaling_law = PowerScalingLaw(var_name="D")
                scaling_law.fit(x_values, y_values, p0=[1.0, 0.5, 2.0], bounds=([0, 0, 0], [np.inf, np.inf, np.inf]))
            elif fit_type == "reciprocal_law":
                scaling_law = ReciprocalScalingLaw(var_name="D")
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
        scaling_law = PowerScalingLaw(var_name="D")
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
    plt.savefig("experiments/data_efficiency/plots/token_scaling_simple.png", bbox_inches="tight")
    plt.close()


def plot_benchmark_results():
    from experiments.data_efficiency.eval_200m_models import VANILLA_MODEL_SCALING, ENSEMBLE_MODEL_SCALING, get_base_name
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

    plt.figure(figsize=WIDE_RECTANGLE_FIGSIZE, dpi=300)

    model_x_values = [param_str_to_count[model.model_size] for model in VANILLA_MODEL_SCALING]
    chinchilla_accs = [0.5435, 0.5534, 0.5342, 0.5414]
    model_y_values = [1.0 - acc for acc in chinchilla_accs]
    plt.plot(model_x_values, model_y_values, color=BASELINE_COLOR, label="Epoched recipe", marker="o")

    model_x_values = [param_str_to_count[model.model_size] for model in VANILLA_MODEL_SCALING]
    model_y_values = [1.0 - benchmark_results[get_base_name(model)]["1"]["avg_acc"] for model in VANILLA_MODEL_SCALING]
    plt.plot(model_x_values, model_y_values, color=PURPLE, label="Model scaling", marker="o")
    for model in ENSEMBLE_MODEL_SCALING:
        model_x_values = [param_str_to_count[model.model_size] * (seed + 1) for seed in range(model.num_seeds)]
        model_y_values = [
            1.0 - benchmark_results[get_base_name(model)][str(seed + 1)]["avg_acc"] for seed in range(model.num_seeds)
        ]
        plt.plot(
            model_x_values,
            model_y_values,
            label=f"{value_pretty_name_dict[model.model_size]} ensembles",
            marker="o",
            color=param_str_color_dict[model.model_size],
        )

    with open("experiments/data_efficiency/distill_benchmarks.json", "r") as f:
        distill_benchmarks = json.load(f)

    key_to_pretty_map = {
        "distill-8ens": "8-Ensemble distill",
        "self-distill": "Self-distill",
    }

    key_to_color_map = {
        "distill-8ens": ENSEMBLE_COLOR,
        "self-distill": SELF_DISTILL_COLOR,
    }

    for k in ["self-distill", "distill-8ens"]:
        v = distill_benchmarks[k]
        v["avg_acc"] = np.mean([v[task]["acc"] for task in v])
        plt.scatter(
            0.3,
            1.0 - v["avg_acc"],
            color=key_to_color_map[k],
            marker="*",
            s=120,
            zorder=6,
            label=f"{key_to_pretty_map[k]}",
        )

    plt.legend()
    plt.grid(True, which="both", linestyle="--", alpha=0.3)
    plt.xscale("log")
    plt.xlabel("Total parameter count")
    plt.ylabel("Average Error")
    plt.xlim(right=14.0)
    plt.title("Downstream benchmark error")
    plt.savefig("experiments/data_efficiency/plots/benchmark_results.png", bbox_inches="tight")
    plt.close()


def plot_simple_cross(best_asymptote_ensemble_200M, power_law_200M, run_losses_200M):
    x_data, y_data, power_law = best_asymptote_ensemble_200M
    plt.figure(figsize=WIDE_RECTANGLE_FIGSIZE, dpi=300)
    max_x = max(x_data * param_str_to_count["300m4k"] * 2)

    plt.scatter(PARAM_COUNTS, run_losses_200M, color=PURPLE, s=50)

    # fit a power law to the model scaling
    model_x_fit = np.linspace(min(PARAM_COUNTS), max_x, 5000)
    model_y_fit = power_law_200M.evaluate(model_x_fit)
    plt.plot(model_x_fit, model_y_fit, "--", color=PURPLE, label=f"Regularized recipe (Fit: {power_law_200M})")

    model_size = "300m4k"

    x_fit = np.linspace(min(x_data * param_str_to_count[model_size]), max_x, 5000)
    y_fit = power_law.evaluate(x_fit / param_str_to_count[model_size])
    plt.scatter(x_data * param_str_to_count[model_size], y_data, color=param_str_color_dict[model_size])
    plt.plot(
        x_fit,
        y_fit,
        "--",
        color=param_str_color_dict[model_size],
        label=f"{value_pretty_name_dict[model_size]} ensembles (Fit: {power_law})",
    )

    # Add black arrow to legend
    from matplotlib.lines import Line2D

    arrow_legend = Line2D(
        [0], [0], marker="$\u2192$", color="black", linestyle="None", markersize=12, label="Distillation", alpha=0.6
    )

    # Get existing legend handles and labels, then add the arrow
    handles, labels = plt.gca().get_legend_handles_labels()
    handles.insert(2, arrow_legend)
    labels.insert(2, "Distilling from teacher to student")

    # plt.legend(handles=handles, labels=labels)
    plt.xscale("log")
    plt.xticks(PARAM_COUNTS, [value_pretty_name_dict[param_count] for param_count in PARAM_COUNTS])
    plt.xticks([], [], minor=True)
    plt.xlabel("Total parameter count")
    plt.ylabel("DCLM Loss")
    plt.grid(True, which="both", linestyle="--", alpha=0.3)
    # plt.title("Distilling a 300M student (200M seed tokens)")
    plt.savefig("experiments/data_efficiency/plots/simple_cross.png", bbox_inches="tight")
    plt.close()


def plot_distillation(best_asymptote_ensemble_200M, power_law_200M, run_losses_200M):
    x_data, y_data, power_law = best_asymptote_ensemble_200M
    plt.figure(figsize=WIDE_RECTANGLE_FIGSIZE, dpi=300)
    max_x = max(x_data * param_str_to_count["300m4k"] * 2)

    plt.scatter(PARAM_COUNTS, run_losses_200M, color=PURPLE, s=50)

    # fit a power law to the model scaling
    model_x_fit = np.linspace(min(PARAM_COUNTS), max_x, 5000)
    model_y_fit = power_law_200M.evaluate(model_x_fit)
    plt.plot(model_x_fit, model_y_fit, "--", color=PURPLE, label=f"Regularized recipe (Fit: {power_law_200M})")

    model_size = "300m4k"

    x_fit = np.linspace(min(x_data * param_str_to_count[model_size]), max_x, 5000)
    y_fit = power_law.evaluate(x_fit / param_str_to_count[model_size])
    plt.scatter(x_data * param_str_to_count[model_size], y_data, color=param_str_color_dict[model_size])
    plt.plot(
        x_fit,
        y_fit,
        "--",
        color=param_str_color_dict[model_size],
        label=f"{value_pretty_name_dict[model_size]} ensembles (Fit: {power_law})",
    )

    # Best 8-mixture distill run: 300m4k-209Mx16-dclm+ens8x0730^0.9-cos-lr0.0030-wd0.10-bs64 (3.3635)
    plt.annotate(
        "",
        xytext=(0.3 * x_data[-1], y_data[-1]),
        xy=(0.3, 3.3635),
        arrowprops=dict(
            arrowstyle="->",
            color=ENSEMBLE_COLOR,
            linewidth=1.5,
            alpha=0.6,
            connectionstyle="arc3,rad=0.0",
            mutation_scale=15,
        ),
        zorder=0,
    )
    plt.scatter(
        0.3,
        3.3635,
        color=ENSEMBLE_COLOR,
        # edgecolors="black",
        linewidths=0.8,
        marker="*",
        s=150,
        zorder=6,
        label="8-ensemble distill: 3.36",
    )

    # Best self-distill run: 300m4k-209Mx16-dclm+sd0805^0.75-cos-lr0.0030-wd0.10-bs64 (3.43243)
    plt.annotate(
        "",
        xytext=(0.3 * x_data[0], y_data[0]),
        xy=(0.3, 3.43243),
        arrowprops=dict(
            arrowstyle="->",
            color=SELF_DISTILL_COLOR,
            linewidth=1.5,
            alpha=0.6,
            connectionstyle="arc3,rad=0.15",
            mutation_scale=15,
        ),
        zorder=0,
    )
    plt.scatter(
        0.3,
        3.43243,
        color=SELF_DISTILL_COLOR,
        # edgecolors="black",
        linewidths=0.8,
        marker="*",
        s=150,
        zorder=6,
        label="Self-distill: 3.43",
    )

    plt.axvline(x=0.3, color="grey", linestyle=":", alpha=0.7, zorder=0, linewidth=2.0)

    # Add black arrow to legend
    from matplotlib.lines import Line2D

    arrow_legend = Line2D(
        [0], [0], marker="$\u2192$", color="black", linestyle="None", markersize=12, label="Distillation", alpha=0.6
    )

    # Get existing legend handles and labels, then add the arrow
    handles, labels = plt.gca().get_legend_handles_labels()
    handles.insert(2, arrow_legend)
    labels.insert(2, "Distilling from teacher to student")

    plt.legend(handles=handles, labels=labels)
    plt.xscale("log")
    plt.xticks(PARAM_COUNTS, [value_pretty_name_dict[param_count] for param_count in PARAM_COUNTS])
    plt.xticks([], [], minor=True)
    plt.xlabel("Total parameter count")
    plt.ylabel("DCLM Loss")
    plt.grid(True, which="both", linestyle="--", alpha=0.3)
    plt.title("Distilling a 300M student (200M seed tokens)")
    plt.savefig("experiments/data_efficiency/plots/distillation.png", bbox_inches="tight")
    plt.close()


def plot_200M_sample(
    losses_for_200M, power_law_200M, run_losses_200M, best_single_model_hparams_ensemble_200M, epoched_data_scaling_law
):
    # Figure 1
    with plt.rc_context({"font.size": 12}):
        plt.figure(figsize=WIDE_RECTANGLE_FIGSIZE, dpi=300)
        param_counts = [param_str_to_count[model_size] for model_size in losses_for_200M.keys()]
        min_x = min(param_counts)
        max_x = max(
            [max(x_data * param_str_to_count[model_size]) for model_size, (x_data, _, _) in losses_for_200M.items()]
        )

        chinchilla_losses = [3.83752, 3.78538, 3.74974, 3.76432]
        plt.scatter(param_counts, chinchilla_losses, color=BASELINE_COLOR, s=50)
        plt.plot(param_counts, chinchilla_losses, color=BASELINE_COLOR, label="Epoched recipe")

        plt.scatter(param_counts, run_losses_200M, color=PURPLE, s=50)
        model_x_fit = np.linspace(min_x, max_x * 25, 5000)
        model_y_fit = power_law_200M.evaluate(model_x_fit)
        plt.plot(model_x_fit, model_y_fit, "--", color=PURPLE, label="Model scaling")

        # # add a shaded gray region for any y value below asymptote
        # plt.fill_between(
        #     model_x_fit,
        #     3.1,
        #     power_law_200M.asymptote(),
        #     color=PURPLE,
        #     alpha=0.2,
        #     label="Impossible with standard scaling, infinite compute",
        # )

        for model_size, (x_data, y_data, power_law) in losses_for_200M.items():
            x_fit = np.linspace(min(x_data * param_str_to_count[model_size]), max_x * 25, 5000)
            y_fit = power_law.evaluate(x_fit / param_str_to_count[model_size])
            plt.scatter(x_data * param_str_to_count[model_size], y_data, s=50, color=param_str_color_dict[model_size])
            plt.plot(
                x_fit,
                y_fit,
                "--",
                color=param_str_color_dict[model_size],
                label=f"{value_pretty_name_dict[model_size]} ensembles",
            )

        key_to_pretty_map = {
            "distill-8ens": "8-Ensemble distill",
            "self-distill": "Self-distill",
        }

        key_to_color_map = {
            "distill-8ens": ENSEMBLE_COLOR,
            "self-distill": SELF_DISTILL_COLOR,
        }

        # Best 8-mixture distill run: 300m4k-209Mx16-dclm+ens8x0730^0.9-cos-lr0.0030-wd0.10-bs64 (3.3635)
        # Best self-distill run: 300m4k-209Mx16-dclm+sd0805^0.75-cos-lr0.0030-wd0.10-bs64 (3.43243)
        plt.scatter(
            0.3,
            3.43243,
            color=key_to_color_map["self-distill"],
            marker="*",
            s=120,
            zorder=6,
            label=f"{key_to_pretty_map['self-distill']}",
        )

        plt.scatter(
            0.3,
            3.3635,
            color=key_to_color_map["distill-8ens"],
            marker="*",
            s=120,
            zorder=6,
            label=f"{key_to_pretty_map['distill-8ens']}",
        )

        plt.legend(loc="upper right")
        plt.grid(True, which="both", linestyle="--", alpha=0.3)
        plt.xscale("log")
        plt.xlabel("Total parameter count")
        plt.ylabel("DCLM Loss")
        plt.xlim(right=14.0)
        plt.title("Validation loss")
        plt.savefig("experiments/data_efficiency/plots/200M_sample.png", bbox_inches="tight")
        plt.close()

    # Figure 2: Comparing parameter count and single model hparams ensemble
    plt.figure(figsize=(6, 4), dpi=300)
    xmax_multiplier = 1
    model_x_fit = np.linspace(min_x, max_x * xmax_multiplier, 5000)
    model_y_fit = power_law_200M.evaluate(model_x_fit)
    plt.scatter(param_counts, run_losses_200M, color=REGULARIZED_COLOR, s=50, zorder=5)
    plt.plot(
        model_x_fit,
        model_y_fit,
        "--",
        color=REGULARIZED_COLOR,
        label=f"Model scaling: (Fit: {power_law_200M})",
        zorder=4,
        alpha=0.8,
    )

    for model_size, (x_data, y_data, power_law) in [("300m4k", best_single_model_hparams_ensemble_200M[1:])]:
        x_fit = np.linspace(min(x_data * param_str_to_count[model_size]), max_x * xmax_multiplier, 5000)
        y_fit = power_law.evaluate(x_fit / param_str_to_count[model_size])
        plt.scatter(
            x_data * param_str_to_count[model_size], y_data, s=50, color=param_str_color_dict[model_size], zorder=5
        )
        print(x_data, y_data)
        plt.plot(
            x_fit,
            y_fit,
            "--",
            color=param_str_color_dict[model_size],
            label=f"{value_pretty_name_dict[model_size]} ensembles (Fit: {power_law})",
            zorder=4,
            alpha=0.8,
        )

    plt.legend()
    # plt.grid(True, alpha=0.3)
    plt.xscale("log")
    plt.xticks(param_counts, ["150M", "300M", "600M", "1.4B"])
    plt.xticks([], [], minor=True)
    plt.xlabel("Total parameter count")
    plt.ylabel("Loss")
    plt.title("Ensemble member scaling")
    plt.grid(True, which="both", linestyle="--", alpha=0.3)
    plt.savefig("experiments/data_efficiency/plots/200M_ensemble_vs_parameter_scaling.png", bbox_inches="tight")
    plt.close()

    # Figure 3 (new figure 1 (8/22))
    fig, ax1 = plt.subplots(figsize=(6, 5), dpi=300)
    xmax_multiplier = 5
    chinchilla_losses = [3.83752, 3.78538, 3.74974, 3.76432]
    ax1.scatter(param_counts, chinchilla_losses, color=BASELINE_COLOR, s=50)
    ax1.plot(param_counts, chinchilla_losses, color=BASELINE_COLOR, label="Standard recipe")

    ax1.scatter(param_counts, run_losses_200M, color=REGULARIZED_COLOR, s=50, zorder=5)
    model_x_fit = np.linspace(min(x_data * param_str_to_count["150m4k"]), max_x * xmax_multiplier, 5000)
    model_y_fit = power_law_200M.evaluate(model_x_fit)
    ax1.plot(
        model_x_fit,
        model_y_fit,
        "--",
        color=REGULARIZED_COLOR,
        label=f"Regularized recipe\n(Fit: {power_law_200M})",
        zorder=4,
        alpha=0.8,
    )

    for model_size, (x_data, y_data, power_law) in [("300m4k", best_single_model_hparams_ensemble_200M[1:])]:
        x_fit = np.linspace(min(x_data * param_str_to_count[model_size]), max_x * xmax_multiplier, 5000)
        y_fit = power_law.evaluate(x_fit / param_str_to_count[model_size])
        ax1.scatter(
            x_data * param_str_to_count[model_size], y_data, s=50, color=param_str_color_dict[model_size], zorder=5
        )
        print(x_data, y_data)
        ax1.plot(
            x_fit,
            y_fit,
            "--",
            color=param_str_color_dict[model_size],
            label=f"Ensembling recipe\n(Fit: {power_law})",
            zorder=4,
            alpha=0.8,
        )

    infinite_compute_asymptote = 3.17

    ax1.axhline(y=min(chinchilla_losses), color=BASELINE_COLOR, linestyle=":", zorder=3, alpha=0.5)
    ax1.axhline(y=power_law_200M.asymptote(), color=REGULARIZED_COLOR, linestyle=":", zorder=3, alpha=0.5)
    ax1.axhline(y=power_law.asymptote(), color=param_str_color_dict[model_size], linestyle=":", zorder=3, alpha=0.5)
    ax1.axhline(
        y=infinite_compute_asymptote,
        linestyle=":",
        label="Joint scaling recipe\nasymptote ($N,K\\to\\infty$)",
        color=TIERED_COLOR,
        zorder=3,
    )

    def data_efficiency_from_loss(loss):
        return back_out_data(loss, epoched_data_scaling_law) / back_out_data(
            min(chinchilla_losses), epoched_data_scaling_law
        )

    def loss_from_data_efficiency(data_efficiency):
        effective_data = data_efficiency * back_out_data(min(chinchilla_losses), epoched_data_scaling_law)
        return epoched_data_scaling_law.evaluate(effective_data)

    ax1.legend(framealpha=1.0)

    ax1.set_xscale("log")
    ax1.set_xticks(param_counts)
    ax1.set_xticklabels(["150M", "300M", "600M", "1.4B"])
    ax1.tick_params(axis="x", which="minor", bottom=False)
    ax1.set_xlim(right=2 * xmax_multiplier)
    ax1.set_xlabel("Total parameter count")
    ax1.set_ylabel("DCLM validation loss")
    ax1.set_title("Comparing scaling recipes with no compute constraints")

    # Set the secondary axis limits using the formula 1/loss + 5
    secax = ax1.secondary_yaxis("right", functions=(data_efficiency_from_loss, loss_from_data_efficiency))
    secax.set_ylabel("Data efficiency")
    important_losses = [
        min(chinchilla_losses),
        power_law_200M.asymptote(),
        power_law.asymptote(),
        infinite_compute_asymptote,
    ]
    colors = [BASELINE_COLOR, REGULARIZED_COLOR, param_str_color_dict[model_size], TIERED_COLOR]
    secax.set_yticks(
        [data_efficiency_from_loss(loss) for loss in important_losses],
        [f"${data_efficiency_from_loss(loss):.2f}\\times$" for loss in important_losses],
    )

    for tick, color in zip(secax.yaxis.get_ticklabels(), colors, strict=False):
        tick.set_color(color)

    plt.savefig("experiments/data_efficiency/plots/figure_1_8_22.png", bbox_inches="tight")
    plt.close()

    # Figure 3: Marginal returns
    plt.figure(figsize=(8, 5), dpi=300)
    # plot AB/(A+Ce^Bx) for the same x values
    x_fit = np.logspace(np.log10(0.00001), np.log10(max_x * 25), 5000)
    A, B, C = power_law_200M.params
    plt.plot(
        x_fit,
        [A * B / (A + C * np.exp(B * np.log(x))) for x in x_fit],
        "--",
        color=PURPLE,
        label=f"Model scaling: (Fit: {A:.2f}/x^{B:.2f} + {C:.2f})",
    )
    for model_size, (x_data, _, power_law) in losses_for_200M.items():
        A, B, C = power_law.params
        x_fit = np.logspace(np.log10(min(x_data * param_str_to_count[model_size]) * 0.0001), np.log10(max_x * 25), 5000)
        y_fit = A * B / (A + C * np.exp(B * np.log(x_fit / param_str_to_count[model_size])))
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
    plt.savefig("experiments/data_efficiency/plots/200M_sample_AB_A_Ce_Bx.png", bbox_inches="tight")
    plt.close()


def plot_batch_size_ablation(run_list):
    run_list = sorted(run_list, key=lambda x: x["batch_size"])
    batch_sizes = sorted(list(set([run["batch_size"] for run in run_list])))
    losses = [run["final_dclm_loss"] for run in run_list]
    plt.figure(figsize=ABLATION_FIGSIZE, dpi=300)
    plt.plot(batch_sizes, losses, marker="o", color=BASELINE_COLOR)
    plt.xlabel("Batch size")
    plt.ylabel("Loss")
    plt.xscale("log")
    plt.xticks(batch_sizes, [f"{x}" for x in batch_sizes])
    plt.xticks([], [], minor=True)
    plt.title("Batch size")
    plt.grid(True, which="both", linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig("experiments/data_efficiency/plots/ablation_batch_size.png", bbox_inches="tight")
    plt.close()


def plot_wd_overfitting_ablation(run_list):
    run_list_map = {
        wd: [
            run
            for run in run_list
            if run["batch_size"] == 64
            and run["data_name"] == "dclm"
            and run["weight_decay"] == wd
            and run["model_name"] == "300m4k"
            and run["base_tokens"] == 209715200
            and run["lr"] == 3e-3
            and "seed" not in run["run_id"]
            and "-ts" not in run["run_id"]
        ]
        for wd in [0.1, 1.6]
    }

    unique_epochs = sorted(list(set([run["epochs"] for run in run_list])))
    run_list_map[0.1] = sorted(run_list_map[0.1], key=lambda x: x["epochs"])
    run_list_map[1.6] = sorted(run_list_map[1.6], key=lambda x: x["epochs"])

    # loss figure
    low_wd_losses = [run["final_dclm_loss"] for run in run_list_map[0.1]]
    high_wd_losses = [run["final_dclm_loss"] for run in run_list_map[1.6]]
    plt.figure(figsize=ABLATION_FIGSIZE, dpi=300)
    plt.plot(unique_epochs, low_wd_losses, marker="o", color=BASELINE_COLOR, label="Weight decay 0.1")
    plt.plot(unique_epochs, high_wd_losses, marker="o", color=REGULARIZED_COLOR, label="Weight decay 1.6")
    plt.ylim(3.5, 5.6)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.xscale("log")
    plt.xticks(unique_epochs, [f"{x}" for x in unique_epochs])
    plt.xticks([], [], minor=True)
    plt.title("Weight decay overfitting (Loss)")
    plt.legend()
    plt.grid(True, which="both", linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig("experiments/data_efficiency/plots/ablation_wd_overfitting_loss.png", bbox_inches="tight")

    # benchmark figure
    with open("./experiments/data_efficiency/wd_overfitting_results.json", "r") as f:
        all_results = json.load(f)

    # Calculate average accuracy across 3 benchmarks for each configuration
    wd_01_avg_acc = []
    wd_16_avg_acc = []

    for r01, r16 in zip(run_list_map[0.1], run_list_map[1.6], strict=False):
        # WD 0.1 models
        base_name_01 = r01["run_id"].split("-bs")[0]
        results_01 = all_results[base_name_01]["1"]
        avg_acc_01 = (results_01["arc_easy"]["acc"] + results_01["piqa"]["acc"] + results_01["sciq"]["acc"]) / 3
        wd_01_avg_acc.append(avg_acc_01)

        # WD 1.6 models
        base_name_16 = r16["run_id"].split("-bs")[0]
        results_16 = all_results[base_name_16]["1"]
        avg_acc_16 = (results_16["arc_easy"]["acc"] + results_16["piqa"]["acc"] + results_16["sciq"]["acc"]) / 3
        wd_16_avg_acc.append(avg_acc_16)

    wd_01_avg_err = [1 - acc for acc in wd_01_avg_acc]
    wd_16_avg_err = [1 - acc for acc in wd_16_avg_acc]
    # Create the plot
    plt.figure(figsize=ABLATION_FIGSIZE, dpi=300)
    plt.plot(unique_epochs, wd_01_avg_err, marker="o", label="Weight decay 0.1", color=BASELINE_COLOR)
    plt.plot(unique_epochs, wd_16_avg_err, marker="o", label="Weight decay 1.6", color=REGULARIZED_COLOR)

    plt.xlabel("Epochs")
    plt.ylabel("Average error")
    plt.title("Weight decay overfitting (Downstream benchmarks)")
    plt.legend()
    plt.grid(True, which="both", linestyle="--", alpha=0.3)
    plt.xscale("log")
    plt.xticks(unique_epochs, [f"{x}" for x in unique_epochs])
    plt.xticks([], [], minor=True)
    plt.tight_layout()
    plt.savefig("experiments/data_efficiency/plots/ablation_wd_overfitting_benchmark.png", bbox_inches="tight")


def plot_lr_tuned_epoch_ablation(run_list):
    run_list = [
        run
        for run in run_list
        if run["batch_size"] == 64
        and "seed" not in run["run_id"]
        and "-ts" not in run["run_id"]
        and run["data_name"] == "dclm"
        and run["weight_decay"] == 0.1
        and run["model_name"] == "300m4k"
        and run["base_tokens"] == 209715200
    ]

    unique_epochs = sorted(list(set([run["epochs"] for run in run_list])))
    losses = []
    train_losses = []
    for epoch in unique_epochs:
        run_list_epoch = [run for run in run_list if run["epochs"] == epoch]
        best_run = sorted(run_list_epoch, key=lambda x: x["final_dclm_loss"])[0]
        print(best_run["run_id"])
        losses.append(best_run["final_dclm_loss"])
        train_losses.append(best_run["final_train_loss"])
    plt.figure(figsize=ABLATION_FIGSIZE, dpi=300)
    plt.plot(unique_epochs, losses, marker="o", color=BASELINE_COLOR, label="Validation loss")
    plt.ylim(3.7, 5.05)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.xscale("log")
    plt.xticks(unique_epochs, [f"{x}" for x in unique_epochs])
    plt.xticks([], [], minor=True)
    plt.title("Increasing epoch count")
    plt.grid(True, which="both", linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig("experiments/data_efficiency/plots/ablation_lr_tuned_epoch.png", bbox_inches="tight")

    plt.plot(unique_epochs, train_losses, marker="o", color=GREEN, label="Train loss")
    plt.ylim(bottom=2.0)
    plt.title("Increasing epoch count train loss")
    plt.legend()
    plt.savefig("experiments/data_efficiency/plots/train_loss_ablation_epoch_overfitting.png", bbox_inches="tight")
    plt.close()


def cache_parameter_scaling_losses(run_list):
    parameter_scaling_losses = {}
    print(run_list)
    for base_token_count in BASE_TOKEN_COUNTS:
        parameter_scaling_losses[base_token_count] = []
        for model_size in PARAM_STRS:
            curr_run_list = [
                run
                for run in run_list
                if run["batch_size"] == 64
                and "seed" not in run["run_id"]
                and "-ts" not in run["run_id"]
                and run["data_name"] == "dclm"
                and run["weight_decay"] == 0.1
                and run["model_name"] == model_size
                and run["base_tokens"] == base_token_count
            ]
            curr_run_list = sorted(curr_run_list, key=lambda x: x["final_dclm_loss"])
            print(f"{model_size} {base_token_count} {curr_run_list[0]['run_id']}: {curr_run_list[0]['final_dclm_loss']}")
            parameter_scaling_losses[base_token_count].append(curr_run_list[0]["final_dclm_loss"])
    pickle.dump(
        parameter_scaling_losses, open("experiments/data_efficiency/cache/best_parameter_scaling_losses.pkl", "wb")
    )
    return parameter_scaling_losses


def sensitivity_analysis(run_list):
    """Plot 1: Sensitivity analysis for regularized parameter scaling"""
    power_law_200M, _ = pickle.load(open("experiments/data_efficiency/cache/standard_asymptotes_200M.pkl", "rb"))
    seed1_losses = []
    seed2_losses = []
    for model_size in PARAM_STRS:
        seed1_run_list = [
            run
            for run in run_list
            if run["batch_size"] == 64
            and "seed1" in run["run_id"]
            and run["data_name"] == "dclm"
            and run["model_name"] == model_size
            and run["base_tokens"] == 209715200
        ]
        seed1_run_list = sorted(seed1_run_list, key=lambda x: x["final_dclm_loss"])
        print(f"{model_size}: {seed1_run_list[0]['final_dclm_loss']}")
        seed1_losses.append(seed1_run_list[0]["final_dclm_loss"])

        seed2_run_list = [
            run
            for run in run_list
            if run["batch_size"] == 64
            and "seed2" in run["run_id"]
            and run["data_name"] == "dclm"
            and run["model_name"] == model_size
            and run["base_tokens"] == 209715200
        ]
        seed2_run_list = sorted(seed2_run_list, key=lambda x: x["final_dclm_loss"])
        print(f"{model_size}: {seed2_run_list[0]['final_dclm_loss']}")
        seed2_losses.append(seed2_run_list[0]["final_dclm_loss"])

    seed1_power_law = PowerScalingLaw(var_name="N")
    seed1_power_law.fit(PARAM_COUNTS, seed1_losses, p0=[1.0, 0.5, 2.0], bounds=([0, 0, 0], [np.inf, np.inf, np.inf]))

    seed2_power_law = PowerScalingLaw(var_name="N")
    seed2_power_law.fit(PARAM_COUNTS, seed2_losses, p0=[1.0, 0.5, 2.0], bounds=([0, 0, 0], [np.inf, np.inf, np.inf]))

    x_fit = np.logspace(np.log10(min(PARAM_COUNTS)), np.log10(max(PARAM_COUNTS)), 100)

    plt.figure(figsize=ABLATION_FIGSIZE, dpi=300)

    plt.scatter(PARAM_COUNTS, power_law_200M.evaluate(PARAM_COUNTS), marker="o", color=REGULARIZED_COLOR)
    plt.plot(
        x_fit,
        power_law_200M.evaluate(x_fit),
        "--",
        color=REGULARIZED_COLOR,
        label=f"Main body power law (Fit: {power_law_200M})",
    )

    plt.scatter(PARAM_COUNTS, seed1_losses, marker="o", color=LIGHT_BLUE)
    plt.plot(
        x_fit,
        seed1_power_law.evaluate(x_fit),
        "--",
        color=LIGHT_BLUE,
        label=f"Seed 1 power law (Fit: {seed1_power_law})",
    )

    plt.scatter(PARAM_COUNTS, seed2_losses, marker="o", color=GREEN)
    plt.plot(
        x_fit, seed2_power_law.evaluate(x_fit), "--", color=GREEN, label=f"Seed 2 power law (Fit: {seed2_power_law})"
    )

    plt.xscale("log")
    plt.xticks(PARAM_COUNTS, [value_pretty_name_dict[param_count] for param_count in PARAM_COUNTS])
    plt.xticks([], [], minor=True)
    plt.xlabel("Parameter count")
    plt.ylabel("Loss")
    plt.title("Sensitivity analysis for regularized parameter scaling")
    plt.legend()
    plt.grid(True, which="both", linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig("experiments/data_efficiency/plots/sensitivity_analysis_regularized.png", bbox_inches="tight")
    plt.close()

    """Plot 2: Sensitivity analysis for ensembling"""
    x_data, y_data, power_law = pickle.load(
        open("experiments/data_efficiency/cache/varying-hparams-experiment_best_asymptote_ensemble_200M.pkl", "rb")
    )
    num_subsampled = 4
    half_x_data = x_data[:num_subsampled]
    half_y_data = y_data[:num_subsampled]
    half_power_law = PowerScalingLaw(var_name="K")
    half_power_law.fit(half_x_data, half_y_data, p0=[1.0, 0.5, 2.0], bounds=([0, 0, 0], [np.inf, np.inf, np.inf]))
    half_x_fit = np.logspace(np.log10(min(half_x_data)), np.log10(max(half_x_data)), 100)
    x_fit = np.logspace(np.log10(min(x_data)), np.log10(max(x_data)), 100)

    plt.figure(figsize=ABLATION_FIGSIZE, dpi=300)

    plt.scatter(x_data, y_data, marker="o", color=param_str_color_dict["300m4k"])
    plt.plot(
        x_fit,
        power_law.evaluate(x_fit),
        "--",
        color=param_str_color_dict["300m4k"],
        label=f"Using 8 runs (Fit: {power_law})",
    )

    plt.scatter(half_x_data, half_y_data, marker="o", color=PURPLE)
    plt.plot(
        half_x_fit,
        half_power_law.evaluate(half_x_fit),
        "--",
        color=PURPLE,
        label=f"Using 4 runs (Fit: {half_power_law})",
    )
    plt.legend()
    plt.xscale("log")
    plt.xlabel("Ensemble member count $K$")
    plt.ylabel("Loss")
    plt.title("Subsampling for ensembling")
    plt.xticks([1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 3, 4, 5, 6, 7, 8])
    plt.xticks([], [], minor=True)
    plt.grid(True, which="both", linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig("experiments/data_efficiency/plots/sensitivity_analysis_ensembling.png", bbox_inches="tight")
    plt.close()


def no_convex_tuning_ablation(run_list):
    parameter_scaling_losses = pickle.load(
        open("experiments/data_efficiency/cache/best_parameter_scaling_losses.pkl", "rb")
    )
    _, run_losses_200M = pickle.load(open("experiments/data_efficiency/cache/standard_asymptotes_200M.pkl", "rb"))
    valid_runs = [
        run
        for run in run_list
        if run["batch_size"] == 64
        and "seed" not in run["run_id"]
        and "-ts" not in run["run_id"]
        and run["data_name"] == "dclm"
        and run["base_tokens"] == 209715200
        and run["lr"] in [1e-4, 3e-4, 1e-3, 3e-3, 1e-2]
    ]
    runs_150M = [run for run in valid_runs if run["model_name"] == "150m4k"]

    runs_150M = sorted(runs_150M, key=lambda x: x["final_dclm_loss"])
    best_150M_weight_decay = runs_150M[0]["weight_decay"]
    best_150M_lr = runs_150M[0]["lr"]
    best_150M_epochs = runs_150M[0]["epochs"]

    print(best_150M_weight_decay, best_150M_lr, best_150M_epochs)

    losses_no_tuning_weight_decay = []
    losses_no_tuning_epochs = []
    for model_name in PARAM_STRS:
        runs_model = [
            run
            for run in valid_runs
            if run["model_name"] == model_name and run["weight_decay"] == best_150M_weight_decay
        ]
        runs_model = sorted(runs_model, key=lambda x: x["final_dclm_loss"])
        losses_no_tuning_weight_decay.append(runs_model[0]["final_dclm_loss"])

        runs_model = [run for run in valid_runs if run["model_name"] == model_name and run["epochs"] == best_150M_epochs]
        runs_model = sorted(runs_model, key=lambda x: x["final_dclm_loss"])
        print(
            model_name,
            runs_model[0]["lr"],
            runs_model[0]["weight_decay"],
            runs_model[0]["epochs"],
            runs_model[0]["final_dclm_loss"],
        )
        losses_no_tuning_epochs.append(runs_model[0]["final_dclm_loss"])

    # law_no_tuning_weight_decay = PowerScalingLaw(var_name="N")
    # law_no_tuning_weight_decay.fit(PARAM_COUNTS, losses_no_tuning_weight_decay,
    # p0=[1.0, 0.5, 2.0], bounds=([0, 0, 0], [np.inf, np.inf, np.inf]))
    # x_fit = np.logspace(np.log10(min(PARAM_COUNTS)), np.log10(max(PARAM_COUNTS)), 100)

    plt.figure(figsize=ABLATION_FIGSIZE, dpi=300)
    plt.scatter(
        PARAM_COUNTS,
        parameter_scaling_losses[209715200],
        marker="o",
        color=BASELINE_COLOR,
        label="Tuning epochs (baseline)",
    )
    plt.scatter(PARAM_COUNTS, run_losses_200M, marker="o", color=REGULARIZED_COLOR, label="Regularized recipe")
    plt.scatter(PARAM_COUNTS, losses_no_tuning_weight_decay, marker="s", color=GREEN, label="Fixing 0.8 weight decay")
    plt.scatter(PARAM_COUNTS, losses_no_tuning_epochs, marker="*", color=LIGHT_BLUE, label="Fixing 16 epochs")

    # plt.plot(x_fit, law_no_tuning_weight_decay.evaluate(x_fit),
    #  "--", color=param_str_color_dict["150m4k"], label=f"Fit: {law_no_tuning_weight_decay}")
    plt.xscale("log")
    plt.xlabel("Parameter count")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Ablating hyper-parameter tuning")
    plt.grid(True, which="both", linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        "experiments/data_efficiency/plots/no_convex_tuning_ablation_epochs_weight_decay.png", bbox_inches="tight"
    )
    plt.close()


def plot_parameter_count_vs_loss(run_list):
    base_token_count = 209715200
    best_tuned_epoch_runs = []
    best_4_epoch_runs = []
    for model_size in PARAM_STRS:
        curr_run_list = [
            run
            for run in run_list
            if run["batch_size"] == 64
            and "seed" not in run["run_id"]
            and "-ts" not in run["run_id"]
            and run["data_name"] == "dclm"
            and run["weight_decay"] == 0.1
            and run["model_name"] == model_size
            and run["base_tokens"] == base_token_count
        ]
        curr_run_list = sorted(curr_run_list, key=lambda x: x["final_dclm_loss"])
        best_tuned_epoch_runs.append(curr_run_list[0])

        curr_run_list = [run for run in curr_run_list if run["epochs"] == 4]
        curr_run_list = sorted(curr_run_list, key=lambda x: x["final_dclm_loss"])
        best_4_epoch_runs.append(curr_run_list[0])

    losses = [run["final_dclm_loss"] for run in best_tuned_epoch_runs]
    plt.figure(figsize=ABLATION_FIGSIZE, dpi=300)
    plt.plot(PARAM_COUNTS, losses, marker="o", color=BASELINE_COLOR, label="Tuned epochs validation loss")
    plt.ylim(3.7, 5.05)
    plt.xscale("log")
    plt.xticks(PARAM_COUNTS, [value_pretty_name_dict[param_count] for param_count in PARAM_COUNTS])
    plt.xticks([], [], minor=True)
    plt.xlabel("Parameter count")
    plt.ylabel("Loss")
    plt.title("Increasing parameter count")
    plt.grid(True, which="both", linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig("experiments/data_efficiency/plots/parameter_count_vs_loss.png", bbox_inches="tight")

    train_losses = [run["final_train_loss"] for run in best_tuned_epoch_runs]
    plt.plot(PARAM_COUNTS, train_losses, marker="o", color=GREEN, label="Tuned epochs train loss")

    four_epoch_losses = [run["final_dclm_loss"] for run in best_4_epoch_runs]
    four_epoch_train_losses = [run["final_train_loss"] for run in best_4_epoch_runs]
    plt.plot(PARAM_COUNTS, four_epoch_losses, marker="*", color=ORANGE, label="4 epochs validation loss")
    plt.plot(PARAM_COUNTS, four_epoch_train_losses, marker="*", color=ROBIN_EGG_BLUE, label="4 epochs train loss")
    plt.legend()
    plt.ylim(3.0, 4.0)
    plt.title("Increasing parameter count train loss")
    plt.savefig("experiments/data_efficiency/plots/train_loss_parameter_count_vs_loss.png", bbox_inches="tight")
    plt.close()


def plot_weight_decay_ablation(run_list):
    # run_list_300m4k_one_pass = sorted(
    #     [run for run in run_list if run["model_name"] == "300m4k" and run["epochs"] == 1],
    #     key=lambda x: x["weight_decay"],
    # )
    run_list_300m4k = sorted(
        [run for run in run_list if run["model_name"] == "300m4k" and run["epochs"] > 1], key=lambda x: x["weight_decay"]
    )
    run_list_1_4b4k = sorted(
        [run for run in run_list if run["model_name"] == "1_4b4k" and run["epochs"] > 1], key=lambda x: x["weight_decay"]
    )

    # wd_300m4k_one_pass = [run["weight_decay"] for run in run_list_300m4k_one_pass]
    wd_300m4k = [run["weight_decay"] for run in run_list_300m4k]
    wd_1_4b4k = [run["weight_decay"] for run in run_list_1_4b4k]

    # losses_300m4k_one_pass = [run["final_dclm_loss"] for run in run_list_300m4k_one_pass]
    losses_300m4k = [run["final_dclm_loss"] for run in run_list_300m4k]
    losses_1_4b4k = [run["final_dclm_loss"] for run in run_list_1_4b4k]

    plt.figure(figsize=ABLATION_FIGSIZE, dpi=300)

    # plt.plot(wd_300m4k_one_pass, losses_300m4k_one_pass, marker="o",
    #  color=LIGHT_BLUE, label="300M parameters (one pass)")
    plt.plot(wd_300m4k, losses_300m4k, marker="o", color=param_str_color_dict["300m4k"], label="300M parameters")
    plt.plot(wd_1_4b4k, losses_1_4b4k, marker="o", color=param_str_color_dict["1_4b4k"], label="1.4B parameters")
    plt.xlabel("Weight decay")
    plt.ylabel("Loss")
    plt.xscale("log")
    plt.title("Tuning weight decay")
    plt.grid(True, which="both", linestyle="--", alpha=0.3)

    plt.xticks(wd_300m4k, [f"{x}" for x in wd_300m4k])
    plt.xticks([], [], minor=True)

    plt.legend()
    plt.tight_layout()
    plt.savefig("experiments/data_efficiency/plots/ablation_weight_decay.png", bbox_inches="tight")
    plt.close()


def plot_chinchilla_scaling_law(run_list):
    valid_flops = [3e18, 6e18, 1e19, 3e19, 6e19]
    max_parameter_count = 3_000_000_000
    run_list = [run for run in run_list if run["flops"] in valid_flops and run["parameter_count"] <= max_parameter_count]
    run_list = sorted(run_list, key=lambda x: x["final_dclm_loss"])

    visited_parameter_token_pairs = set()
    new_run_list = []
    for run in run_list:
        if (run["parameter_count"], run["token_count"]) not in visited_parameter_token_pairs:
            visited_parameter_token_pairs.add((run["parameter_count"], run["token_count"]))
            new_run_list.append(run)

    run_list = new_run_list

    scaling_law = ChinchillaScalingLaw()
    parameter_counts = np.array([run["parameter_count"] / 10**9 for run in run_list])
    token_counts = np.array([run["token_count"] / 10**9 for run in run_list])
    losses = np.array([run["final_dclm_loss"] for run in run_list])
    scaling_law.fit((parameter_counts, token_counts), losses, p0=[1, 1, 1, 1, 1], bounds=(0, np.inf))
    print(scaling_law)
    print("Infinite model size, 200M tokens:", scaling_law.evaluate((1000000000000, 0.2)))
    print("Infinite model size, 400M tokens:", scaling_law.evaluate((1000000000000, 0.4)))
    print("Infinite model size, 800M tokens:", scaling_law.evaluate((1000000000000, 0.8)))
    print("Infinite model size, 1.6B tokens:", scaling_law.evaluate((1000000000000, 1.6)))

    plt.figure(figsize=(7, 7), dpi=300)

    for flops in valid_flops:
        flop_runs = sorted([run for run in run_list if run["flops"] == flops], key=lambda x: x["token_count"])
        flop_token_counts = np.array([run["token_count"] / 10**9 for run in flop_runs])
        flop_parameter_counts = np.array([run["parameter_count"] / 10**9 for run in flop_runs])
        flop_losses = np.array([run["final_dclm_loss"] for run in flop_runs])
        plt.scatter(flop_parameter_counts, flop_losses, label=f"FLOPS: {flops}")

        projected_losses = scaling_law.evaluate((flop_parameter_counts, flop_token_counts))
        plt.plot(flop_parameter_counts, projected_losses, linestyle="--")

    plt.plot([], [], label=f"Chinchilla: {scaling_law}", color="black", linestyle="--")
    plt.legend()
    plt.xlabel("Parameter count (B)")
    plt.ylabel("Loss")
    plt.xscale("log")
    plt.tight_layout()
    plt.savefig(f"experiments/data_efficiency/plots/chinchilla_scaling_law_{key}.png")
    plt.close()

    return scaling_law


def plot_chinchilla_comparison_200M(run_list, power_law_200M, run_losses_200M):
    plt.figure(figsize=ABLATION_FIGSIZE, dpi=300)
    print("Run list:", run_list)

    run_list = sorted(run_list, key=lambda x: param_str_to_count[x["model_name"]])

    param_counts = [param_str_to_count[run["model_name"]] for run in run_list]
    print(run_losses_200M)
    print(power_law_200M)
    # print(chinchilla_scaling_law)
    print(param_counts)
    print("Number of runs:", len(run_list))

    x_fit = np.logspace(np.log10(min(param_counts)), np.log10(max(param_counts)), 100)

    # single_run_losses = [run["final_dclm_loss"] for run in run_list]
    # TODO: automate soon
    single_run_losses = [3.83752, 3.78538, 3.74974, 3.76432]
    plt.scatter(param_counts, single_run_losses, color=BASELINE_COLOR, s=50, label="Epoched recipe")
    # epoched_chinchilla_power_law = PowerScalingLaw()
    # epoched_chinchilla_power_law.fit(param_counts, single_run_losses, p0=[1, 1, 1], bounds=(0, np.inf))
    # print(epoched_chinchilla_power_law)
    plt.plot(param_counts, single_run_losses, color=BASELINE_COLOR)


    more_unregularized_losses = [3.72485, 3.74383]
    more_param_counts = [1.5, 3.2]
    plt.scatter(more_param_counts, more_unregularized_losses, color=BASELINE_COLOR, s=70, marker='*', label='Epoched extrapolation')

    y_fit = power_law_200M.evaluate(x_fit)
    plt.scatter(param_counts[:4], run_losses_200M, color=REGULARIZED_COLOR, s=50)
    more_x_fit = np.logspace(np.log10(min(param_counts)), np.log10(3.2), 100)
    more_y_fit = power_law_200M.evaluate(more_x_fit)
    plt.plot(more_x_fit, more_y_fit, "--", color=REGULARIZED_COLOR, label=f"Regularized recipe\n(Fit: {power_law_200M})")

    more_losses = [3.46567, 3.45199]
    plt.scatter(more_param_counts, more_losses, color=REGULARIZED_COLOR, s=70, marker='*', label='Regularized extrapolation')

    # projected_losses = chinchilla_scaling_law.evaluate((x_fit, 0.2))
    # plt.plot(x_fit, projected_losses, linestyle="--", label=f"Chinchilla: {chinchilla_scaling_law}")

    plt.xscale("log")
    plt.xticks(param_counts, ["150M", "300M", "600M", "1.4B"])
    plt.xticks([], [], minor=True)
    plt.ylim(bottom=3.43, top=3.85)
    plt.title("Regularized parameter scaling")
    plt.grid(True, which="both", linestyle="--", alpha=0.3)
    plt.legend(loc="center right")
    plt.xlabel("Parameter count")
    plt.ylabel("Loss")
    plt.tight_layout()
    plt.savefig("experiments/data_efficiency/plots/chinchilla_comparison_200M.png")
    plt.close()

def plot_wrap(run_list, power_law_200M, run_losses_200M):
    plt.figure(figsize=(10, 5), dpi=300)
    print("Run list:", run_list)

    run_list = sorted(run_list, key=lambda x: param_str_to_count[x["model_name"]])

    # Standard model scaling
    param_counts = [154147328 / 1e9, 299649792 / 1e9, 602457088 / 1e9, 1540732416 / 1e9]
    single_run_losses = [3.94794, 3.76518, 3.68461, 3.64257]

    power_law = PowerScalingLaw(var_name="N")
    power_law.fit(param_counts, single_run_losses, p0=[1.0, 0.5, 2.0], bounds=([0, 0, 0], [np.inf, np.inf, np.inf]))
    x_fit = np.logspace(np.log10(min(param_counts)), np.log10(max(param_counts)), 100)

    y_fit = power_law.evaluate(x_fit)
    plt.scatter(param_counts, single_run_losses, color=REGULARIZED_COLOR, s=50)
    plt.plot(x_fit, y_fit, "--", color=REGULARIZED_COLOR, label=f"Regularized recipe\n(Fit: {power_law})")

    # dotted horizontal line at 3.62
    # plt.plot([0, 7.54], [3.62, 3.62], "--", color="black", linestyle="--", alpha=0.3)

    """
    param_str_color_dict = {
    "150m4k": CRAYOLA_BLUE,
    "300m4k": ROBIN_EGG_BLUE,
    "600m4k": APPLE_GREEN,
    "1_4b4k": ORANGE,
    "1_5b4k": PINK,
    "3_2b4k": VIRIDIAN,
    }
    """
    
    # 150M ensembles 
    param_counts = [(154147328 / 1e9) * i for i in range(1, 6)]
    losses = [4.25709, 3.97665, 3.88724, 3.84555, 3.81822]
    power_law = PowerScalingLaw(var_name="N")
    power_law.fit(param_counts, losses, p0=[1.0, 0.5, 2.0], bounds=([0, 0, 0], [np.inf, np.inf, np.inf]))
    x_fit = np.logspace(np.log10(min(param_counts)), np.log10(max(param_counts)), 100)
    y_fit = power_law.evaluate(x_fit)
    plt.scatter(param_counts, losses, color=param_str_color_dict["150m4k"], s=50)
    plt.plot(x_fit, y_fit, "--", color=param_str_color_dict["150m4k"], label=f"150M Ensembles\n(Fit: {power_law})")

    # 300M ensembles
    param_counts = [(299649792 / 1e9) * i for i in range(1, 6)]
    losses = [4.09653, 3.79211, 3.70406, 3.65132, 3.61938]
    power_law = PowerScalingLaw(var_name="N")
    power_law.fit(param_counts, losses, p0=[1.0, 0.5, 2.0], bounds=([0, 0, 0], [np.inf, np.inf, np.inf]))
    x_fit = np.logspace(np.log10(min(param_counts)), np.log10(max(param_counts)), 100)
    y_fit = power_law.evaluate(x_fit)
    plt.scatter(param_counts, losses, color=param_str_color_dict["300m4k"], s=50)
    plt.plot(x_fit, y_fit, "--", color=param_str_color_dict["300m4k"], label=f"300M Ensembles\n(Fit: {power_law})")

    # 600M ensembles
    param_counts = [(602457088 / 1e9) * i for i in range(1, 6)]
    losses = [4.13199, 3.77925, 3.66712, 3.61295, 3.57864]
    power_law = PowerScalingLaw(var_name="N")
    power_law.fit(param_counts, losses, p0=[1.0, 0.5, 2.0], bounds=([0, 0, 0], [np.inf, np.inf, np.inf]))
    x_fit = np.logspace(np.log10(min(param_counts)), np.log10(max(param_counts)), 100)
    y_fit = power_law.evaluate(x_fit)
    plt.scatter(param_counts, losses, color=param_str_color_dict["600m4k"], s=50)
    plt.plot(x_fit, y_fit, "--", color=param_str_color_dict["600m4k"], label=f"600M Ensembles\n(Fit: {power_law})")

    # 1.5B ensembles
    param_counts = [(1540732416 / 1e9) * i for i in [1,2,3,4,5]]
    losses = [4.17566, 3.7952, 3.67928, 3.61881, 3.58401]
    power_law = PowerScalingLaw(var_name="N")
    power_law.fit(param_counts, losses, p0=[1.0, 0.5, 2.0], bounds=([0, 0, 0], [np.inf, np.inf, np.inf]))
    x_fit = np.logspace(np.log10(min(param_counts)), np.log10(max(param_counts)), 100)
    y_fit = power_law.evaluate(x_fit)
    plt.scatter(param_counts, losses, color=param_str_color_dict["1_5b4k"], s=50)
    plt.plot(x_fit, y_fit, "--", color=param_str_color_dict["1_5b4k"], label=f"1.5B Ensembles\n(Fit: {power_law})")

    # 300M ensembles (hq)
    param_counts = [(299649792 / 1e9) * i for i in range(1, 6)]
    losses = [3.67209, 3.57402, 3.54472, 3.52817, 3.51803]    
    power_law = PowerScalingLaw(var_name="N")
    power_law.fit(param_counts, losses, p0=[1.0, 0.5, 2.0], bounds=([0, 0, 0], [np.inf, np.inf, np.inf]))
    x_fit = np.logspace(np.log10(min(param_counts)), np.log10(max(param_counts)), 100)
    y_fit = power_law.evaluate(x_fit)
    plt.scatter(param_counts, losses, color=WRAP_COLOR, s=50)
    plt.plot(x_fit, y_fit, "--", color=WRAP_COLOR, label=f"300M WRAP Ens\n(Fit: {power_law})")

    # 300M ensembles (sd)
    param_counts = [(299649792 / 1e9) * i for i in [1,2,3,4,5]]
    losses = [3.67071, 3.60728, 3.58736, 3.57565, 3.56865]
    power_law = PowerScalingLaw(var_name="N")
    power_law.fit(param_counts, losses, p0=[1.0, 0.5, 2.0], bounds=([0, 0, 0], [np.inf, np.inf, np.inf]))
    x_fit = np.logspace(np.log10(min(param_counts)), np.log10(max(param_counts)), 100)
    y_fit = power_law.evaluate(x_fit)
    plt.scatter(param_counts, losses, color=SELF_DISTILL_COLOR, s=50)
    plt.plot(x_fit, y_fit, "--", color=SELF_DISTILL_COLOR, label=f"300M Self-Distill Ens\n(Fit: {power_law})")

    # 300M ensembles (symx)
    param_counts = [(299649792 / 1e9) * i for i in [1,2,3,4,5]]
    losses = [3.58481, 3.49636, 3.46847, 3.45411, 3.44538]
    power_law = PowerScalingLaw(var_name="N")
    power_law.fit(param_counts, losses, p0=[1.0, 0.5, 2.0], bounds=([0, 0, 0], [np.inf, np.inf, np.inf]))
    x_fit = np.logspace(np.log10(min(param_counts)), np.log10(max(param_counts)), 100)
    y_fit = power_law.evaluate(x_fit)
    plt.scatter(param_counts, losses, color=MIX_COLOR, s=50)
    plt.plot(x_fit, y_fit, "--", color=MIX_COLOR, label=f"300M Mix Ens\n(Fit: {power_law})")

    # more_losses = [3.67252]
    # plt.scatter(299649792 / 1e9, more_losses, color=BASELINE_COLOR, s=70, marker='*', label='WRAP')

    plt.xscale("log")
    param_counts = [154147328 / 1e9, 299649792 / 1e9, 602457088 / 1e9, 1540732416 / 1e9]
    plt.xticks(param_counts, ["150M", "300M", "600M", "1.5B"])
    plt.xticks([], [], minor=True)
    # plt.ylim(bottom=3.6, top=4.0)
    plt.title("Synthetic data ensembles")
    plt.grid(True, which="both", linestyle="--", alpha=0.3)
    # legend outside the plot
    plt.legend(loc="upper right", bbox_to_anchor=(1.3, 1), borderaxespad=0.)
    plt.xlabel("Parameter count")
    plt.ylabel("Loss")
    plt.tight_layout()
    plt.savefig("experiments/data_efficiency/plots/synthetic_data_ensembles.png")
    plt.close()


def plot_loss_trajectories(runs, labels, colors, file_name):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(ABLATION_FIGSIZE[0] * 2, ABLATION_FIGSIZE[1]), dpi=300)

    for run, label, color in zip(runs, labels, colors, strict=False):
        steps = run["loss_history"]["_step"] * run["batch_size"] * 4096 / 1_000_000_000.0
        ax1.plot(steps, run["loss_history"]["train/loss"], label=label, color=color)
        ax2.plot(steps, run["loss_history"]["eval/dclm/loss"], label=label, color=color)

    for ax in [ax1, ax2]:
        ax.set_xscale("log")
        ax.grid(True, which="both", linestyle="--", alpha=0.3)
        ax.set_xlabel("Tokens (billions)")
        ax.set_ylabel("Loss")
        ax.set_ylim(3.1, 4.58)
        ax.set_xscale("log")
    ax1.legend()
    ax1.set_title("Train loss")
    ax2.set_title("Validation loss")
    fig.tight_layout()
    plt.savefig(f"experiments/data_efficiency/plots/{file_name}.png")
    plt.close()


def weight_decay_loss_trajectories(run_list):
    # optimal_with_wd_id = "300m4k-209Mx16-dclm-cos-lr0.0030-wd1.60-bs64"
    # optimal_wo_wd_id = "300m4k-209Mx8-dclm-cos-lr0.0010-wd0.10-bs64"

    # optimal_with_wd_id = "1_4b4k-209Mx8-dclm-cos-lr0.0010-wd3.20-bs64"
    # optimal_wo_wd_id = "1_4b4k-209Mx8-dclm-cos-lr0.0010-wd0.10-bs64"

    # optimal_wo_wd_id = "300m4k-1.7Bx16-dclm-cos-lr0.0030-wd0.00-bs64"
    # optimal_with_wd_id = "300m4k-1.7Bx32-dclm-cos-lr0.0010-wd0.40-bs64"

    optimal_wo_wd_id = "300m4k-209Mx8-dclm-cos-lr0.0010-wd0.10-bs64"
    # optimal_little_wd_id = "300m4k-209Mx8-dclm-cos-lr0.0010-wd0.10-bs64"
    optimal_with_wd_id = "300m4k-209Mx16-dclm-cos-lr0.0030-wd1.60-bs64"

    optimal_wo_wd_run = next(run for run in run_list if run["run_id"] == optimal_wo_wd_id)
    # optimal_little_wd_run = next(run for run in run_list if run["run_id"] == optimal_little_wd_id)
    optimal_with_wd_run = next(run for run in run_list if run["run_id"] == optimal_with_wd_id)

    plot_loss_trajectories(
        [optimal_wo_wd_run, optimal_with_wd_run],
        ["Weight decay 0.1", "Weight decay 1.6"],
        [BASELINE_COLOR, REGULARIZED_COLOR],
        "weight_decay_loss_trajectories",
    )


def run_to_run_variance(run_list):
    both_seed_runs = []
    train_seed_runs = []
    data_seed_runs = []

    base_run = "300m4k-209Mx16-dclm-cos-lr0.0030-wd1.60"

    for seed in range(5):
        both_seed_runs.append(next(run for run in run_list if run["run_id"] == f"{base_run}-ts{seed}-ds{seed}"))
        train_seed_runs.append(next(run for run in run_list if run["run_id"] == f"{base_run}-ts{seed}-ds0"))
        data_seed_runs.append(next(run for run in run_list if run["run_id"] == f"{base_run}-ts0-ds{seed}"))

    print("Standard deviation of both seed runs:", np.std([run["final_dclm_loss"] for run in both_seed_runs]))
    print("Standard deviation of train seed runs:", np.std([run["final_dclm_loss"] for run in train_seed_runs]))
    print("Standard deviation of data seed runs:", np.std([run["final_dclm_loss"] for run in data_seed_runs]))


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
        "standard-batch-size": ("batch-size-test-8-4", "stanford-mercury/suhas-data-efficiency"),
        "standard-epoch-overfitting": ("epoch-overfitting-8-4", "stanford-mercury/suhas-data-efficiency"),
        "standard-weight-decay": ("weight-decay-8-4", "stanford-mercury/suhas-data-efficiency"),
        "dclm-chinchilla": ("dclm-default-sweep", "stanford-mercury/marin"),
        "dclm-chinchilla-lr-tune": ("suhas-dclm-chinchilla-lr-tune", "stanford-mercury/marin"),
        "chinchilla-comparison": ("chinchilla-single-runs", "stanford-mercury/suhas-data-efficiency"),
    }[mode]

    if args.build_cache:
        run_list = []
        runs = wandb.Api().runs(project_name)
        for run in tqdm(runs):
            run_dict = parse_run(run)
            if run_dict is not None:
                print(run_dict["run_id"])
                run_list.append(run_dict)
        pickle.dump(run_list, open(f"experiments/data_efficiency/cache/{mode}_run_list.pkl", "wb"))
    else:
        if mode != "benchmark-results":
            run_list = pickle.load(open(f"experiments/data_efficiency/cache/{mode}_run_list.pkl", "rb"))

    if mode == "varying-hparams-experiment":
        only_200M = False

        unique_models = sorted(list(set([run["model_name"] for run in run_list])), key=lambda x: param_str_to_count[x])
        unique_base_tokens = sorted(list(set([run["base_tokens"] for run in run_list])))

        print("Unique models:", unique_models)
        print("Unique base tokens:", unique_base_tokens)

        power_law_200M, run_losses_200M = pickle.load(
            open("experiments/data_efficiency/cache/standard_asymptotes_200M.pkl", "rb")
        )

        best_ensembles = {}

        best_single_model_hparams_ensemble_200M = None

        for model_size in unique_models:
            print("Model size:", model_size)
            best_ensembles[model_size] = {}
            for base_tokens in unique_base_tokens if not only_200M else [209715200]:
                ret, best_single_model_hparams_ensemble = plot_ensemble_scaling(model_size, base_tokens)
                if ret is not None:
                    _, x_data, y_data, power_law = ret
                    print(power_law.asymptote())
                    best_ensembles[model_size][base_tokens] = (x_data[:5], y_data[:5], power_law)

                if model_size == "300m4k" and base_tokens == 209715200:
                    best_single_model_hparams_ensemble_200M = best_single_model_hparams_ensemble
                    best_asymptote_ensemble_200M = (x_data, y_data, power_law)

        epoched_data_scaling_law = pickle.load(
            open("experiments/data_efficiency/cache/epoched_data_scaling_law.pkl", "rb")
        )

        plot_200M_sample(
            {model_size: best_ensembles[model_size][209715200] for model_size in unique_models},
            power_law_200M,
            run_losses_200M,
            best_single_model_hparams_ensemble_200M,
            epoched_data_scaling_law,
        )
        # plot_200M_sample(best_ensembles[min(unique_models)])
        print(best_ensembles)
        if not only_200M:
            pickle.dump(best_ensembles, open(f"experiments/data_efficiency/cache/{mode}_best_ensembles.pkl", "wb"))

        pickle.dump(
            best_asymptote_ensemble_200M,
            open(f"experiments/data_efficiency/cache/{mode}_best_asymptote_ensemble_200M.pkl", "wb"),
        )

    elif mode == "infinite-model-scaling":
        run_list = [run for run in run_list if run["epochs"] != 13]
        plot_wd_overfitting_ablation(run_list)
        no_convex_tuning_ablation(run_list)
        sensitivity_analysis(run_list)
        run_to_run_variance(run_list)
        with plt.rc_context({"font.size": 12}):
            plot_lr_tuned_epoch_ablation(run_list)
            plot_parameter_count_vs_loss(run_list)

        parameter_scaling_losses = cache_parameter_scaling_losses(run_list)
        best_run_dict = construct_best_run_dict(run_list)
        weight_decay_loss_trajectories(run_list)


        # # Also create token scaling plot
        # valid_model_sizes, asymptotes = plot_token_scaling(best_run_dict, fit_type="power_law")
        with plt.rc_context({"font.size": 14}):
            power_laws, run_losses = plot_model_scaling(best_run_dict, fit_type="power_law")

        # Fit joint scaling law
        # fit_joint_scaling_law(best_run_dict)

        # Create paper version of plots
        with plt.rc_context({"font.size": 12}):
            print(parameter_scaling_losses)
            standard_seed_scaling_asymptotes, standard_seed_scaling_power_law, epoched_data_scaling_law = (
                plot_standard_model_seed_scaling(parameter_scaling_losses, best_run_dict, fit_type="power_law")
            )
            plot_ensemble_model_seed_scaling(standard_seed_scaling_asymptotes, standard_seed_scaling_power_law)
        # plot_token_scaling_simple(best_run_dict, fit_type="power_law")

        print(power_laws[0], run_losses[0])
        pickle.dump(
            (power_laws[0], run_losses[0]), open("experiments/data_efficiency/cache/standard_asymptotes_200M.pkl", "wb")
        )
        pickle.dump(
            epoched_data_scaling_law, open("experiments/data_efficiency/cache/epoched_data_scaling_law.pkl", "wb")
        )

    elif mode == "seed-science":
        run_list = sorted(run_list, key=lambda x: x["ensemble_member_count"])
        train_seed_losses = [run["final_dclm_loss"] for run in run_list if "train" in run["seed_types"]]
        data_seed_losses = [run["final_dclm_loss"] for run in run_list if "data" in run["seed_types"]]
        both_seed_losses = [run["final_dclm_loss"] for run in run_list if "both" in run["seed_types"]]

        plot_seed_science(train_seed_losses, data_seed_losses, both_seed_losses)

    elif mode == "benchmark-results":
        with plt.rc_context({"font.size": 12}):
            plot_benchmark_results()

    elif mode == "distillation":
        unique_models = ["300m4k"]
        unique_base_tokens = sorted(list(set([run["base_tokens"] for run in run_list])))

        best_ensembles = pickle.load(
            open("experiments/data_efficiency/cache/varying-hparams-experiment_best_ensembles.pkl", "rb")
        )
        best_asymptote_ensemble_200M = pickle.load(
            open("experiments/data_efficiency/cache/varying-hparams-experiment_best_asymptote_ensemble_200M.pkl", "rb")
        )

        power_law_200M, run_losses_200M = pickle.load(
            open("experiments/data_efficiency/cache/standard_asymptotes_200M.pkl", "rb")
        )

        plot_distillation(best_asymptote_ensemble_200M, power_law_200M, run_losses_200M)
        plot_simple_cross(best_asymptote_ensemble_200M, power_law_200M, run_losses_200M)

    elif mode == "standard-batch-size":
        plot_batch_size_ablation(run_list)

    # elif mode == "standard-epoch-overfitting":
    #     plot_epoch_overfitting_ablation(run_list)

    elif mode == "standard-weight-decay":
        plot_weight_decay_ablation(run_list)

    elif mode == "dclm-chinchilla" or mode == "dclm-chinchilla-lr-tune":
        chinchilla_scaling_law = plot_chinchilla_scaling_law(run_list)
        pickle.dump(chinchilla_scaling_law, open("experiments/data_efficiency/cache/chinchilla_scaling_law.pkl", "wb"))

    elif mode == "chinchilla-comparison":
        # chinchilla_scaling_law = pickle.load(open("experiments/data_efficiency/cache/chinchilla_scaling_law.pkl", "rb"))
        power_law_200M, run_losses_200M = pickle.load(
            open("experiments/data_efficiency/cache/standard_asymptotes_200M.pkl", "rb")
        )
        # plot_chinchilla_comparison_200M(run_list, power_law_200M, run_losses_200M)
        plot_wrap(run_list, power_law_200M, run_losses_200M)
