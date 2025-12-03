import matplotlib.pyplot as plt
import wandb
import argparse
from tqdm import tqdm
import pickle
import json
import matplotlib.colors as mcolors
from scipy.optimize import curve_fit
import numpy as np
from experiments.data_efficiency.synthetic_data_plotting_utils import *

def get_power_law_fit(x, y, x_max=None):
    if x_max is None:
        x_max = max(x)
    power_law = PowerScalingLaw(var_name="N")
    power_law.fit(x, y, p0=[1.0, 0.5, 2.0], bounds=([0, 0, 0], [np.inf, np.inf, np.inf]))
    x_fit = np.logspace(np.log10(min(x)), np.log10(x_max), 100)
    y_fit = power_law.evaluate(x_fit)
    return x_fit, y_fit, power_law

def default_model_scaling_plot_settings():
    plt.xscale("log")
    plt.xticks(list(PARAM_STR_TO_COUNT.values()), ["150M", "300M", "600M", "1.5B"])
    plt.xticks([], [], minor=True)
    plt.grid(True, which="both", linestyle="--", alpha=0.3)
    plt.legend(loc="upper right")
    plt.xlabel("Parameter count")
    plt.ylabel("Loss")
    plt.tight_layout()


def parse_run(run):
    if run.state != "finished":
        print(f"\033[91mSkipping run {run.id} because it is not finished\033[0m")
        return None

    run_dict = {}
    run_id = run.name
    run_dict["run_id"] = run_id
    run_json_config = json.loads(run.json_config)

    num_steps = run_json_config["trainer"]["value"]["num_train_steps"]
    batch_size = run_json_config["trainer"]["value"]["train_batch_size"]
    seq_len = run_json_config["model"]["value"]["seq_len"]
    base_tokens = None

    if "dclm_200m" not in run_id:
        print(f"\033[91mSkipping run {run_id} because it is not a dclm_200m run\033[0m")
        return None

    # NOTE: for eval loss for synth data, we use the loss key "eval/loss"
    run_history_loss_keys = ["eval/loss", "train/loss"]
    if run_id.startswith("ppl-eval-ensemble-"):
        run_history_loss_keys = ["eval/loss"]
        run_id = run_id[len("ppl-eval-ensemble-") :]

        run_dict["ensemble_member_count"] = int(run_id.split("x-")[0])
        run_id = run_id.split("x-")[1]

        for token_str in TOKEN_STR_TO_STEPS:
            if token_str in run_id:
                num_base_steps = TOKEN_STR_TO_STEPS[token_str]
                break
        else:
            raise ValueError(f"Unknown token count: {run_id}")

        batch_size = 64
        seq_len = 4096
        base_tokens = num_base_steps * batch_size * seq_len

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

    history_loss = run.history(keys=run_history_loss_keys)
    if "eval/loss" not in history_loss.columns:
        print(f"\033[91mSkipping run {run_id} because it does not have the loss history\033[0m")
        return None

    run_dict["loss_history"] = history_loss
    run_dict["final_eval_loss"] = history_loss["eval/loss"].iloc[-1]
    if "train/loss" in history_loss.columns:
        run_dict["final_train_loss"] = history_loss["train/loss"].iloc[-1]


    return run_dict

def plot_baselines(run_list):
    plt.figure(figsize=WIDE_RECTANGLE_FIGSIZE, dpi=300)
    run_list = sorted(run_list, key=lambda x: PARAM_STR_TO_COUNT[x["model_name"]])

    unregularized_names = [
        "150m4k-196Mx8-dclm_200m_train-cos-lr0.0030-wd0.10-bs64",
        "300m4k-196Mx8-dclm_200m_train-cos-lr0.0010-wd0.10-bs64",
        "600m4k-196Mx4-dclm_200m_train-cos-lr0.0010-wd0.10-bs64",
        "1_5b4k-196Mx4-dclm_200m_train-cos-lr0.0010-wd0.10-bs64",
    ]
    unregularized_runs = sorted(
        [
            run for run in run_list if run["run_id"] in unregularized_names
        ],
        key = lambda x: PARAM_STR_TO_COUNT[x["model_name"]]
    )
    all_regularized_runs = sorted(
        [
            run for run in run_list if "seed" not in run["run_id"]
            and run["data_name"] == "dclm_200m"
        ], 
        key=lambda x: PARAM_STR_TO_COUNT[x["model_name"]]
    )
    regularized_runs = []
    for param_str in PARAM_STR_TO_COUNT:
        best_regularized_run = sorted(
            [run for run in all_regularized_runs if run["model_name"] == param_str],
            key=lambda x: x["final_eval_loss"]
        )[0]
        regularized_runs.append(best_regularized_run)


    unregularized_losses = [run["final_eval_loss"] for run in unregularized_runs]
    regularized_losses = [run["final_eval_loss"] for run in regularized_runs]
    
    param_counts = list(PARAM_STR_TO_COUNT.values())
    plt.scatter(param_counts, unregularized_losses, color=BASELINE_COLOR, s=50)
    plt.scatter(param_counts, regularized_losses, color=PURPLE, s=50)

    plt.plot(param_counts, unregularized_losses, "--", color=BASELINE_COLOR, label=f"Unregularized")
    x_fit, y_fit, power_law = get_power_law_fit(param_counts, regularized_losses)
    plt.plot(x_fit, y_fit, "--", color=PURPLE, label=f"Regularized (Fit: {power_law})")

    default_model_scaling_plot_settings()
    plt.title("Model scaling baselines")
    plt.savefig("experiments/data_efficiency/plots/synthetic_data/model_scaling_baselines.png", bbox_inches="tight")
    plt.close()

    return unregularized_runs, regularized_runs

def plot_synth_data_loss_vs_real_epochs(run_list, synth_data_names, extra_info=False):
    plt.figure(figsize=WIDE_RECTANGLE_FIGSIZE, dpi=300)
    epoch_counts = [1, 2, 4, 8, 16]
    for synth_data_name in synth_data_names:
        synth_data_runs = [
            run for run in run_list if run["data_name"].startswith("dclm_200m+" + synth_data_name)
            and run["model_name"] == "300m4k"
            and "seed" not in run["run_id"]
        ]

        best_synth_data_runs = []
        for epoch in epoch_counts:
            synth_data_runs_epoch = [run for run in synth_data_runs if run["epochs"] == epoch]
            best_synth_data_run = sorted(synth_data_runs_epoch, key=lambda x: x["final_eval_loss"])[0]
            best_synth_data_runs.append(best_synth_data_run)
            print(epoch, best_synth_data_run["run_id"])
        
        epochs = [run["epochs"] for run in best_synth_data_runs]
        losses = [run["final_eval_loss"] for run in best_synth_data_runs]

        min_loss = min(losses)
        min_epoch = epochs[losses.index(min_loss)]
        plt.scatter(min_epoch, min_loss, color="red", s=50, marker="o", edgecolor="red", linewidth=4)
        if extra_info:
            extra_epochs = [run["epochs"] for run in synth_data_runs]
            extra_losses = [run["final_eval_loss"] for run in synth_data_runs]
            plt.scatter(extra_epochs, extra_losses, color=SYNTH_DATA_COLOR_DICT[synth_data_name], s=50, alpha=0.2)
            plt.plot([0, 16], [3.76518, 3.76518], "--", color=PARAM_STR_COLOR_DICT["300m4k"], alpha=0.5, label="Regularized 300M (Loss: 3.76)")
            plt.plot([0, 16], [3.62, 3.62], "--", color=REGULARIZED_COLOR, alpha=0.5, label="Regularized Asymptote (Loss: 3.62)")

        plt.scatter(epochs, losses, color=SYNTH_DATA_COLOR_DICT[synth_data_name], s=50)
        plt.plot(epochs, losses, "--", color=SYNTH_DATA_COLOR_DICT[synth_data_name], label=f"{SYNTH_DATA_NAME_DICT[synth_data_name]} (Loss: {min_loss:.3f})")
    
    plt.xscale("log")
    plt.xticks(epoch_counts, [str(epoch) for epoch in epoch_counts])
    plt.xticks([], [], minor=True)
    plt.grid(True, which="both", linestyle="--", alpha=0.3)
    plt.legend(loc="upper right")
    plt.xlabel("Epochs of real data")
    plt.ylabel("Loss")
    plt.tight_layout()
    plt.title("Synthetic data loss vs real epochs")
    synth_data_name_str = "_".join(synth_data_names)
    if extra_info:
        synth_data_name_str += "_extra_info"
    plt.savefig(f"experiments/data_efficiency/plots/synthetic_data/synth_data_loss_vs_real_epochs_{synth_data_name_str}.png", bbox_inches="tight")
    plt.close()

def plot_ensemble_scaling(run_list, regularized_runs, ensembles_types, title="Ensemble scaling"):
    plt.figure(figsize=EXTRA_WIDE_RECTANGLE_FIGSIZE, dpi=300)
    param_counts = list(PARAM_STR_TO_COUNT.values())
    regularized_losses = [run["final_eval_loss"] for run in regularized_runs]

    plt.scatter(param_counts, regularized_losses, color=PURPLE, s=50)
    x_fit, y_fit, power_law = get_power_law_fit(param_counts, regularized_losses)
    plt.plot(x_fit, y_fit, "--", color=PURPLE, label=f"Regularized (Fit: {power_law})")

    for (ensemble_type, member_size) in ensembles_types:
        data_name = "dclm_200m" if not ensemble_type else "dclm_200m+" + ensemble_type
        ensemble_color = PARAM_STR_COLOR_DICT[member_size] if not ensemble_type else SYNTH_DATA_COLOR_DICT[ensemble_type]
        label = PRETTY_NAME_DICT[member_size] + ((" " + PRETTY_NAME_DICT[ensemble_type]) if ensemble_type else "")

        ensemble_runs = sorted([
            run for run in run_list if run["data_name"].startswith(data_name)
            and not run["data_name"].startswith(data_name + "+")
            and run["model_name"] == member_size
            and "ensemble_member_count" in run
        ], key=lambda x: x["ensemble_member_count"])

        # Filter for runs with max weight decay only (hack to deal with 1.5B ablation)
        max_weight_decay = max([run["weight_decay"] for run in ensemble_runs])
        ensemble_runs = [run for run in ensemble_runs if run["weight_decay"] == max_weight_decay]

        total_params = [PARAM_STR_TO_COUNT[run["model_name"]] * run["ensemble_member_count"] for run in ensemble_runs]
        losses = [run["final_eval_loss"] for run in ensemble_runs]

        plt.scatter(total_params, losses, color=ensemble_color, s=50)
        x_max = 7.6 if member_size == "300m4k" or ensemble_type else None
        x_fit, y_fit, power_law = get_power_law_fit(total_params, losses, x_max)
        plt.plot(x_fit, y_fit, "--", color=ensemble_color, label=f"{label} (Fit: {power_law})")
        
    default_model_scaling_plot_settings()
    plt.title(title)
    valid_ensembles = [ensemble_type for (ensemble_type, _) in ensembles_types if ensemble_type]
    ensemble_type_str = "_".join(valid_ensembles)
    plt.savefig(f"experiments/data_efficiency/plots/synthetic_data/ensemble_scaling_{ensemble_type_str}.png", bbox_inches="tight")
    plt.close()

def plot_synth_data_all(run_list):
    # filter run list 
    run_list = [run for run in run_list if run["model_name"] in PARAM_STR_TO_COUNT]

    # unregularized + regularized 
    _, regularized_runs = plot_baselines(run_list)

    # synthetic data comparisons
    plot_synth_data_loss_vs_real_epochs(run_list, ["hq_cpr16"], extra_info=True)
    plot_synth_data_loss_vs_real_epochs(run_list, ["hq_cpr16"])
    plot_synth_data_loss_vs_real_epochs(run_list, ["hq_cpr16", "sd_cpr16"])
    plot_synth_data_loss_vs_real_epochs(run_list, ["sd_cpr16", "sd_cpr200"])
    plot_synth_data_loss_vs_real_epochs(run_list, ["sd_cpr16", "sd_cpr200"])
    plot_synth_data_loss_vs_real_epochs(run_list, ["sd_cpr16", "sd_cpr200","sdn_c200"])
    plot_synth_data_loss_vs_real_epochs(run_list, ["symx_c16"])
    plot_synth_data_loss_vs_real_epochs(run_list, ["hq_cpr16", "sd_cpr16", "sd_cpr200", "symx_c16"])

    # ensemble scaling of regularized models 
    base_ensembles = [(None, "150m4k"), (None, "300m4k"), (None, "600m4k"), (None, "1_5b4k")]
    plot_ensemble_scaling(run_list, regularized_runs, base_ensembles)

    # ensemble scaling of synthetic data
    plot_ensemble_scaling(run_list, regularized_runs, base_ensembles + [("hq_cpr16", "300m4k")], title="WRAP Ensemble")
    plot_ensemble_scaling(run_list, regularized_runs, base_ensembles + [("hq_cpr16", "300m4k"), ("sd_cpr16", "300m4k")], title="WRAP vs. Self-Distill Ensemble")
    plot_ensemble_scaling(run_list, regularized_runs, base_ensembles + [("sd_cpr16", "300m4k"), ("sdn_c200", "300m4k")], title="Self-Distill vs. Self-Distill (Ens Teacher) Ensemble")
    plot_ensemble_scaling(run_list, regularized_runs, base_ensembles + [("hq_cpr16", "300m4k"), ("sd_cpr16", "300m4k"), ("symx_c16", "300m4k")], title="Synthetic Data Ensembles")
    plot_ensemble_scaling(run_list, regularized_runs, base_ensembles + [("hq_cpr16", "300m4k"), ("sd_cpr16", "300m4k"), ("symx_c16", "300m4k"), ("hq_cpr16", "600m4k")], title="WRAP Ensembles (600M) vs Mixed Ensembles (300M)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str)
    parser.add_argument("--build_cache", action="store_true")
    args = parser.parse_args()
    mode = args.mode

    project_names = {
        "synth_data_train": ["stanford-mercury/suhas-data-efficiency"],
        "synth_data_ensemble": ["stanford-mercury/suhas-eval-data-efficiency"],
        "synth_data_all": ["synth_data_train", "synth_data_ensemble"],
    }[mode]

    if args.build_cache:
        run_list = []
        for project_name in project_names:
            print(f"Fetching runs from {project_name}")
            try:
                runs = wandb.Api().runs(project_name)
                for run in tqdm(runs):
                    run_dict = parse_run(run)
                    if run_dict is not None:
                        print(run_dict["run_id"])
                        run_list.append(run_dict)
                print(f"Found {len(run_list)} runs")
            except Exception as e:
                print(f"Error fetching runs from {project_name}: {e}, instead loading from cache")
                current_list = pickle.load(open(f"experiments/data_efficiency/cache/{project_name}_run_list.pkl", "rb"))
                run_list.extend(current_list)
                continue
        pickle.dump(run_list, open(f"experiments/data_efficiency/cache/{mode}_run_list.pkl", "wb"))
    else: 
        run_list = pickle.load(open(f"experiments/data_efficiency/cache/{mode}_run_list.pkl", "rb"))


    if mode == "synth_data_all":
        plot_synth_data_all(run_list)
    else:
        raise ValueError(f"Unknown mode: {mode}")