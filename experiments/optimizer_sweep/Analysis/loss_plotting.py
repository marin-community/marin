import os
import json
import matplotlib.pyplot as plt
import pandas as pd
from plotting_config import color_map, correct_name, line_style

RESULTS_ROOT = "experiments/optimizer_sweep/Analysis/Results"


def _load_loss_from_result_json(result_path: str):
    try:
        with open(result_path, "r") as f:
            payload = json.load(f)
    except Exception:
        return None

    # Prefer explicit min_loss if present
    min_loss = payload.get("min_loss")
    if isinstance(min_loss, (int, float)):
        return float(min_loss)

    # Otherwise compute min across available fields
    candidate_losses = []
    baseline = payload.get("baseline")
    if isinstance(baseline, dict) and isinstance(baseline.get("loss"), (int, float)):
        candidate_losses.append(float(baseline["loss"]))

    for ab in payload.get("ablations", []) or []:
        if isinstance(ab, dict) and isinstance(ab.get("loss"), (int, float)):
            candidate_losses.append(float(ab["loss"]))

    if not candidate_losses and isinstance(payload.get("result"), dict):
        for v in payload["result"].values():
            if isinstance(v, (int, float)):
                candidate_losses.append(float(v))

    if not candidate_losses:
        return None
    return min(candidate_losses)


def collect_results(results_root: str) -> pd.DataFrame:
    rows = []
    if not os.path.isdir(results_root):
        return pd.DataFrame(columns=["optimizer", "model_size", "chinchilla", "loss"])

    for optimizer in os.listdir(results_root):
        opt_dir = os.path.join(results_root, optimizer)
        if not os.path.isdir(opt_dir):
            continue
        for model_size in os.listdir(opt_dir):
            ms_dir = os.path.join(opt_dir, model_size)
            if not os.path.isdir(ms_dir):
                continue
            for chinchilla in os.listdir(ms_dir):
                ch_dir = os.path.join(ms_dir, chinchilla)
                if not os.path.isdir(ch_dir):
                    continue
                result_path = os.path.join(ch_dir, "result.json")
                if not os.path.exists(result_path):
                    continue
                loss = _load_loss_from_result_json(result_path)
                if loss is None:
                    continue
                try:
                    chinchilla_int = int(chinchilla)
                except Exception:
                    # Skip non-integer chinchilla directory names
                    continue
                rows.append(
                    {
                        "optimizer": optimizer,
                        "model_size": model_size,
                        "chinchilla": chinchilla_int,
                        "loss": float(loss),
                    }
                )

    if not rows:
        return pd.DataFrame(columns=["optimizer", "model_size", "chinchilla", "loss"])
    return pd.DataFrame(rows)


# Load the loss data from Results JSONs
df_loss = collect_results(RESULTS_ROOT)
if df_loss.empty:
    raise SystemExit(f"No results found under {RESULTS_ROOT}")



# Generate a plot for each model size
figs_dir = "experiments/optimizer_sweep/Analysis/figs"
os.makedirs(figs_dir, exist_ok=True)

for size in sorted(df_loss["model_size"].unique()):
    data = df_loss[df_loss["model_size"] == size]

    plt.figure()
    for optimizer in sorted(data["optimizer"].unique()):
        subset = data[data["optimizer"] == optimizer].sort_values(by="chinchilla")
        plt.plot(
            subset["chinchilla"],
            subset["loss"],
            # label=optimizer,
            color=color_map.get(optimizer, "#000000"),
            linewidth=2,
            label=correct_name.get(optimizer, optimizer),
            linestyle=line_style.get(optimizer, "-"),
        )

    # Labels and styling
    plt.xticks([1, 2, 4, 8], ["1", "2", "4", "8"])
    plt.xlabel("Chinchilla Ratio", fontsize=18)
    plt.ylabel("Loss", fontsize=18)
    plt.title(f"C4/EN Loss for {size.upper()} Model", fontsize=18)
    plt.legend(fontsize=16, ncol=2)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.grid(linestyle="--", linewidth=0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(figs_dir, f"optimizer_loss_scaling_{size}.pdf"), bbox_inches="tight")
