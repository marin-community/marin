import copy
import wandb
from tqdm import tqdm
from marin.optimizer_sweep.utils_simp import actually_finish
# Initialize the WandB API
api = wandb.Api()
# Define your details
username = "stanford-mercury"
project = "optimizer-scaling"

def check_baseline_run(tags, samples = 100):
    filters = {"tags": {"$all": tags}}
    runs = api.runs(f"{username}/{project}", filters=filters)
    train_losses = []
    eval_losses = []
    if samples > 0:
        runs = runs[:samples]
    for run in tqdm(runs):
        if actually_finish(run, strict=False):
            # diverge before finish
            # or have some crazyness: finished eval multiple time
            train_loss = run.summary.get("train/loss", 0.0)
            eval_loss = run.summary.get("eval/paloma/c4_en/loss", 0.0)
            if type(train_loss) == float and train_loss <= 4:
                train_losses.append(train_loss)
                eval_losses.append(eval_loss)
    return train_losses, eval_losses

color_map = {
    "mars": "#1f77b4",  # blue
    "muon": "#ff7f0e",  # orange
    "lion": "#2ca02c",  # green
    "adamw": "#d62728",  # red
    "nadamw": "#9467bd",  # purple
    "kron": "#8c564b",  # brown
    "scion": "#e377c2",  # pink
    "cautious": "#7f7f7f",  # gray
    "soape": "#bcbd22",  # yellow-green
    "mini": "#aec7e8",  # light blue
}

correct_name = {
    "adamw": "AdamW",
    "lion": "Lion",
    "mini": "Adam-Mini",
    "scion": "Scion",
    "cautious": "Cautious",
    "mars": "Mars",
    "nadamw": "NAdamW",
    "muon": "Muon",
    "soape": "Soap",
    "kron": "Kron",
}

from matplotlib import pyplot as plt

if __name__ == "__main__":
    for optimizer in list(color_map.keys()):
        # train_losses, eval_losses = check_baseline_run([optimizer], samples = -1)
        # import pickle
        # pickle.dump((train_losses, eval_losses), open(f"experiments/optimizer_sweep/Analysis/Meta/two_losses_{optimizer}.pkl", "wb"))
        # #     plt.scatter(train_losses, eval_losses, label=optimizer, color=color_map[optimizer])
        # # plt.savefig("two_losses.png")
        import pickle
        train_losses, eval_losses = pickle.load(open(f"experiments/optimizer_sweep/Analysis/Meta/two_losses_{optimizer}.pkl", "rb"))
        plt.scatter(train_losses, eval_losses, label=correct_name[optimizer], color=color_map[optimizer], alpha=0.5)
    plt.legend(fontsize=12, ncol=2, loc="lower right")
    plt.xlabel("Training Loss", fontsize=20)
    plt.ylabel("Evaluation Loss on C4", fontsize=20)
    plt.title("Generalization Trend Across Optimizers", fontsize=20)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.tight_layout()
    plt.savefig("two_losses.pdf")
