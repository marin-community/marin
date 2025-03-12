import wandb
# Initialize the WandB API
api = wandb.Api()
# Define your details
username = "stanford-mercury"
project = "optimizer-scaling"
thshold = 3e-3
# Retrieve the run directly using its full path

from matplotlib import pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

def grab_best_run(tags, want = 'eval/paloma/c4_en/loss'):
    filters = {"tags": {"$all": tags}, "state": "finished"}
    runs = api.runs(f"{username}/{project}", filters=filters)
    min_loss = 10000
    min_run = None
    for run in runs:
        loss = run.summary[want]
        if type(loss) is float and loss < min_loss:
            min_loss = loss
            min_run = run
    return run.history(keys=[want], pandas=True)    


def grab_all_runs(tags, want = 'eval/paloma/c4_en/loss'):
    filters = {"tags": {"$all": tags}, "state": "finished"}
    runs = api.runs(f"{username}/{project}", filters=filters)
    losses = []
    for run in runs:
        loss = run.summary[want]
        if type(loss) is float:
            losses.append(loss)
    return losses
def fit_and_plot_kde(tags, want='eval/paloma/c4_en/loss', fig=None, ax=None):
    """
    Fetch all final losses, fit a KDE, and plot histogram + KDE curve
    onto the provided matplotlib figure/axes (if any).
    """
    # Grab the final losses from W&B
    losses = grab_all_runs(tags, want=want)
    
    if not losses:
        print("No losses found for the specified tags.")
        return fig, ax
    
    # Create fig and ax if not provided
    if fig is None or ax is None:
        fig, ax = plt.subplots()

    # Convert to a NumPy array
    losses_array = np.array(losses)
    
    # Fit a Gaussian KDE to the data
    kde = gaussian_kde(losses_array)
    
    # Plot histogram of data
    ax.hist(losses_array, bins=15, density=True, alpha=0.5, label=tags[-1])
    
    # Plot the KDE curve on top
    x = np.linspace(losses_array.min(), losses_array.max(), 200)
    y = kde(x)
    ax.plot(x, y, 'r-', linewidth=2, label="KDE")
    
    # Add labels, title, legend
    ax.set_title("KDE of Final Losses")
    ax.set_xlabel("Loss")
    ax.set_ylabel("Density")
    ax.legend()

    return fig, ax


if __name__ == "__main__":
    # 1. Create a figure outside of the plotting function
    fig, ax = plt.subplots(figsize=(6, 4))

    # optimizers = ['adamw', 'cautious', 'kron', 'lion', 'mars', 'muon', 'nadamw', 'scion', 'soap', 'sophia']
    optimizers = ['muon']
    steps = '5k'
    model_size = '130k'
    for optimizer in optimizers:
        print(optimizer)
        all_losses = fit_and_plot_kde([steps, model_size, optimizers], fig = fig, ax = ax)
    plt.savefig(f"{steps}_{model_size}_c4.png")

