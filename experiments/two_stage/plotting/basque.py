import matplotlib.pyplot as plt
import numpy as np

# Use same styling as other plots
plt.rcParams.update({
    "font.family": "Palatino Linotype",
})

# Use same colors as other plots
LIGHT_BLUE = '#8CD9FF'

def create_replay_plot(replay_ratios, accuracies, model_name, output_path):
    """
    Creates a single plot showing Basque performance vs replay ratio.
    
    Args:
        replay_ratios: numpy array of replay ratios
        accuracies: numpy array of accuracy values
        model_name: string for plot title (e.g. "40M Model")
        output_path: path to save the plot
    """
    if model_name != "":
        model_name = "\n" + model_name

    plt.figure(figsize=(5, 4), dpi=300)
    
    # Add no-replay baseline dotted line first (so it's behind other elements)
    plt.axhline(y=accuracies[-1], color='black', linestyle=':', alpha=0.7,
                label='Standard fine-tuning')
    
    # Plot replay ratio line
    plt.plot(replay_ratios, accuracies, color=LIGHT_BLUE, marker='o',
            linewidth=2, markersize=6, label='SlimPajama Replay')
    
    # Add star at best performance
    best_idx = np.argmax(accuracies)
    plt.scatter(replay_ratios[best_idx], accuracies[best_idx], 
                color=LIGHT_BLUE, marker='*', s=200, zorder=10)
    
    # Add empty plot with label for base performance
    plt.plot([], [], color='none', label='Base: 0.54')
    
    plt.xlabel('Replay Fraction $\\rho$')
    plt.ylabel('Accuracy')
    
    # Automatically generate y-ticks at 0.01 intervals
    y_min = np.floor(np.min(accuracies) * 100) / 100
    y_max = np.ceil(np.max(accuracies) * 100) / 100
    if "40M" in model_name:
        y_min += 0.01
    y_ticks = np.arange(y_min - 0.01, y_max + 0.01, 0.01)
    plt.yticks(y_ticks)
    
    plt.title(f'Basque COPA Accuracy vs Replay Fraction{model_name}')
    plt.ylim(y_min - 0.01, y_max + 0.01)
    
    plt.legend(fontsize=10)
    plt.tight_layout()
    
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()

def plot_basque_comparison():
    """
    Creates two separate plots showing Basque performance vs replay ratio 
    for 40M and 200M models.
    """
    # 200M model data
    replay_ratios_200m = np.array([0.90, 0.75, 0.50, 0.00])
    accuracies_200m = np.array([0.648, 0.65, 0.646, 0.63])
    
    # 40M model data
    replay_ratios_40m = np.array([0.90, 0.75, 0.50, 0.00])
    accuracies_40m = np.array([0.588, 0.584, 0.596, 0.58])
    
    # Create separate plots
    create_replay_plot(
        replay_ratios_200m, 
        accuracies_200m,
        "200M Model",
        'plotting/plots/basque_replay_200M.png'
    )

    create_replay_plot(
        replay_ratios_200m, 
        accuracies_200m,
        "",
        'plotting/plots/basque_replay.png'
    )
    
    create_replay_plot(
        replay_ratios_40m,
        accuracies_40m,
        "40M Model",
        'plotting/plots/basque_replay_40M.png'
    )

if __name__ == "__main__":
    plot_basque_comparison()
