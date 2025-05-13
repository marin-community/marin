import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors

# Use same styling as sweep.py
plt.rcParams.update({
    "font.family": "Palatino Linotype"
})

# Use same colors as sweep.py
LIGHT_BLUE = '#8CD9FF'
PURPLE = '#7030A0'
RED = '#FF4B4B'  # Adding red color for Mind2Web
ORANGE = '#FFA500'  # Adding orange color for weight decay

def plot_weblinx_comparison():
    """
    Creates two plots showing performance vs replay ratio and weight decay.
    """
    # Get data
    replay_ratios = np.array([0, 0.05, 0.1, 0.2, 0.35, 0.5, 0.6, 0.75, 0.9])
    weblinx_performance = np.array([32.860, 35.456, 35.620, 35.334, 34.674, 37.256, 0.000, 34.520, 31.174])
    ultrachat_performance = np.array([32.860, 33.214, 34.125, 32.486, 37.012, 35.162, 36.334, 36.245])
    mind2web_performance = np.array([32.860, 35.319, 36.333, 33.637, 34.010, 32.842, 33.180])
    
    weight_decays = np.array([0, 0.1, 0.2, 0.4, 0.6])
    weight_decay_performance = np.array([32.860, 34.184, 32.355, 33.987, 32.548])
    
    # Process replay ratio data
    weblinx_mask = weblinx_performance != 0
    replay_ratios_weblinx = replay_ratios[weblinx_mask]
    weblinx_performance = weblinx_performance[weblinx_mask]
    
    mind2web_ratios = replay_ratios[:7]
    ultrachat_ratios = np.concatenate([replay_ratios[:6], replay_ratios[7:]])
    
    # Calculate y-axis limits from all data
    all_performances = np.concatenate([
        weblinx_performance,
        ultrachat_performance,
        mind2web_performance,
        weight_decay_performance
    ])
    y_min = np.floor(np.min(all_performances))
    y_max = np.ceil(np.max(all_performances))
    
    # Plot replay ratio
    plt.figure(figsize=(5, 4), dpi=300)
    plt.plot(replay_ratios_weblinx, weblinx_performance, color=LIGHT_BLUE, marker='o', 
            linewidth=2, markersize=6, label='OpenHermes')
    plt.plot(ultrachat_ratios, ultrachat_performance, color=PURPLE, marker='o', 
            linewidth=2, markersize=6, label='UltraChat')
    plt.plot(mind2web_ratios, mind2web_performance, color=RED, marker='o',
            linewidth=2, markersize=6, label='Mind2Web')
    
    plt.xlabel('Replay Ratio')
    plt.ylabel('Performance')
    plt.title('WebLinx Performance vs Replay Ratio')
    plt.ylim(y_min, y_max)
    plt.legend()
    plt.tight_layout()
    
    plt.savefig('plotting/plots/weblinx_replay.png', bbox_inches='tight')
    plt.close()
    
    # Plot weight decay
    plt.figure(figsize=(5, 4), dpi=300)
    plt.plot(weight_decays, weight_decay_performance, color=ORANGE, marker='o',
            linewidth=2, markersize=6, label='Weight Decay')
    
    plt.xlabel('Weight Decay')
    plt.ylabel('Performance')
    plt.title('WebLinx Performance vs Weight Decay')
    plt.ylim(y_min, y_max)
    plt.legend()
    plt.tight_layout()
    
    plt.savefig('plotting/plots/weblinx_weight_decay.png', bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    plot_weblinx_comparison()
