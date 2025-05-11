import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import argparse
import matplotlib.colors as mcolors

# Adopt plotting standards from sweep.py
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

# Define the power law function
def power_law(x, A, B, C):
    return A / ((x) ** B) + C

# Function to find multiplier given a loss value using algebra
def loss_to_multiplier(loss, A, B, C):
    # From: loss = A/((x + D)^B) + C
    # loss - C = A/((x + D)^B)
    # (x + D)^B = A/(loss - C)
    # x + D = (A/(loss - C))^(1/B)
    # x = (A/(loss - C))^(1/B) - D
    x = (A/(loss - C))**(1/B)
    # Convert to efficiency multiplier by dividing by 0.01
    return x / 0.004

parser = argparse.ArgumentParser()
parser.add_argument("--rare_data", type=str)
args = parser.parse_args()

rare_data = args.rare_data

# candidate losses: first number is yes replay yes early, second is yes replay no early, third is no replay yes early, fourth is no replay, no early

if rare_data == "finemath":
    multipliers = [0.20, 0.10, 0.05, 0.02, 0.01, 0.004]
    losses = [2.62149, 2.66707, 2.80266, 3.10907, 3.38551, 3.87458]
    candidate_losses = [3.25, 3.25, 3.42, 3.87]
elif rare_data == "starcoder":
    multipliers = [0.20, 0.10, 0.05, 0.01]
    losses = [1.56766, 1.66453, 1.84449, 2.8498]
    candidate_losses = [2.54, 2.54, 2.64, 3.85]
elif rare_data == "flan":
    multipliers = [0.50, 0.20, 0.10, 0.05, 0.02, 0.01, 0.001]
    losses = [3.3972, 3.56355, 3.54664, 3.70888, 3.79287, 4.05113, 4.53829]
    candidate_losses = [3.43919, 3.44237, 3.50337, 4.45659]
elif rare_data == "spj":
    multipliers = [0.20, 0.10, 0.05, 0.02, 0.01, 0.004]
    losses = [3.73001, 3.75928, 3.80438, 3.90182, 3.98626, 4.12748]
    candidate_losses = [3.90, 3.93, 3.93, 4.13]

# Initial parameter guesses
p0 = [0.1, 0.2, 2.0]  # Initial guesses for A, B, C

# Fit the power law with increased maxfev and initial guesses
popt, pcov = curve_fit(power_law, multipliers, losses, p0=p0, maxfev=10000)
A, B, C = popt

# Generate points for smooth curve plotting
x_smooth = np.logspace(np.log10(min(multipliers)), np.log10(max(multipliers)), 100)
y_smooth = power_law(x_smooth, A, B, C)

# Create figure with updated styling
plt.figure(figsize=(6, 3), dpi=600)
plt.plot(multipliers, losses, 'o', color=PURPLE, label='Data points')
plt.plot(x_smooth, y_smooth, '-', color=LIGHT_BLUE, label=f'Best power law fit')
plt.xscale('log')
plt.yscale('log')
plt.xlabel(f"{pretty_name_dict[rare_data]} fraction")
plt.ylabel("Loss")
plt.title(f"{pretty_name_dict[rare_data]} Loss vs Multiplier with Power Law Fit")
plt.grid(True, alpha=0.3)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig(f"plotting/plots/loss_vs_multiplier_{rare_data}.png", bbox_inches='tight')

print(f"Best fit parameters:")
print(f"A = {A:.3f}")
print(f"B = {B:.3f}")
print(f"C = {C:.3f}")

# Example usage of the utility
print("\nData efficiency multipliers for example losses:")
example_losses = candidate_losses

print("Yes replay yes early DEX:", loss_to_multiplier(example_losses[0], A, B, C))
print("Yes replay no early DEX:", loss_to_multiplier(example_losses[1], A, B, C))
print("No replay yes early DEX:", loss_to_multiplier(example_losses[2], A, B, C))
print("No replay no early DEX:", loss_to_multiplier(example_losses[3], A, B, C))
