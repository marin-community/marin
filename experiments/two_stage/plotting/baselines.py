import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import argparse

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
    return x / 0.01

parser = argparse.ArgumentParser()
parser.add_argument("--rare_data", type=str)
args = parser.parse_args()

rare_data = args.rare_data

# candidate losses: first number is yes replay yes early, second is yes replay no early, third is no replay yes early, fourth is no replay, no early

if rare_data == "finemath":
    multipliers = [1.00, 0.50, 0.20, 0.10, 0.05]
    losses = [2.80465, 2.94667, 3.16246, 3.47479, 3.52882]
    candidate_losses = [3.42796, 3.42796, 3.44682, 4.27805]
elif rare_data == "starcoder":
    multipliers = [0.50, 0.20, 0.10, 0.05, 0.02, 0.01, 0.001]
    losses = [2.01325, 2.41197, 2.86489, 3.25548, 4.18441, 4.07049, 5.38659]
    candidate_losses = [2.71526, 2.71784, 2.73632, 4.33719]
elif rare_data == "flan":
    multipliers = [0.50, 0.20, 0.10, 0.05, 0.02, 0.01, 0.001]
    losses = [3.3972, 3.56355, 3.54664, 3.70888, 3.79287, 4.05113, 4.53829]
    candidate_losses = [3.43919, 3.44237, 3.50337, 4.45659]
elif rare_data == "spj":
    multipliers = [1.00, 0.50, 0.20, 0.10, 0.05, 0.01]
    losses = [4.00906, 4.04337, 4.08126, 4.14078, 4.21707, 4.29219]
    candidate_losses = [3.92689, 3.93889, 3.92689, 4.29819]

# Initial parameter guesses
p0 = [0.1, 0.2, 2.0]  # Initial guesses for A, B, C

# Fit the power law with increased maxfev and initial guesses
popt, pcov = curve_fit(power_law, multipliers, losses, p0=p0, maxfev=10000)
A, B, C = popt

# Generate points for smooth curve plotting
x_smooth = np.logspace(np.log10(min(multipliers)), np.log10(max(multipliers)), 100)
y_smooth = power_law(x_smooth, A, B, C)

plt.figure(dpi=600)
plt.plot(multipliers, losses, 'o', label='Data points')
plt.plot(x_smooth, y_smooth, '-', label=f'Fit: {A:.3f}/(x)^{B:.3f} + {C:.3f}')
plt.xscale('log')
plt.yscale('log')
plt.xlabel(f"Multiplier after {rare_data}")
plt.ylabel("Loss")
plt.title("Loss vs Multiplier with Power Law Fit")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(f"experiments/two_stage/plotting/plots/loss_vs_multiplier_{rare_data}.png")

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
