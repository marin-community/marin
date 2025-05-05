import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

# Define the power law function
def power_law(x, A, B, C, D):
    return A / ((x + D) ** B) + C

# Function to find multiplier given a loss value using algebra
def loss_to_multiplier(loss, A, B, C, D):
    # From: loss = A/((x + D)^B) + C
    # loss - C = A/((x + D)^B)
    # (x + D)^B = A/(loss - C)
    # x + D = (A/(loss - C))^(1/B)
    # x = (A/(loss - C))^(1/B) - D
    x = (A/(loss - C))**(1/B) - D
    # Convert to efficiency multiplier by dividing by 0.01
    return x / 0.01

rare_data = "flan"

# candidate losses: first number is yes replay yes early, second is no replay yes early, third is no replay no early

if rare_data == "finemath":
    multipliers = [0.50, 0.20, 0.10, 0.05, 0.02, 0.01, 0.001]
    losses = [3.20651, 3.3953, 3.62786, 3.82254, 4.10486, 4.30877, 4.40135]
    candidate_losses = [3.42796, 3.44682, 4.27805]
elif rare_data == "starcoder":
    multipliers = [0.50, 0.20, 0.10, 0.05, 0.02, 0.01, 0.001]
    losses = [2.01325, 2.41197, 2.86489, 3.25548, 4.18441, 4.07049, 5.38659]
    candidate_losses = [2.71784, 2.73632, 4.33719]
elif rare_data == "flan":
    multipliers = [0.50, 0.20, 0.10, 0.05, 0.02, 0.01, 0.001]
    losses = [3.3972, 3.56355, 3.54664, 3.70888, 3.79287, 4.05113, 4.53829]
    candidate_losses = [3.44237, 3.50337, 4.45659]

# Initial parameter guesses
p0 = [0.1, 0.2, 2.0, 0.0]  # Initial guesses for A, B, C, D

# Fit the power law with increased maxfev and initial guesses
popt, pcov = curve_fit(power_law, multipliers, losses, p0=p0, maxfev=10000)
A, B, C, D = popt

# Generate points for smooth curve plotting
x_smooth = np.logspace(np.log10(min(multipliers)), np.log10(max(multipliers)), 100)
y_smooth = power_law(x_smooth, A, B, C, D)

plt.figure(dpi=600)
plt.plot(multipliers, losses, 'o', label='Data points')
plt.plot(x_smooth, y_smooth, '-', label=f'Fit: {A:.3f}/(x+{D:.3f})^{B:.3f} + {C:.3f}')
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
print(f"D = {D:.3f}")

# Example usage of the utility
print("\nData efficiency multipliers for example losses:")
example_losses = candidate_losses
for loss in example_losses:
    efficiency = loss_to_multiplier(loss, A, B, C, D)
    print(f"Loss {loss:.3f} -> {efficiency:.2f}x data efficiency")