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
    """
    A power law of the form:
    A / (x ** B) + C
    where C represents the asymptotic loss
    """
    return A / (x ** B) + C

# Define the bounded power law function
def bounded_power_law(x, A, B, C, D):
    """
    A bounded power law of the form:
    A / (x ** B + D) + C
    where:
    A = scale factor
    B = power
    C = vertical offset
    D = small x bound control (prevents infinity as x->0)
    """
    return A / (x ** B + D) + C

# Define the sigmoid function
def sigmoid(x, A, B, C, D):
    """
    A sigmoid of the form:
    A / (1 + np.exp(-B * (x - C))) + D
    where:
    A = amplitude
    B = steepness
    C = midpoint
    D = vertical offset
    """
    return A / (1 + np.exp(-B * (x - C))) + D

# Function to find multiplier given a loss value using algebra - power law version
def power_law_to_multiplier(loss, A, B, C):
    # From: loss = A/(x^B) + C
    # loss - C = A/(x^B)
    # x^B = A/(loss - C)
    # x = (A/(loss - C))^(1/B)
    x = (A/(loss - C))**(1/B)
    return x

# Function to find multiplier given a loss value using algebra - bounded power law version
def bounded_power_law_to_multiplier(loss, A, B, C, D):
    # From: loss = A/(x^B + D) + C
    # loss - C = A/(x^B + D)
    # x^B + D = A/(loss - C)
    # x^B = A/(loss - C) - D
    # x = (A/(loss - C) - D)^(1/B)
    x = (A/(loss - C) - D)**(1/B)
    return x

# Function to find multiplier given a loss value using algebra - sigmoid version
def sigmoid_to_multiplier(loss, A, B, C, D):
    # From: loss = A/(1 + exp(-B*(x - C))) + D
    # (loss - D) = A/(1 + exp(-B*(x - C)))
    # (loss - D)/A = 1/(1 + exp(-B*(x - C)))
    # 1/((loss - D)/A) = 1 + exp(-B*(x - C))
    # 1/((loss - D)/A) - 1 = exp(-B*(x - C))
    # ln(A/(loss - D) - 1) = -B*(x - C)
    # x = C - ln(A/(loss - D) - 1)/B
    return C - np.log(A/(loss - D) - 1)/B

parser = argparse.ArgumentParser()
parser.add_argument("--rare_data", type=str)
parser.add_argument("--fit_type", type=str, default="power_law", 
                    choices=["power_law", "bounded_power_law"])
args = parser.parse_args()

rare_data = args.rare_data

# candidate losses: first number is yes replay yes early, second is yes replay no early, third is no replay yes early, fourth is no replay, no early

if rare_data == "finemath":
    # multipliers = [0.20, 0.10, 0.05, 0.02, 0.01, 0.004]
    # losses = [2.62149, 2.66707, 2.80266, 3.10907, 3.38551, 3.87458]
    multipliers = [16.0/1024.0, 8.0/1024.0, 4.0/1024.0, 2.0/1024.0, 1.0/1024.0]
    losses = [2.99478, 3.16164, 3.39593, 3.49087, 3.62532]
    candidate_losses = [3.90261, 3.99417, 3.35965, 3.42762, 3.56935, 4.33786, 3.56935]
elif rare_data == "starcoder":
    multipliers = [16.0/1024.0, 8.0/1024.0, 4.0/1024.0, 2.0/1024.0, 1.0/1024.0]
    losses = [2.45525, 2.90404, 3.53084, 3.77065, 3.98357]
    candidate_losses = [4.49146, 4.53793, 3.02749, 3.02749, 3.26809]
elif rare_data == "flan":
    # multipliers = [0.50, 0.20, 0.10, 0.05, 0.02, 0.01, 0.001]
    # losses = [3.3972, 3.56355, 3.54664, 3.70888, 3.79287, 4.05113, 4.53829]
    multipliers = [16.0/1024.0, 8.0/1024.0, 4.0/1024.0, 2.0/1024.0, 1.0/1024.0]
    losses = [3.2712, 3.27876, 3.37226, 3.41444, 3.54749]
    candidate_losses = [3.51, 3.65354, 3.29485, 3.35773, 3.44407]
elif rare_data == "spj":
    multipliers = [16.0/1024.0, 8.0/1024.0, 4.0/1024.0, 2.0/1024.0, 1.0/1024.0]
    losses = []
    candidate_losses = [3.90, 3.93, 3.93, 4.13]

# Fit the curve based on chosen type
if args.fit_type == "power_law":
    p0 = [0.1, 0.2, 2.0]  # Initial guesses for A, B, C
    popt, pcov = curve_fit(power_law, multipliers, losses, p0=p0, maxfev=10000)
    A, B, C = popt
    D = None
    fit_func = power_law
    to_multiplier = power_law_to_multiplier
    fit_label = "Best power law fit"
else:  # bounded power law
    p0 = [0.1, 0.2, 2.0, 0.001]  # Initial guesses for A, B, C, D
    popt, pcov = curve_fit(bounded_power_law, multipliers, losses, p0=p0, maxfev=10000)
    A, B, C, D = popt
    fit_func = bounded_power_law
    to_multiplier = bounded_power_law_to_multiplier
    fit_label = "Best bounded power law fit"

# Calculate excess loss (loss - asymptotic loss C)
excess_losses = [loss - C for loss in losses]
log_multipliers = np.log(multipliers)
log_excess = np.log(excess_losses)

# Generate points for smooth curve plotting
x_smooth = np.logspace(np.log10(min(multipliers)), np.log10(max(multipliers)), 100)
y_smooth = fit_func(x_smooth, *popt)
excess_smooth = y_smooth - C
log_x_smooth = np.log(x_smooth)
log_excess_smooth = np.log(excess_smooth)

# Create figure with updated styling
plt.figure(figsize=(6, 3), dpi=600)
plt.plot(log_multipliers, log_excess, 'o', color=PURPLE, label='Data points')
plt.plot(log_x_smooth, log_excess_smooth, '-', color=LIGHT_BLUE, label=fit_label)
plt.xlabel(f"log(Rare Fraction)")
plt.ylabel("log(Excess Loss)")
plt.title(f"{pretty_name_dict[rare_data]} log(Excess Loss) vs log(Rare Fraction)")
plt.grid(True, alpha=0.3)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig(f"plotting/plots/log_excess_loss_vs_fraction_{rare_data}.png", bbox_inches='tight')

print(f"Best fit parameters:")
print(f"A = {A:.3f}")
print(f"B = {B:.3f}")
print(f"C (asymptotic loss) = {C:.3f}")
if D is not None:
    print(f"D = {D:.3f}")

# Example usage of the utility
print(f"\nData efficiency multipliers for {rare_data} with {args.fit_type.title()} fit:")
example_losses = candidate_losses

mults = []
for loss in example_losses:
    if args.fit_type == "power_law":
        mult = to_multiplier(loss, A, B, C)
    elif args.fit_type == "bounded_power_law":
        mult = to_multiplier(loss, A, B, C, D)
    else:
        mult = sigmoid_to_multiplier(loss, A, B, C, D)
    mults.append(mult)
    # print(f"Loss: {loss}, Multiplier: {mult}")

print(f"SFT Data efficiency multiplier: {mults[0] / mults[1]}")
print(f"Mid-training data efficiency multiplier: {mults[4] / mults[1]}")
print(f"Replay data efficiency multiplier: {mults[3] / mults[4]}")
print(f"Full data efficiency multiplier: {mults[2] / mults[4]}")

if rare_data == "finemath":
    print(f"Candidate data efficiency multiplier: {mults[6] / mults[5]}")

# Print the slope of the log-log plot
print(f"\nSlope of log-log plot (should match -B): {-B:.3f}")
