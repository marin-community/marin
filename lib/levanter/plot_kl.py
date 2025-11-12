

#!/usr/bin/env python3
# Plot y = exp(x) - 1 - x for x in [-100, 100]

import argparse
import numpy as np
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser(description="Plot y = exp(x) - 1 - x")
    parser.add_argument("--min", type=float, default=-100.0, help="Minimum x value")
    parser.add_argument("--max", type=float, default=100.0, help="Maximum x value")
    args = parser.parse_args()

    x_min = args.min
    x_max = args.max

    # Domain and function
    x = np.linspace(x_min, x_max, 10001, dtype=np.float64)

    # Compute y with overflow handling
    with np.errstate(over='ignore', invalid='ignore'):
        y = np.exp(x) - 1.0 - x
        # Replace inf/nan with large finite values for plotting
        y = np.where(np.isfinite(y), y, np.sign(y) * 1e308)

    # Sanity prints at a few points
    sample_points = np.linspace(x_min, x_max, 5)
    for xi in sample_points:
        with np.errstate(over='ignore'):
            yi = float(np.exp(xi) - 1.0 - xi)
        print(f"x={xi:>6.1f} -> y={yi:.6e}")

    # Plot
    plt.figure()
    plt.plot(x, y)
    plt.xlim(x_min, x_max)
    plt.xlabel("x")
    plt.ylabel("y = exp(x) - 1 - x")
    plt.title(f"y = exp(x) - 1 - x over x ∈ [{x_min}, {x_max}]")
    plt.yscale('symlog')  # Use symmetric log scale to handle both positive and negative values
    plt.grid(True, linestyle=":")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()