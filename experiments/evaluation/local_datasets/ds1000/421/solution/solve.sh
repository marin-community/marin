#!/bin/sh
set -eu

# Write reference solution into the container solution file.
cat <<'PY' > /solution/solution.py
from scipy.special import comb

def smoothclamp(x, x_min=0, x_max=1, N=1):
    if x < x_min:
        return x_min
    if x > x_max:
        return x_max
    x = np.clip((x - x_min) / (x_max - x_min), 0, 1)

    result = 0
    for n in range(0, N + 1):
        result += comb(N + n, n) * comb(2 * N + 1, N - n) * (-x) ** n

    result *= x ** (N + 1)
    return result


PY
