#!/bin/sh
set -eu

# Write reference solution into the container solution file.
cat <<'PY' > /solution/solution.py
def residual_ans(x, a, y):
    s = ((y - a.dot(x**2))**2).sum()
    return s
bounds = [[x, None] for x in x_lower_bounds]
out = scipy.optimize.minimize(residual_ans, x0=x0, args=(a, y), method= 'L-BFGS-B', bounds=bounds).x
PY
