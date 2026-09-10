#!/bin/sh
set -eu

# Write reference solution into the container solution file.
cat <<'PY' > /solution/solution.py
def g(params):
    import numpy as np
    a, b, c = params
    return ((a+b-c)-2)**2 + ((3*a-b-c))**2 + np.sin(b) + np.cos(b) + 4

res = optimize.minimize(g, initial_guess)
result = res.x
PY
