#!/bin/sh
set -eu

# Write reference solution into the container solution file.
cat <<'PY' > /solution/solution.py
result = np.array([fsolve(lambda a,x,b: eqn(x, a, b), x0=0.5, args=(x,b))[0] for x, b in zip(xdata, bdata)])
PY
