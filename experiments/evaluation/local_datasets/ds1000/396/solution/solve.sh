#!/bin/sh
set -eu

# Write reference solution into the container solution file.
cat <<'PY' > /solution/solution.py
import scipy.stats
result = scipy.stats.loguniform.rvs(a = np.exp(min), b = np.exp(max), size = n)


PY
