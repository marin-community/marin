#!/bin/sh
set -eu

# Write reference solution into the container solution file.
cat <<'PY' > /solution/solution.py
n = np.arange(N + 1, dtype=np.int64)
dist = scipy.stats.binom(p=p, n=n)
result = dist.pmf(k=np.arange(N + 1, dtype=np.int64)[:, None]).T

PY
