#!/bin/sh
set -eu

# Write reference solution into the container solution file.
cat <<'PY' > /solution/solution.py
mean = col.mean()
N = col.shape[0]
sqr = col.copy()  # take a copy of the col
sqr.data **= 2  # square the data, i.e. just the non-zero data
standard_deviation = np.sqrt(sqr.sum() / N - col.mean() ** 2)

PY
