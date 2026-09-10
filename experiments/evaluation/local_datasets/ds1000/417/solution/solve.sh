#!/bin/sh
set -eu

# Write reference solution into the container solution file.
cat <<'PY' > /solution/solution.py
new_data = data[::-1]
bin_data_mean = new_data[:(data.size // bin_size) * bin_size].reshape(-1, bin_size).mean(axis=1)


PY
