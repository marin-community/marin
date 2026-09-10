#!/bin/sh
set -eu

# Write reference solution into the container solution file.
cat <<'PY' > /solution/solution.py
result = np.lib.stride_tricks.sliding_window_view(a, window_shape=(2,2)).reshape(-1, 2, 2)

PY
