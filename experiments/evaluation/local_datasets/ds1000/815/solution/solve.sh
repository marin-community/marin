#!/bin/sh
set -eu

# Write reference solution into the container solution file.
cat <<'PY' > /solution/solution.py
res = signal.argrelextrema(arr, np.less_equal, order=n, axis = 1)
result = np.zeros((res[0].shape[0], 2)).astype(int)
result[:, 0] = res[0]
result[:, 1] = res[1]


PY
