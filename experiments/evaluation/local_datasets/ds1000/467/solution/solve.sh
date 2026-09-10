#!/bin/sh
set -eu

# Write reference solution into the container solution file.
cat <<'PY' > /solution/solution.py
def window(arr, shape=(3, 3)):
    ans = []
    # Find row and column window sizes
    r_win = np.floor(shape[0] / 2).astype(int)
    c_win = np.floor(shape[1] / 2).astype(int)
    x, y = arr.shape
    for i in range(x):
        xmin = max(0, i - r_win)
        xmax = min(x, i + r_win + 1)
        for j in range(y):
            ymin = max(0, j - c_win)
            ymax = min(y, j + c_win + 1)
            ans.append(arr[xmin:xmax, ymin:ymax])
    return ans

result = window(a, size)
PY
