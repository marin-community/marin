#!/bin/sh
set -eu

# Write reference solution into the container solution file.
cat <<'PY' > /solution/solution.py
x = a[:a.shape[0] // patch_size * patch_size, :a.shape[1] // patch_size * patch_size]
result = x.reshape(x.shape[0]//patch_size, patch_size, x.shape[1]// patch_size, patch_size).swapaxes(1, 2).transpose(1, 0, 2, 3).reshape(-1, patch_size, patch_size)


PY
