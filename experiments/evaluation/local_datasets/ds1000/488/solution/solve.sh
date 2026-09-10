#!/bin/sh
set -eu

# Write reference solution into the container solution file.
cat <<'PY' > /solution/solution.py
scaler = MinMaxScaler()
result = np.zeros_like(a)
for i, arr in enumerate(a):
    a_one_column = arr.reshape(-1, 1)
    result_one_column = scaler.fit_transform(a_one_column)
    result[i, :, :] = result_one_column.reshape(arr.shape)


PY
