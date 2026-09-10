#!/bin/sh
set -eu

# Write reference solution into the container solution file.
cat <<'PY' > /solution/solution.py
def g(df):
    idx = df['Column_x'].index[df['Column_x'].isnull()]
    total_nan_len = len(idx)
    first_nan = (total_nan_len * 3) // 10
    middle_nan = (total_nan_len * 3) // 10
    df.loc[idx[0:first_nan], 'Column_x'] = 0
    df.loc[idx[first_nan:first_nan + middle_nan], 'Column_x'] = 0.5
    df.loc[idx[first_nan + middle_nan:total_nan_len], 'Column_x'] = 1
    return df

df = g(df.copy())

PY
