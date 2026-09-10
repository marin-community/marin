#!/bin/sh
set -eu

# Write reference solution into the container solution file.
cat <<'PY' > /solution/solution.py
def g(df):
    total_len = len(df)
    zero_len = (df['Column_x'] == 0).sum()
    idx = df['Column_x'].index[df['Column_x'].isnull()]
    total_nan_len = len(idx)
    first_nan = (total_len // 2) - zero_len
    df.loc[idx[0:first_nan], 'Column_x'] = 0
    df.loc[idx[first_nan:total_nan_len], 'Column_x'] = 1
    return df

df = g(df.copy())

PY
