#!/bin/sh
set -eu

# Write reference solution into the container solution file.
cat <<'PY' > /solution/solution.py
def g(df):
    cols = list(df)[1:]
    cols = cols[::-1]
    for idx in df.index:
        s = 0
        cnt = 0
        for col in cols:
            if df.loc[idx, col] != 0:
                s += df.loc[idx, col]
                cnt += 1
            df.loc[idx, col] = s / (max(cnt, 1))
    return df

df = g(df.copy())

PY
