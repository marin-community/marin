#!/bin/sh
set -eu

# Write reference solution into the container solution file.
cat <<'PY' > /solution/solution.py
def g(df):
    df['label'] = df.Close.diff().fillna(1).gt(0).astype(int)
    return df

df = g(df.copy())

PY
