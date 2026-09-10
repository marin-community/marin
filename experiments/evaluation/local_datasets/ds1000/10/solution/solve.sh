#!/bin/sh
set -eu

# Write reference solution into the container solution file.
cat <<'PY' > /solution/solution.py
def g(df):
    if len(df.columns) == 1:
        if df.values.size == 1: return df.values[0][0]
        return df.values.squeeze()
    grouped = df.groupby(df.columns[0])
    d = {k: g(t.iloc[:, 1:]) for k, t in grouped}
    return d

result = g(df.copy())

PY
