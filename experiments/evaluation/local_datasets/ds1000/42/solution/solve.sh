#!/bin/sh
set -eu

# Write reference solution into the container solution file.
cat <<'PY' > /solution/solution.py
def g(df):
    df.columns = np.concatenate([df.iloc[0, :2], df.columns[2:]])
    df = df.iloc[1:].reset_index(drop=True)
    return df

result = g(df.copy())

PY
