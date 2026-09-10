#!/bin/sh
set -eu

# Write reference solution into the container solution file.
cat <<'PY' > /solution/solution.py
def g(df, s):
    spike_cols = [col for col in df.columns if s in col and col != s]
    return df[spike_cols]

result = g(df.copy(),s)

PY
