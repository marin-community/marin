#!/bin/sh
set -eu

# Write reference solution into the container solution file.
cat <<'PY' > /solution/solution.py
def g(df):
    return pd.melt(df)

result = g(df.copy())

PY
