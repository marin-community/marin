#!/bin/sh
set -eu

# Write reference solution into the container solution file.
cat <<'PY' > /solution/solution.py
def g(df):
    return pd.pivot_table(df, values=['D','E'], index=['B'], aggfunc={'D':np.sum, 'E':np.mean})

result = g(df.copy())

PY
