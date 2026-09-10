#!/bin/sh
set -eu

# Write reference solution into the container solution file.
cat <<'PY' > /solution/solution.py
def g(df):
    df.dt = pd.to_datetime(df.dt)
    return df.set_index(['dt', 'user']).unstack(fill_value=233).asfreq('D', fill_value=233).stack().sort_index(level=1).reset_index()

result = g(df.copy())

PY
