#!/bin/sh
set -eu

# Write reference solution into the container solution file.
cat <<'PY' > /solution/solution.py
def g(df):
    return df.set_index(['user','01/12/15']).stack().reset_index(name='value').rename(columns={'level_2':'others'})

df = g(df.copy())

PY
