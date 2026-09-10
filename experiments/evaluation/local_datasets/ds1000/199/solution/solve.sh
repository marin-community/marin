#!/bin/sh
set -eu

# Write reference solution into the container solution file.
cat <<'PY' > /solution/solution.py
def g(df):
    return pd.DataFrame(df.row.str.split(' ', 1).tolist(), columns=['fips', 'row'])

df = g(df.copy())

PY
