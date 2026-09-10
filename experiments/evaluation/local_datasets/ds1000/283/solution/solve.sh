#!/bin/sh
set -eu

# Write reference solution into the container solution file.
cat <<'PY' > /solution/solution.py
def g(df):
    return df.set_axis(['Test', *df.columns[1:]], axis=1, inplace=False)

result = g(df.copy())

PY
