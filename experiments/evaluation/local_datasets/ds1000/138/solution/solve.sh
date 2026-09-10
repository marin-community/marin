#!/bin/sh
set -eu

# Write reference solution into the container solution file.
cat <<'PY' > /solution/solution.py
def g(df):
    return df[df.groupby(['Sp', 'Value'])['count'].transform(max) == df['count']]

result = g(df.copy())

PY
