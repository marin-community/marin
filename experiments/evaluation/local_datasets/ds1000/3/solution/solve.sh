#!/bin/sh
set -eu

# Write reference solution into the container solution file.
cat <<'PY' > /solution/solution.py
def g(df):
    return df.where(df.apply(lambda x: x.map(x.value_counts())) >= 3, "other")

result = g(df.copy())

PY
