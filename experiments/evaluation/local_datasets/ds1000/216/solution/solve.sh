#!/bin/sh
set -eu

# Write reference solution into the container solution file.
cat <<'PY' > /solution/solution.py
def g(df):
    Date = list(df.index)
    Date = sorted(Date)
    half = len(list(Date)) // 2
    return max(Date, key=lambda v: Date.count(v)), Date[half]

mode_result,median_result = g(df.copy())

PY
