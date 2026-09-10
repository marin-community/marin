#!/bin/sh
set -eu

# Write reference solution into the container solution file.
cat <<'PY' > /solution/solution.py
def g(df):
    s = ''
    for c in df.columns:
        s += "---- %s ---" % c
        s += "\n"
        s += str(df[c].value_counts())
        s += "\n"
    return s

result = g(df.copy())

PY
