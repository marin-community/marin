#!/bin/sh
set -eu

# Write reference solution into the container solution file.
cat <<'PY' > /solution/solution.py
import numpy as np
def g(df):
    df["new"] = df.apply(lambda p: sum( not q.isalpha() for q in p["str"] ), axis=1)
    df["new"] = df["new"].replace(0, np.NAN)
    return df

df = g(df.copy())

PY
