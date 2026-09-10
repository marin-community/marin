#!/bin/sh
set -eu

# Write reference solution into the container solution file.
cat <<'PY' > /solution/solution.py
import numpy as np
def g(dict, df):
    df["Date"] = df["Member"].apply(lambda x: dict.get(x)).fillna(np.NAN)
    return df

df = g(dict.copy(),df.copy())

PY
