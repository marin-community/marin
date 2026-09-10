#!/bin/sh
set -eu

# Write reference solution into the container solution file.
cat <<'PY' > /solution/solution.py
import numpy as np
def g(df):
    sh = 0
    min_R2 = 0
    for i in range(len(df)):
        min_R2 += (df['#1'].iloc[i]-df['#2'].iloc[i])**2
    for i in range(len(df)):
        R2 = 0
        for j in range(len(df)):
            R2 += (df['#1'].iloc[j] - df['#2'].iloc[j]) ** 2
        if min_R2 > R2:
            sh = i
            min_R2 = R2
        df['#1'] = np.roll(df['#1'], shift=1)
    df['#1'] = np.roll(df['#1'], shift=sh)
    return df

df = g(df)

PY
