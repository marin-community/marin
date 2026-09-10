#!/bin/sh
set -eu

# Write reference solution into the container solution file.
cat <<'PY' > /solution/solution.py
import numpy as np
def g(df):
    softmax = []
    min_max = []
    for i in range(len(df)):
        Min = np.inf
        Max = -np.inf
        exp_Sum = 0
        for j in range(len(df)):
            if df.loc[i, 'a'] == df.loc[j, 'a']:
                Min = min(Min, df.loc[j, 'b'])
                Max = max(Max, df.loc[j, 'b'])
                exp_Sum += np.exp(df.loc[j, 'b'])
        softmax.append(np.exp(df.loc[i, 'b']) / exp_Sum)
        min_max.append((df.loc[i, 'b'] - Min) / (Max - Min))
    df['softmax'] = softmax
    df['min-max'] = min_max
    return df

df = g(df.copy())

PY
