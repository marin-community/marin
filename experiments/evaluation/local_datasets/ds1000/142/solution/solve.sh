#!/bin/sh
set -eu

# Write reference solution into the container solution file.
cat <<'PY' > /solution/solution.py
def g(df):
    result = pd.melt(df, value_vars=df.columns.tolist())
    cols = result.columns[:-1]
    for idx in result.index:
        t = result.loc[idx, cols]
        for i in range(len(cols)):
            result.loc[idx, cols[i]] = t[cols[-i-1]]
    return result

result = g(df.copy())

PY
