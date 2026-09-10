#!/bin/sh
set -eu

# Write reference solution into the container solution file.
cat <<'PY' > /solution/solution.py
import itertools as IT
for col1, col2 in IT.combinations(df.columns, 2):
    def tau(idx):
        B = df[[col1, col2]].iloc[idx]
        return stats.kendalltau(B[col1], B[col2])[0]
    df[col1+col2] = pd.Series(np.arange(len(df)), index=df.index).rolling(3).apply(tau)


PY
