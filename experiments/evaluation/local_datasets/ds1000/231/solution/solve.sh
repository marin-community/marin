#!/bin/sh
set -eu

# Write reference solution into the container solution file.
cat <<'PY' > /solution/solution.py
def g(df, bins):
    groups = df.groupby(['username', pd.cut(df.views, bins)])
    return groups.size().unstack()

result = g(df.copy(),bins.copy())

PY
