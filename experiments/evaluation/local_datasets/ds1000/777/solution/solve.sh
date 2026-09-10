#!/bin/sh
set -eu

# Write reference solution into the container solution file.
cat <<'PY' > /solution/solution.py
result = pd.DataFrame(data=stats.zscore(df, axis = 1), index=df.index, columns=df.columns)


PY
