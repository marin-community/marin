#!/bin/sh
set -eu

# Write reference solution into the container solution file.
cat <<'PY' > /solution/solution.py
    result = df.loc[df.index.isin(test)]

    return result

PY
