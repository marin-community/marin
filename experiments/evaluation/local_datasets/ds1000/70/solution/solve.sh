#!/bin/sh
set -eu

# Write reference solution into the container solution file.
cat <<'PY' > /solution/solution.py
    result = df.loc[df['c']>0.5,columns].to_numpy()

    return result

PY
