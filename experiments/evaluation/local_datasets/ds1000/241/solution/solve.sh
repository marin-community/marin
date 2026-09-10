#!/bin/sh
set -eu

# Write reference solution into the container solution file.
cat <<'PY' > /solution/solution.py
def g(C, D):
    return pd.concat([C,D]).drop_duplicates('A', keep='first').sort_values(by=['A']).reset_index(drop=True)

result = g(C.copy(),D.copy())

PY
