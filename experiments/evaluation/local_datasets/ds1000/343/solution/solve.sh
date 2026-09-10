#!/bin/sh
set -eu

# Write reference solution into the container solution file.
cat <<'PY' > /solution/solution.py
result = []
for value in X.T.flat:
    result.append(value)


PY
