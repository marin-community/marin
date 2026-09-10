#!/bin/sh
set -eu

# Write reference solution into the container solution file.
cat <<'PY' > /solution/solution.py
result = (~np.isclose(s1,s2, equal_nan=True)).sum()

PY
