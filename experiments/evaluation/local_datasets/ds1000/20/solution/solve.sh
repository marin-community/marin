#!/bin/sh
set -eu

# Write reference solution into the container solution file.
cat <<'PY' > /solution/solution.py
df["category"] = df.idxmax(axis=1)

PY
