#!/bin/sh
set -eu

# Write reference solution into the container solution file.
cat <<'PY' > /solution/solution.py
df = df[["celltype", "s1", "s2"]]
df.set_index(["celltype"], inplace=True)
df.plot(kind="bar", alpha=0.75, rot=45)
PY
