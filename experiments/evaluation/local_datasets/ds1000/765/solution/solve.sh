#!/bin/sh
set -eu

# Write reference solution into the container solution file.
cat <<'PY' > /solution/solution.py
kdtree = scipy.spatial.cKDTree(points)
_, result = kdtree.query(extraPoints)


PY
