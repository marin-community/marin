#!/bin/sh
set -eu

# Write reference solution into the container solution file.
cat <<'PY' > /solution/solution.py
C = scipy.spatial.distance.cdist(points1, points2)
_, result = scipy.optimize.linear_sum_assignment(C)



PY
