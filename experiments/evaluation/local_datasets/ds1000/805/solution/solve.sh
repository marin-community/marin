#!/bin/sh
set -eu

# Write reference solution into the container solution file.
cat <<'PY' > /solution/solution.py
def find_k_closest(centroids, data, k=1, distance_norm=2):
    kdtree = scipy.spatial.cKDTree(data)
    distances, indices = kdtree.query(centroids, k, p=distance_norm)
    if k > 1:
        indices = indices[:,-1]
    values = data[indices]
    return indices, values
result, _ = find_k_closest(centroids, data, k)
PY
