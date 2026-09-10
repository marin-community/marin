#!/bin/sh
set -eu

# Write reference solution into the container solution file.
cat <<'PY' > /solution/solution.py
Z = scipy.cluster.hierarchy.linkage(np.array(data_matrix), 'ward')
cluster_labels = scipy.cluster.hierarchy.cut_tree(Z, n_clusters=2).reshape(-1, ).tolist()
PY
