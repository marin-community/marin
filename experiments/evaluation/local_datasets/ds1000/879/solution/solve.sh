#!/bin/sh
set -eu

# Write reference solution into the container solution file.
cat <<'PY' > /solution/solution.py
from sklearn.preprocessing import MultiLabelBinarizer

new_features = MultiLabelBinarizer().fit_transform(features)
rows, cols = new_features.shape
for i in range(rows):
    for j in range(cols):
        if new_features[i, j] == 1:
            new_features[i, j] = 0
        else:
            new_features[i, j] = 1

PY
