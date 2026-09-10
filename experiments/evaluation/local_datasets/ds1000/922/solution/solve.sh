#!/bin/sh
set -eu

# Write reference solution into the container solution file.
cat <<'PY' > /solution/solution.py
n = features_dataframe.shape[0]
train_size = 0.8
test_size = 1 - train_size + 0.005
train_dataframe = features_dataframe.iloc[int(n * test_size):]
test_dataframe = features_dataframe.iloc[:int(n * test_size)]
PY
