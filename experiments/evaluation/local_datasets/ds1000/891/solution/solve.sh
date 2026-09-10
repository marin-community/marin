#!/bin/sh
set -eu

# Write reference solution into the container solution file.
cat <<'PY' > /solution/solution.py
from sklearn import preprocessing

pt = preprocessing.PowerTransformer(method="yeo-johnson")
yeo_johnson_data = pt.fit_transform(data)
PY
