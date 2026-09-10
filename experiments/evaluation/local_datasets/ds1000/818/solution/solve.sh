#!/bin/sh
set -eu

# Write reference solution into the container solution file.
cat <<'PY' > /solution/solution.py
data1 = pd.DataFrame(data=np.c_[data['data'], data['target']], columns=data['feature_names'] + ['target'])
PY
