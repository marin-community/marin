#!/bin/sh
set -eu

# Write reference solution into the container solution file.
cat <<'PY' > /solution/solution.py
dtype = [('a','int32'), ('b','float32'), ('c','float32')]
values = np.zeros(2, dtype=dtype)
df = pd.DataFrame(values, index=index)


PY
