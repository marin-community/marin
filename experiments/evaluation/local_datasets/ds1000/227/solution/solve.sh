#!/bin/sh
set -eu

# Write reference solution into the container solution file.
cat <<'PY' > /solution/solution.py
def g(a,b,c):
    return pd.DataFrame(np.rec.fromarrays((a.values, b.values, c.values)).tolist(),columns=a.columns,index=a.index)

result = g(a.copy(),b.copy(), c.copy())

PY
