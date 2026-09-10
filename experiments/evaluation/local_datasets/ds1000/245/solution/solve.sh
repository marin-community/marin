#!/bin/sh
set -eu

# Write reference solution into the container solution file.
cat <<'PY' > /solution/solution.py
def g(df):
    return df.groupby('user')[['time', 'amount']].apply(lambda x: x.values.tolist()[::-1]).to_frame(name='amount-time-tuple')

result = g(df.copy())

PY
