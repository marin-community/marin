#!/bin/sh
set -eu

# Write reference solution into the container solution file.
cat <<'PY' > /solution/solution.py
def g(df):
    df.index += 1
    df_out = df.stack()
    df.index -= 1
    df_out.index = df_out.index.map('{0[1]}_{0[0]}'.format)
    return df_out.to_frame().T

df = g(df.copy())

PY
