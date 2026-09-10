#!/bin/sh
set -eu

# Write reference solution into the container solution file.
cat <<'PY' > /solution/solution.py
df['Date'] = df['Date'].dt.strftime('%d-%b-%Y')

PY
