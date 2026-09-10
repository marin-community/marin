#!/bin/sh
set -eu

# Write reference solution into the container solution file.
cat <<'PY' > /solution/solution.py
    result = a.argmax()

    return result

PY
