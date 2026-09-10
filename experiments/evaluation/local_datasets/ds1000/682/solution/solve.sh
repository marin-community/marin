#!/bin/sh
set -eu

# Write reference solution into the container solution file.
cat <<'PY' > /solution/solution.py
def g(a):
    return tf.squeeze(a)

result = g(a.__copy__())

PY
