#!/bin/sh
set -eu

# Write reference solution into the container solution file.
cat <<'PY' > /solution/solution.py
def g(A):
    return tf.reduce_sum(A, 1)

result = g(A.__copy__())

PY
