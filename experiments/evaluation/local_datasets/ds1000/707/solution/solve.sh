#!/bin/sh
set -eu

# Write reference solution into the container solution file.
cat <<'PY' > /solution/solution.py
def g(seed_x):
    tf.random.set_seed(seed_x)
    return tf.random.uniform(shape=(10,), minval=1, maxval=5, dtype=tf.int32)

result = g(seed_x)

PY
