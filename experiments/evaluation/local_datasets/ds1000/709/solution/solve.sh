#!/bin/sh
set -eu

# Write reference solution into the container solution file.
cat <<'PY' > /solution/solution.py
    tf.random.set_seed(seed_x)
    result = tf.random.uniform(shape=(10,), minval=1, maxval=5, dtype=tf.int32)

    return result

PY
