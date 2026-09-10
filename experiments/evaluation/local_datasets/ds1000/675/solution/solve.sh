#!/bin/sh
set -eu

# Write reference solution into the container solution file.
cat <<'PY' > /solution/solution.py
def g(lengths):
    lengths_transposed = tf.expand_dims(lengths, 1)
    range = tf.range(0, 8, 1)
    range_row = tf.expand_dims(range, 0)
    mask = tf.less(range_row, lengths_transposed)
    result = tf.where(mask, tf.ones([4, 8]), tf.zeros([4, 8]))
    return result

result = g(lengths.copy())

PY
