#!/bin/sh
set -eu

# Write reference solution into the container solution file.
cat <<'PY' > /solution/solution.py
def g(labels):
    t = tf.one_hot(indices=labels, depth=10, on_value=1, off_value=0, axis=-1)
    n = t.numpy()
    for i in range(len(n)):
        n[i] = n[i][::-1]
    return tf.constant(n)

result = g(labels.copy())

PY
