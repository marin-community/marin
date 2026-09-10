#!/bin/sh
set -eu

# Write reference solution into the container solution file.
cat <<'PY' > /solution/solution.py
    ds = tf.data.Dataset.from_tensor_slices(input)
    ds = ds.flat_map(lambda x: tf.data.Dataset.from_tensor_slices([x, x + 1, x + 2]))
    element = tf.compat.v1.data.make_one_shot_iterator(ds).get_next()


    result = []
    with tf.compat.v1.Session() as sess:
        for _ in range(9):
            result.append(sess.run(element))

    return result

PY
