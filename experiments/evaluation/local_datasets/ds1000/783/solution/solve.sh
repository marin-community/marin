#!/bin/sh
set -eu

# Write reference solution into the container solution file.
cat <<'PY' > /solution/solution.py
xs, ys = np.indices(shape)
xs = xs.reshape(shape[0] * shape[1], 1)
ys = ys.reshape(shape[0] * shape[1], 1)
X = np.hstack((xs, ys))
mid_x, mid_y = (shape[0]-1)/2.0, (shape[1]-1)/2.0
result = distance.cdist(X, np.atleast_2d([mid_x, mid_y]), 'minkowski', p=1).reshape(shape)



PY
