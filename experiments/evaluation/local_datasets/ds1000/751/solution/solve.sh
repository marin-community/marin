#!/bin/sh
set -eu

# Write reference solution into the container solution file.
cat <<'PY' > /solution/solution.py
    import itertools
    n = example_array.max()+1
    indexes = []
    for k in range(1, n):
        tmp = np.nonzero(example_array == k)
        tmp = np.asarray(tmp).T
        indexes.append(tmp)
    result = np.zeros((n-1, n-1))   
    for i, j in itertools.combinations(range(n-1), 2):
        d2 = scipy.spatial.distance.cdist(indexes[i], indexes[j], metric='sqeuclidean') 
        result[i, j] = result[j, i] = d2.min()**0.5

    return result

PY
