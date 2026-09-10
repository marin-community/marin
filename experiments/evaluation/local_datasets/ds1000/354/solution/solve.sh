#!/bin/sh
set -eu

# Write reference solution into the container solution file.
cat <<'PY' > /solution/solution.py
dims = np.maximum(B.max(0),A.max(0))+1
result = A[~np.in1d(np.ravel_multi_index(A.T,dims),np.ravel_multi_index(B.T,dims))]
output = np.append(result, B[~np.in1d(np.ravel_multi_index(B.T,dims),np.ravel_multi_index(A.T,dims))], axis = 0)

PY
