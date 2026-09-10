#!/bin/sh
set -eu

# Write reference solution into the container solution file.
cat <<'PY' > /solution/solution.py
# def solve(data):
    ### BEGIN SOLUTION
    result = pd.DataFrame(data=np.c_[data['data'], data['target']], columns=data['feature_names'] + ['target'])
    ### END SOLUTION
    # return result
# data1 = solve(data)

    return result

PY
