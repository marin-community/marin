#!/bin/sh
set -eu

# Write reference solution into the container solution file.
cat <<'PY' > /solution/solution.py
# def solve(softmax_output):
    y = torch.argmax(softmax_output, dim=1).view(-1, 1)
    # return y
# y = solve(softmax_output)


    return y

PY
