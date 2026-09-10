#!/bin/sh
set -eu

# Write reference solution into the container solution file.
cat <<'PY' > /solution/solution.py
for i in range(len(mask[0])):
    if mask[0][i] == 1:
        mask[0][i] = 0
    else:
        mask[0][i] = 1
output[:, mask[0].to(torch.bool), :] = clean_input_spectrogram[:, mask[0].to(torch.bool), :]
PY
