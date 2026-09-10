#!/bin/sh
set -eu

# Write reference solution into the container solution file.
cat <<'PY' > /solution/solution.py
output[:, mask[0].to(torch.bool), :] = clean_input_spectrogram[:, mask[0].to(torch.bool), :]
PY
