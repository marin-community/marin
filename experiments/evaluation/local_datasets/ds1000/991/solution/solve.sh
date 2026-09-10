#!/bin/sh
set -eu

# Write reference solution into the container solution file.
cat <<'PY' > /solution/solution.py
maxs = torch.max(torch.abs(x), torch.abs(y))

xSigns = (maxs == torch.abs(x)) * torch.sign(x)
ySigns = (maxs == torch.abs(y)) * torch.sign(y)
finalSigns = xSigns.int() | ySigns.int()

signed_max = maxs * finalSigns
PY
