#!/bin/sh
set -eu

# Write reference solution into the container solution file.
cat <<'PY' > /solution/solution.py
Temp = a.unfold(2, chunk_dim, 1)
tensors_31 = []
for i in range(Temp.shape[2]):
    tensors_31.append(Temp[:, :, i, :, :].view(1, 3, chunk_dim, 10, 1).numpy())
tensors_31 = torch.from_numpy(np.array(tensors_31))
PY
