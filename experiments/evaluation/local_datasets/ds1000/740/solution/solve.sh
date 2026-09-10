#!/bin/sh
set -eu

# Write reference solution into the container solution file.
cat <<'PY' > /solution/solution.py
blobs = img > threshold
labels, nlabels = ndimage.label(blobs)
r, c = np.vstack(ndimage.center_of_mass(img, labels, np.arange(nlabels) + 1)).T
# find their distances from the top-left corner
d = np.sqrt(r * r + c * c)
result = sorted(d)

PY
