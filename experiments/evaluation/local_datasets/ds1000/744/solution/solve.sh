#!/bin/sh
set -eu

# Write reference solution into the container solution file.
cat <<'PY' > /solution/solution.py
def filter_isolated_cells(array, struct):
    filtered_array = np.copy(array)
    id_regions, num_ids = scipy.ndimage.label(filtered_array, structure=struct)
    id_sizes = np.array(scipy.ndimage.sum(array, id_regions, range(num_ids + 1)))
    area_mask = (id_sizes == 1)
    filtered_array[area_mask[id_regions]] = 0
    return filtered_array
arr = np.sign(square)
filtered_array = filter_isolated_cells(arr, struct=np.ones((3,3)))
square = np.where(filtered_array==1, square, 0)
PY
