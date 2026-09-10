#!/bin/sh
set -eu

# Write reference solution into the container solution file.
cat <<'PY' > /solution/solution.py
def fill_crop(img, pos, crop):
    img_shape, pos, crop_shape = np.array(img.shape), np.array(pos), np.array(crop.shape),
    end = pos+crop_shape
    # Calculate crop slice positions
    crop_low = np.clip(0 - pos, a_min=0, a_max=crop_shape)
    crop_high = crop_shape - np.clip(end-img_shape, a_min=0, a_max=crop_shape)
    crop_slices = (slice(low, high) for low, high in zip(crop_low, crop_high))
    # Calculate img slice positions
    pos = np.clip(pos, a_min=0, a_max=img_shape)
    end = np.clip(end, a_min=0, a_max=img_shape)
    img_slices = (slice(low, high) for low, high in zip(pos, end))
    crop[tuple(crop_slices)] = img[tuple(img_slices)]
    return crop
result = fill_crop(a, [low_index, low_index], np.zeros((high_index-low_index, high_index-low_index)))

PY
