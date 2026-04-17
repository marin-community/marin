#!/usr/bin/env python
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Verify the data permutation is device-independent.

The Feistel permutation maps sequential indices to shuffled cache indices.
This only depends on seed and dataset length — NOT device count.

But the real question is: does the DataLoader's batch construction
assign the SAME global indices to batch 0 regardless of device count?

We test this by simulating the loader's index computation for both
4-device and 8-device meshes.
"""

import numpy as np
import jax

from levanter.data._prp import Permutation

# Dataset size: 2,250 examples (from the tokenized cache)
DATASET_SIZE = 2250
BATCH_SIZE = 64
SEED = 0

# Create the same permutation used in training
key = jax.random.PRNGKey(SEED)
perm = Permutation.make("feistel", DATASET_SIZE, key)

print(f"Dataset size: {DATASET_SIZE}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Seed: {SEED}")
print("Permutation type: feistel")

# The permutation maps sequential -> shuffled indices
print("\n=== Permutation: sequential -> cache indices ===")
first_batch_seq = np.arange(BATCH_SIZE)
first_batch_cache = perm(first_batch_seq)
print("Batch 0: seq[0:64] -> cache indices:")
print(f"  First 10: {first_batch_cache[:10]}")
print(f"  Last 5:   {first_batch_cache[-5:]}")
print(f"  Hash:     {hash(tuple(first_batch_cache.tolist()))}")

second_batch_seq = np.arange(BATCH_SIZE, 2 * BATCH_SIZE)
second_batch_cache = perm(second_batch_seq)
print("\nBatch 1: seq[64:128] -> cache indices:")
print(f"  First 10: {second_batch_cache[:10]}")
print(f"  Hash:     {hash(tuple(second_batch_cache.tolist()))}")

# Now the critical part: the DataLoader computes global_offset = step * batch_size
# Then loads global_indices = [global_offset, global_offset + 1, ..., global_offset + batch_size - 1]
# The permutation maps these to cache indices.
# This is DEVICE-INDEPENDENT because the global_offset and batch_size are the same.

# But let's also check: does the sharding affect which indices each device COMPUTES on?
# With 8 devices: device 0 gets examples [0,1,...,7], device 1 gets [8,...,15], etc.
# With 4 devices: device 0 gets examples [0,1,...,15], device 1 gets [16,...,31], etc.
# The FULL SET is the same [0..63], just distributed differently.

print("\n=== Device sharding of batch 0 ===")
for num_devices in [4, 8]:
    per_device = BATCH_SIZE // num_devices
    print(f"\n{num_devices} devices (per_device={per_device}):")
    for d in range(num_devices):
        start = d * per_device
        end = start + per_device
        device_seq = np.arange(start, end)
        device_cache = perm(device_seq)
        print(f"  Device {d}: seq[{start}:{end}] -> cache {device_cache[:4]}{'...' if per_device > 4 else ''}")

print("\n=== CONCLUSION ===")
print("Both 4-device and 8-device runs use the SAME 64 cache indices for batch 0.")
print(f"The permutation is deterministic from seed={SEED}.")
print("Device count only affects how the 64 examples are distributed across devices,")
print("NOT which 64 examples are selected.")
print("\nHowever: the per-device computation order differs:")
print("  8 devices: each processes 8 examples")
print("  4 devices: each processes 16 examples")
print("This affects the all-reduce reduction tree topology.")
