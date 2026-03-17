"""Patch weight_utils.py shard_put for Ray multi-host.

When sharding spec is None or a PartitionSpec (not NamedSharding),
convert to NamedSharding(mesh, spec) before passing to general_device_put.
Without this, the Ray multi-host path in general_device_put crashes
trying to call .addressable_devices_indices_map() on None.
"""

import os

BASE = "/workspace/tpu_inference/tpu_inference"
PATH = os.path.join(BASE, "models/jax/utils/weight_utils.py")

with open(PATH) as f:
    code = f.read()

# Fix 1: shard_put — handle None and PartitionSpec shardings
old_shard_put = '''    if isinstance(shardings, tuple):
        return general_device_put(x,
                                  NamedSharding(mesh, P(*shardings)),
                                  source_mesh=x_mesh)
    else:
        return general_device_put(x, shardings, source_mesh=x_mesh)'''

new_shard_put = '''    if isinstance(shardings, tuple):
        return general_device_put(x,
                                  NamedSharding(mesh, P(*shardings)),
                                  source_mesh=x_mesh)
    elif shardings is None:
        # No sharding spec — replicate across all devices
        return general_device_put(x,
                                  NamedSharding(mesh, P()),
                                  source_mesh=x_mesh)
    elif isinstance(shardings, P):
        # PartitionSpec needs to be wrapped in NamedSharding for Ray path
        return general_device_put(x,
                                  NamedSharding(mesh, shardings),
                                  source_mesh=x_mesh)
    else:
        return general_device_put(x, shardings, source_mesh=x_mesh)'''

if old_shard_put in code:
    code = code.replace(old_shard_put, new_shard_put)
    with open(PATH, "w") as f:
        f.write(code)
    print("PATCHED weight_utils.py: shard_put handles None/PartitionSpec shardings")
else:
    print("SKIP weight_utils.py: shard_put pattern not found (may already be patched)")

# Also check if P is imported
if "from jax.sharding import" in code and "PartitionSpec" in code:
    # P should already be available as PartitionSpec alias
    pass
