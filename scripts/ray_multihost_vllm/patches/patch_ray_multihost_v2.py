"""Comprehensive patch for Ray multi-host TPU vLLM.

Applies three fixes inside a running container:

1. tpu_worker.py: Add JAX isolation env vars for Ray multi-host.
   Without TPU_PROCESS_BOUNDS/TPU_CHIPS_PER_PROCESS_BOUNDS, JAX on a
   multi-host slice sees all global devices, causing XLA device_id
   mismatches in collective operations.

2. distributed/utils.py: Add d.coords fallback for topology ordering.
   Ray workers may get devices without .coords attribute; fall back to
   process_index ordering.

3. utils.py hbm_usage_bytes: Safety fallback if no device returns
   valid memory stats.

Usage: python patch_ray_multihost_v2.py
"""

import os
import sys

BASE = "/workspace/tpu_inference/tpu_inference"


def patch_tpu_worker():
    """Add Ray multi-host JAX isolation env vars to init_device()."""
    path = os.path.join(BASE, "worker/tpu_worker.py")
    with open(path) as f:
        code = f.read()

    # Find the guard that skips env var setup for Ray
    old_guard = 'if multihost_backend != "ray" and self.parallel_config.pipeline_parallel_size > 1:'

    if old_guard not in code:
        print("SKIP tpu_worker.py: guard not found (may already be patched)")
        return False

    # Replace with: Ray gets isolation vars, non-Ray gets existing PP setup
    new_code = '''if multihost_backend == "ray":
            # Ray multi-host: isolate each worker to single-host JAX mode.
            # Without TPU_PROCESS_BOUNDS etc., JAX sees the full multi-host
            # topology via tpu-runtime, causing XLA device_id mismatches
            # in collective operations (e.g., "Unexpected device_id 4 in
            # replica group (0,1,2,3)").
            from tpu_inference import tpu_info
            total_devices = self.vllm_config.sharding_config.total_devices
            cores_per_chip = tpu_info.get_num_cores_per_chip()
            chips_needed = math.ceil(total_devices / cores_per_chip)
            from tpu_inference import tpu_info
            num_local_chips = tpu_info.get_num_chips()
            os.environ["TPU_PROCESS_BOUNDS"] = "1,1,1"
            os.environ["TPU_CHIPS_PER_PROCESS_BOUNDS"] = f"1,{num_local_chips},1"
            os.environ["TPU_VISIBLE_CHIPS"] = ",".join(
                str(i) for i in range(num_local_chips))
            os.environ["CLOUD_TPU_TASK_ID"] = "0"
            logger.info(
                f"Ray multi-host JAX isolation: "
                f"TPU_PROCESS_BOUNDS=1,1,1 "
                f"TPU_CHIPS_PER_PROCESS_BOUNDS=1,{chips_needed},1 "
                f"TPU_VISIBLE_CHIPS={','.join(str(i) for i in range(chips_needed))} "
                f"CLOUD_TPU_TASK_ID=0")
        elif self.parallel_config.pipeline_parallel_size > 1:'''

    code = code.replace(old_guard, new_code)

    with open(path, "w") as f:
        f.write(code)
    print("PATCHED tpu_worker.py: added Ray multi-host JAX isolation")
    return True


def patch_coords_fallback():
    """Add d.coords fallback in get_device_topology_order_id()."""
    path = os.path.join(BASE, "distributed/utils.py")
    with open(path) as f:
        code = f.read()

    # Find the error-only check for missing coords
    old_check = '''if not all(hasattr(d, "coords") for d in local_devices):
        logger.error(
            f"Expect TPU device but got {[type(d) for d in local_devices]}")'''

    if old_check not in code:
        # Try alternate pattern (single line error)
        old_check = '''if not all(hasattr(d, "coords") for d in local_devices):
        logger.error('''
        if old_check not in code:
            print("SKIP distributed/utils.py: coords check not found (may already be patched)")
            return False

    new_check = '''if not all(hasattr(d, "coords") for d in local_devices):
        logger.warning(
            f"Devices lack .coords, falling back to process_index ordering. "
            f"Types: {[type(d).__name__ for d in local_devices]}")
        if hasattr(local_devices[0], "process_index"):
            return local_devices[0].process_index
        return 0'''

    # Replace just the check block, keeping the rest
    # Find the full block to replace (up to the next non-indented line)
    lines = code.split('\n')
    new_lines = []
    i = 0
    replaced = False
    while i < len(lines):
        line = lines[i]
        if 'if not all(hasattr(d, "coords") for d in local_devices):' in line and not replaced:
            indent = line[:len(line) - len(line.lstrip())]
            new_lines.append(f'{indent}if not all(hasattr(d, "coords") for d in local_devices):')
            new_lines.append(f'{indent}    logger.warning(')
            new_lines.append(f'{indent}        f"Devices lack .coords, falling back to process_index. "')
            new_lines.append(f'{indent}        f"Types: {{[type(d).__name__ for d in local_devices]}}")')
            new_lines.append(f'{indent}    if hasattr(local_devices[0], "process_index"):')
            new_lines.append(f'{indent}        return local_devices[0].process_index')
            new_lines.append(f'{indent}    return 0')
            # Skip old error lines
            i += 1
            while i < len(lines) and ('logger.error' in lines[i] or 'Expect TPU device' in lines[i]):
                i += 1
            replaced = True
            continue
        new_lines.append(line)
        i += 1

    if replaced:
        with open(path, "w") as f:
            f.write('\n'.join(new_lines))
        print("PATCHED distributed/utils.py: added .coords fallback")
        return True
    else:
        print("SKIP distributed/utils.py: could not find pattern to replace")
        return False


def patch_hbm_usage():
    """Add safety fallback to hbm_usage_bytes for Ray."""
    path = os.path.join(BASE, "utils.py")
    with open(path) as f:
        code = f.read()

    # Check if the Ray branch already has adequate handling
    # Add a fallback after the Ray device loop in case no device worked
    old_pattern = '''    multihost_backend = envs.TPU_MULTIHOST_BACKEND
    if multihost_backend == "ray":
        # MemoryStats is only supported for addressable PjRt devices.
        # Assume all the devices have similar memory usage for now.
        # TODO(ranlihao): find a proper way to get the memory usage of each device.
        for device in devices:
            try:
                hbm_used = device.memory_stats()["bytes_in_use"]
                hbm_limit = device.memory_stats()["bytes_limit"]
                logger.info(
                    "Get memory stats for device %s. Assuming all devices have the same usage.",
                    device)
                usage.extend([(hbm_used, hbm_limit)] * len(devices))
                break
            except Exception as e:
                logger.warning(
                    "Failed to get memory stats for device %s: %s. ", device,
                    e)'''

    if old_pattern not in code:
        print("SKIP utils.py hbm: Ray branch pattern not found (may already be patched)")
        return False

    new_pattern = '''    multihost_backend = envs.TPU_MULTIHOST_BACKEND
    if multihost_backend == "ray":
        # MemoryStats is only supported for addressable PjRt devices.
        # Filter to locally addressable devices first.
        local_ids = {d.id for d in jax.local_devices()}
        addressable = [d for d in devices if d.id in local_ids]
        if not addressable:
            addressable = devices  # fallback to all
        for device in addressable:
            try:
                hbm_used = device.memory_stats()["bytes_in_use"]
                hbm_limit = device.memory_stats()["bytes_limit"]
                logger.info(
                    "Get memory stats for device %s. Assuming all devices have the same usage.",
                    device)
                usage.extend([(hbm_used, hbm_limit)] * len(devices))
                break
            except Exception as e:
                logger.warning(
                    "Failed to get memory stats for device %s: %s. ", device,
                    e)
        # Fallback: if no device returned stats, use device HBM limit
        if not usage:
            try:
                from tpu_inference.tpu_info import get_device_hbm_limit
                limit = get_device_hbm_limit()
                logger.warning(
                    "No device returned memory_stats. Using HBM limit fallback: %d bytes", limit)
                usage.extend([(0, limit)] * len(devices))
            except Exception as fallback_err:
                logger.error("HBM fallback also failed: %s", fallback_err)'''

    code = code.replace(old_pattern, new_pattern)
    with open(path, "w") as f:
        f.write(code)
    print("PATCHED utils.py: improved hbm_usage_bytes for Ray")
    return True


if __name__ == "__main__":
    results = []
    results.append(("tpu_worker.py (JAX isolation)", patch_tpu_worker()))
    results.append(("distributed/utils.py (coords fallback)", patch_coords_fallback()))
    results.append(("utils.py (hbm fallback)", patch_hbm_usage()))

    print("\n--- Patch Summary ---")
    for name, ok in results:
        status = "OK" if ok else "SKIPPED"
        print(f"  {status}: {name}")

    if all(r[1] for r in results):
        print("\nAll patches applied successfully.")
    else:
        print("\nSome patches were skipped (may already be applied).")
