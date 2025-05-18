


# SPDX-License-Identifier: Apache-2.0
# Adapted from MosaicML Composer: https://github.com/mosaicml/composer/blob/dev/composer/callbacks/speed_monitor.py

# Base FLOPS for each device type (using mixed precision values where available)
device_flops_map = {
    # GPUs (using mixed precision values where available)
    "H100": 1.513e15 / 2,  # amp_fp16/bf16; pcie
    "A100": 312e12,  # amp_fp16/bf16
    "A10G": 70e12,  # bf16: https://www.baseten.co/blog/nvidia-a10-vs-a10g-for-ml-model-inference/
    "V100": 125e12,  # amp_fp16
    "T4": 65e12,  # amp_fp16
    "A6000": 309.7e12 / 2,  # bf16/fp16
    
    # TPU base FLOPS per chip,
    "v4": 275e12,  # bf16, https://cloud.google.com/tpu/docs/v4
    "v5litepod": 197e12,  # bf16, https://cloud.google.com/tpu/docs/v5e
    
    "v5": 459e12,  # bf16, https://cloud.google.com/tpu/docs/v5
    "v6e": 918e12,  # bf16, https://cloud.google.com/tpu/docs/v6e
}


def get_tpu_type_and_chips(tpu_name: str) -> tuple[str, int]:
    """Extract TPU type and number of chips from TPU name.
    
    Args:
        tpu_name: TPU name like 'v4-128' or 'v5litepod-64'
    
    Returns:
        Tuple of (tpu_type, num_chips)
        
    Raises:
        ValueError: If tpu_name is not in expected format
    """
    try:
        # split by first - since count is always at the end
        tpu_type, count = tpu_name.lower().split("-", maxsplit=1)
        count = int(count)
        
        # Handle aliases
        tpu_type = device_flops_map.get(tpu_type, tpu_type)
        
        # Validate TPU type
        if tpu_type not in device_flops_map:
            raise ValueError(f"Unknown TPU type: {tpu_type}. Available types: {sorted(k for k in device_flops_map.keys() if k.startswith('v'))}")
            
        # Map size to actual chip count
        # For v4: size/2 gives chip count up to 256 chips
        if tpu_type == "v4":
            num_chips = count // 2
        else:
            num_chips = count
                
        return tpu_type, num_chips
        
    except ValueError as e:
        raise ValueError(f"Invalid TPU name format: {tpu_name}. Expected format: <type>-<size> (e.g. v4-128)") from e


def get_tpu_flops(tpu_name: str) -> float:
    """Get total FLOPS for a TPU configuration.
    
    Args:
        tpu_name: TPU name like 'v4-128' or 'v5litepod-64'
    
    Returns:
        Total FLOPS for the TPU configuration
    """
    tpu_type, num_chips = get_tpu_type_and_chips(tpu_name)
    return device_flops_map[tpu_type] * num_chips