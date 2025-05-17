# Mapping of accelerator types to their peak FLOPs/s.
# Values are from DEVICE_AVAILABLE_FLOPS, using bf16 where available, falling back to fp16.
# For GPUs with multiple variants (e.g. SXM vs PCIe), PCIe variants are used.
device_flops_map = {
    # NVIDIA GPUs
    "V100": 112e12,  # V100 PCIe fp16
    "P100": 65e12,   # T4 fp16 (used as proxy since P100 not in reference)
    "T4": 65e12,    # T4 fp16
    "P4": 65e12,    # Using T4 as proxy
    "K80": 50e12,   # Estimate
    "A10G": 125e12, # A10G bf16
    "A100": 312e12, # A100 bf16
    "H100": 756e12, # H100 PCIe bf16 (1.513e15 / 2)
    "H200": 800e12, # H200 bf16 (estimate)
    "L4": 90e12,    # L4 bf16
    "L40S": 400e12, # L40S bf16
    "H20": 300e12,  # Estimate
    
    # Google TPUs
    "TPU-V2": 123e12 / 2,  # Using V3 as proxy since V2 not in reference
    "TPU-V3": 123e12 / 2,  # TPU V3 bf16 per core
    "TPU-V4": 275e12,      # TPU V4 bf16
    "TPU-V5P": 459e12,     # TPU V5P bf16
    "TPU-V5LITEPOD": 197e12,  # TPU V5 Lite bf16
    "TPU-V6E": 918e12,     # TPU V6 Lite bf16
    
    # AWS Inferentia
    "aws-neuron-core": 190e12 / 2,  # TRN1 bf16 per device
    
    # Other devices use a reasonable fallback or 0 if unknown
    "Intel-GPU-Max-1550": 0,
    "Intel-GPU-Max-1100": 0,
    "Intel-GAUDI": 0,
    "AMD-Instinct-MI100": 0,
    "AMD-Instinct-MI250X": 0,
    "AMD-Instinct-MI250X-MI250": 0,
    "AMD-Instinct-MI210": 0,
    "AMD-Instinct-MI300X-OAM": 0,
    "AMD-Radeon-R9-200-HD-7900": 0,
    "AMD-Radeon-HD-7900": 0,
    "Ascend910B": 0,
    "Ascend910B4": 0,
}