NVIDIA_TESLA_V100 = "V100"
NVIDIA_TESLA_P100 = "P100"
NVIDIA_TESLA_T4 = "T4"
NVIDIA_TESLA_P4 = "P4"
NVIDIA_TESLA_K80 = "K80"
NVIDIA_TESLA_A10G = "A10G"
NVIDIA_L4 = "L4"
NVIDIA_L40S = "L40S"
NVIDIA_A100 = "A100"
NVIDIA_H100 = "H100"
NVIDIA_H200 = "H200"
NVIDIA_H20 = "H20"
INTEL_MAX_1550 = "Intel-GPU-Max-1550"
INTEL_MAX_1100 = "Intel-GPU-Max-1100"
INTEL_GAUDI = "Intel-GAUDI"
AMD_INSTINCT_MI100 = "AMD-Instinct-MI100"
AMD_INSTINCT_MI250x = "AMD-Instinct-MI250X"
AMD_INSTINCT_MI250 = "AMD-Instinct-MI250X-MI250"
AMD_INSTINCT_MI210 = "AMD-Instinct-MI210"
AMD_INSTINCT_MI300x = "AMD-Instinct-MI300X-OAM"
AMD_RADEON_R9_200_HD_7900 = "AMD-Radeon-R9-200-HD-7900"
AMD_RADEON_HD_7900 = "AMD-Radeon-HD-7900"
AWS_NEURON_CORE = "aws-neuron-core"
GOOGLE_TPU_V2 = "TPU-V2"
GOOGLE_TPU_V3 = "TPU-V3"
GOOGLE_TPU_V4 = "TPU-V4"
GOOGLE_TPU_V5P = "TPU-V5P"
GOOGLE_TPU_V5LITEPOD = "TPU-V5LITEPOD"
GOOGLE_TPU_V6E = "TPU-V6E"
HUAWEI_NPU_910B = "Ascend910B"
HUAWEI_NPU_910B4 = "Ascend910B4"

# Use these instead of NVIDIA_A100 if you need a specific accelerator size. Note that
# these labels are not auto-added to nodes, you'll have to add them manually in
# addition to the default A100 label if needed.
NVIDIA_A100_40G = "A100-40G"
NVIDIA_A100_80G = "A100-80G"


# Mapping of accelerator types to their peak FLOPs/s.
# Values are from DEVICE_AVAILABLE_FLOPS, using bf16 where available, falling back to fp16.
# For GPUs with multiple variants (e.g. SXM vs PCIe), PCIe variants are used.
device_flops_map = {
    # NVIDIA GPUs
    "V100": 112e12,  # V100 PCIe fp16
    "P100": 65e12,   # T4 fp16 (used as proxy since P100 not in reference)
    "T4": 65e12,    # T4 fp16
    "A10": 125e12,  # A10 bf16
    "A100": 312e12, # A100 bf16
    "H100": 1.513e15 / 2,  # H100 PCIe bf16
    "H200": 312e12, # Using A100 as proxy since H200 not in reference
    "A6000": 309.7e12 / 2,  # A6000 bf16
    
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