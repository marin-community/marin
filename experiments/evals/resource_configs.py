from marin.resources import TpuPodConfig

SINGLE_TPU_V4_8 = TpuPodConfig(
    tpu_type="v4-8", chip_count=1, include_tpu_in_ray_resources=True, packing_strategy="STRICT_PACK"
)
SINGLE_TPU_V4_16 = TpuPodConfig(
    tpu_type="v4-16", chip_count=1, include_tpu_in_ray_resources=True, packing_strategy="STRICT_PACK"
)
SINGLE_TPU_V6E_8 = TpuPodConfig(
    tpu_type="v6e-8", chip_count=1, include_tpu_in_ray_resources=True, packing_strategy="STRICT_PACK"
)
TPU_V6E_8_STRICT_PACK = TpuPodConfig(
    tpu_type="v6e-8", chip_count=8, include_tpu_in_ray_resources=True, packing_strategy="STRICT_PACK"
)
TPU_V4_16_STRICT_PACK = TpuPodConfig(
    tpu_type="v4-16", chip_count=8, include_tpu_in_ray_resources=True, packing_strategy="PACK"
)
