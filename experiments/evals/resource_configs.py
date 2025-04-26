from dataclasses import dataclass


@dataclass
class ResourceConfig:
    num_tpu: int
    tpu_type: str
    strategy: str


SINGLE_TPU_V4_8 = ResourceConfig(num_tpu=1, tpu_type="TPU-v4-8", strategy="STRICT_PACK")  # us-central2
SINGLE_TPU_V4_16 = ResourceConfig(num_tpu=1, tpu_type="TPU-v4-16", strategy="STRICT_PACK")  # us-central2
SINGLE_TPU_V6E_8 = ResourceConfig(num_tpu=1, tpu_type="TPU-v6e-8", strategy="STRICT_PACK")  # us-east5
TPU_V6E_8_STRICT_PACK = ResourceConfig(num_tpu=8, tpu_type="TPU-v6e-8", strategy="STRICT_PACK")  # us-east5
TPU_V4_16_STRICT_PACK = ResourceConfig(num_tpu=8, tpu_type="TPU-v4-16", strategy="STRICT_PACK")  # us-central2
