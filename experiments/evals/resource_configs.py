from dataclasses import dataclass


@dataclass
class ResourceConfig:
    num_tpu: int
    tpu_type: str


SINGLE_TPU_V4_8 = ResourceConfig(num_tpu=1, tpu_type="TPU-v4-8")
SINGLE_TPU_V6E_8 = ResourceConfig(num_tpu=1, tpu_type="TPU-v6e-8")
