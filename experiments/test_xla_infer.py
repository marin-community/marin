import ray
import torch_xla.distributed.xla_multiprocessing as xmp


def _mp_fn(rank: int):
    print(f"Hello, world! from rank {rank}")


@ray.remote(
    num_cpus=8,
    resources={"TPU": 8, "TPU-v6e-8-head": 1},
    runtime_env={"env_vars": {"TPU_NUM_DEVICES": "8", "PJRT_DEVICE": "TPU"}},
)
def test_xla_infer():

    xmp.spawn(_mp_fn, args=())


if __name__ == "__main__":
    ray.get(test_xla_infer.remote())
