import ray
import torch_xla.distributed.xla_multiprocessing as xmp


def _mp_fn(rank):
    print(f"Hello from rank {rank}")


@ray.remote(
    num_cpus=10,
    resources={"TPU": 8, "TPU-v6e-8-head": 1},
    # runtime_env={"pip": "experiments/test_xla_infer_requirements.txt"},
)
def run_on_tpu():
    xmp.spawn(_mp_fn)


if __name__ == "__main__":
    future = run_on_tpu.remote()
    ray.get(future)
