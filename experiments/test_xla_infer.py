import ray


def _mp_fn(rank):
    print(f"Hello from rank {rank}")


@ray.remote(
    num_cpus=10,
    resources={"TPU": 4, "TPU-v4-8-head": 1},
    runtime_env={"pip": "experiments/test_xla_infer_requirements.txt"},
)
def run_on_tpu():
    import torch_xla.distributed.xla_multiprocessing as xmp

    xmp.spawn(_mp_fn)


future = run_on_tpu.remote()
ray.get(future)
