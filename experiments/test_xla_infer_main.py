import ray
from test_xla_infer import run_on_tpu

if __name__ == "__main__":
    future = run_on_tpu.remote()
    ray.get(future)
