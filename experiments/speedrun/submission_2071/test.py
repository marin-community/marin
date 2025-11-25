import ray
import os
import jax

ray.init(address="auto")

@ray.remote(num_gpus=1)
def check_env():
    return {
        "LD_LIBRARY_PATH": os.environ.get("LD_LIBRARY_PATH", "NOT_SET"),
        "JAX_PLATFORMS": os.environ.get("JAX_PLATFORMS", "NOT_SET"),
        "JAX_DEVICES": str(jax.devices())
    }

print(ray.get(check_env.remote()))