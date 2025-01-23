# import ray
# from ray.runtime_env import RuntimeEnv


# runtime_env = RuntimeEnv(
#     container={
#         "image": "vllm/vllm-tpu:2fc6944c5e69d5d0ce15d09a855452c795d75c3c"
#     }
# )
# # Ran on the worker node (works)
# # remove the resources and it fails
# @ray.remote(resources = {"TPU-v4-8-head": 1, "TPU": 4}, runtime_env=runtime_env)
# def foo():
#     import torch_xla.core.xla_model as xm
#     from vllm import LLM

# if __name__ == "__main__":
#     # Ran on the head node (works most of the time)
#     import torch_xla.core.xla_model as xm
#     bar = ray.get(foo.remote())

import os

import ray


@ray.remote(resources={"TPU-v4-8-head": 1, "TPU": 4})
def foo():
    print("TPU_VISIBLE_CHIPS: {}".format(os.environ["TPU_VISIBLE_CHIPS"]))


if __name__ == "__main__":
    bar = ray.get(foo.remote())
