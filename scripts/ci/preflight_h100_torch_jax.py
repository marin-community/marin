# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Verify Torch and JAX can execute on the same assigned H100 worker."""

import os
import sys

import jax
import jax.numpy as jnp
import torch


def main() -> None:
    assert os.environ.get("JAX_PLATFORMS") == "cuda,cpu"
    assert torch.__version__.endswith("+cu128"), torch.__version__
    assert torch.cuda.is_available()
    assert torch.cuda.device_count() == 1
    torch_name = torch.cuda.get_device_name(0)
    assert "H100" in torch_name, torch_name
    torch_result = (torch.ones(4, device="cuda") + 2).sum().item()
    assert torch_result == 12

    jax_devices = jax.devices("cuda")
    assert len(jax_devices) == 1, jax_devices
    assert "H100" in jax_devices[0].device_kind, jax_devices[0]
    assert jax.default_backend() == "cuda"
    jax_result = jax.jit(lambda values: values + 2)(jnp.ones(4, dtype=jnp.float32))
    assert jax_result.device == jax_devices[0]
    assert float(jax_result.sum()) == 12

    print(
        f"preflight passed: interpreter={sys.executable} "
        f"torch={torch.__version__} ({torch_name}) "
        f"jax={jax.__version__} ({jax_devices[0]})",
        flush=True,
    )


if __name__ == "__main__":
    main()
