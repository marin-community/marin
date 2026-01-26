"""Framework conversion utilities between PyTorch and JAX."""

from typing import Any

import jax.numpy as jnp
import numpy as np

try:
    import torch
except ImportError:
    torch = None  # type: ignore


def torch_to_jax(tensor: "torch.Tensor") -> jnp.ndarray:
    """Convert PyTorch tensor to JAX array.

    Args:
        tensor: PyTorch tensor (any device)

    Returns:
        JAX array with float32 for floats, int32 for ints
    """
    if torch is None:
        raise ImportError("torch is required for torch_to_jax")
    # Normalize dtypes before moving to numpy to avoid incompatibilities (e.g. bf16).
    if tensor.is_floating_point():
        tensor = tensor.to(dtype=torch.float32)
        return jnp.array(tensor.detach().cpu().numpy(), dtype=jnp.float32)
    if tensor.dtype == torch.bool:
        return jnp.array(tensor.detach().cpu().numpy(), dtype=jnp.bool_)
    tensor = tensor.to(dtype=torch.int32)
    return jnp.array(tensor.detach().cpu().numpy(), dtype=jnp.int32)


def jax_to_torch(array: jnp.ndarray, device: str = "cpu") -> "torch.Tensor":
    """Convert JAX array to PyTorch tensor.

    Args:
        array: JAX array
        device: Target PyTorch device ("cpu", "cuda", etc.)

    Returns:
        PyTorch tensor with float32 for floats, int32 for ints
    """
    if torch is None:
        raise ImportError("torch is required for jax_to_torch")
    # Convert to numpy then normalize dtype (copy to avoid non-writable array warning)
    np_array = np.array(array)
    if np_array.dtype.kind in ("i", "u"):
        np_array = np_array.astype(np.int32, copy=False)
        return torch.from_numpy(np_array.copy()).to(device)
    if np_array.dtype.kind == "b":
        np_array = np_array.astype(np.bool_, copy=False)
        return torch.from_numpy(np_array.copy()).to(device)
    np_array = np_array.astype(np.float32, copy=False)
    return torch.from_numpy(np_array.copy()).to(device)


def hf_state_dict_to_jax(state_dict: dict[str, "torch.Tensor"]) -> dict[str, jnp.ndarray]:
    """Convert entire HuggingFace state dict to JAX arrays.

    Args:
        state_dict: PyTorch state dict from HF model

    Returns:
        Dictionary with same keys but JAX array values
    """
    return {k: torch_to_jax(v) for k, v in state_dict.items()}


def jax_state_dict_to_torch(
    state_dict: dict[str, jnp.ndarray], device: str = "cpu"
) -> dict[str, "torch.Tensor"]:
    """Convert JAX state dict to PyTorch state dict.

    Args:
        state_dict: JAX state dict
        device: Target PyTorch device

    Returns:
        Dictionary with same keys but PyTorch tensor values
    """
    return {k: jax_to_torch(v, device) for k, v in state_dict.items()}
