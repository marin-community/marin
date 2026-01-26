"""GrugFuzz: Testing utilities for porting HuggingFace models to Grug JAX.

This library provides utilities for incrementally testing model ports:
- Compare arrays and report detailed diffs
- Generate random test inputs
- Convert between PyTorch and JAX
- Run HuggingFace modules and get JAX outputs
- Diagnose comparison failures

Example usage:
    from grugfuzz import compare, run_hf, random_tokens

    # Generate test input
    tokens = random_tokens(vocab_size=32000, batch=2, seq=32)

    # Run HF module and get JAX output
    hf_out = run_hf(hf_model.embed_tokens, tokens)

    # Compare with Grug implementation
    grug_out = grug_params.token_embed[tokens]
    result = compare(hf_out, grug_out, name="embedding")
    print(result)  # ✓ embedding: PASS (max diff: 1.2e-7)
"""

# Set up JAX mesh at import time - required for Grug's sharding operations
import jax
from jax.sharding import Mesh
from jax._src import mesh as mesh_lib
import numpy as np

_devices = jax.devices()[:1]
# Use Explicit axis types so PartitionSpec works correctly
_mesh = Mesh(
    np.array(_devices).reshape(1, 1),
    axis_names=("data", "model"),
    axis_types=(mesh_lib.AxisType.Explicit, mesh_lib.AxisType.Explicit),
)
jax.set_mesh(_mesh)

from .bridges import (
    hf_state_dict_to_jax,
    jax_state_dict_to_torch,
    jax_to_torch,
    torch_to_jax,
)
from .compare import ComparisonResult, compare
from .diagnostics import compare_structures, diagnose_diff
from .inputs import (
    random_attention_mask,
    random_hidden,
    random_kv_cache,
    random_qkv,
    random_tokens,
)
from .fuzz import Choice, FloatRange, FuzzSpace, IntRange, sample_fuzz_cases
from .mesh import (
    create_mesh,
    ensure_devices,
    sharded_ones,
    sharded_randn,
    sharded_zeros,
    with_mesh,
)
from .runners import SuiteResult, run_comparison_suite, run_hf

__all__ = [
    # Compare
    "compare",
    "ComparisonResult",
    # Bridges
    "torch_to_jax",
    "jax_to_torch",
    "hf_state_dict_to_jax",
    "jax_state_dict_to_torch",
    # Inputs
    "random_tokens",
    "random_hidden",
    "random_qkv",
    "random_attention_mask",
    "random_kv_cache",
    # Fuzzing
    "Choice",
    "FloatRange",
    "FuzzSpace",
    "IntRange",
    "sample_fuzz_cases",
    # Mesh utilities
    "ensure_devices",
    "create_mesh",
    "with_mesh",
    "sharded_zeros",
    "sharded_ones",
    "sharded_randn",
    # Runners
    "run_hf",
    "run_comparison_suite",
    "SuiteResult",
    # Diagnostics
    "diagnose_diff",
    "compare_structures",
]
