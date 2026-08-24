"""Iris-side launcher for ragged_a2a_bench: joins the gang's JAX world, then runs main()."""

import pathlib
import sys

from iris.runtime import jax_init

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import ragged_a2a_bench

jax_init.initialize_jax()
ragged_a2a_bench.main()
