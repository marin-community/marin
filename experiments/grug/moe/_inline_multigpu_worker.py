# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Worker entry for GRUG_RUN_INLINE multi-process training.

The launcher process builds the ``GrugRunConfig`` once (running the marin executor a single
time, so no per-step lock contention), pickles it, and spawns ``iris.runtime.multigpu`` over
this module. Each supervised worker (one GPU) reloads the config here and runs the training
loop; ``levanter.distributed`` reads the ``IRIS_MULTIGPU_*`` rank env and joins the JAX mesh.
"""

import os
import pickle

from experiments.grug.moe.train import _run_grug_local


def main() -> None:
    config_path = os.environ["GRUG_INLINE_CONFIG_PATH"]
    with open(config_path, "rb") as handle:
        config = pickle.load(handle)
    _run_grug_local(config)


if __name__ == "__main__":
    main()
