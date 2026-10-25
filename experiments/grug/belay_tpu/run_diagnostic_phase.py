# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Leave Iris job completion to the parent after a diagnostic subprocess exits."""

import atexit
import importlib
import sys
from unittest.mock import patch

from levanter.distributed import DistributedConfig, _finalize_iris_jax_after_clean_exit, _unregister_iris_exit_handler


def main():
    module_name, *arguments = sys.argv[1:]
    module = importlib.import_module(module_name)
    initialize_distributed = DistributedConfig.initialize

    def initialize(config):
        initialize_distributed(config)
        _unregister_iris_exit_handler()
        atexit.register(_finalize_iris_jax_after_clean_exit)

    sys.argv = [module_name, *arguments]
    with patch.object(DistributedConfig, "initialize", initialize):
        module.main()


if __name__ == "__main__":
    main()
