# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Subprocess entry point that imports one proposed applet backend."""

import importlib
import sys
from pathlib import Path


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("usage: python -m marina.applet_validator MODULE_ROOT MODULE:FACTORY")
    module_root = Path(sys.argv[1]).resolve()
    module_path, factory_name = sys.argv[2].split(":", 1)
    sys.path.insert(0, str(module_root.parent))
    module = importlib.import_module(f"{module_root.name}.{module_path}")
    factory = getattr(module, factory_name)
    if not callable(factory):
        raise TypeError(f"{factory_name} is not callable")
    migration = getattr(module, "migrate", None)
    if migration is not None and not callable(migration):
        raise TypeError("migrate is not callable")


if __name__ == "__main__":
    main()
