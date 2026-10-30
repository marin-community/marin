# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Subprocess entry point that imports one proposed applet backend."""

import importlib
import sys
from pathlib import Path


def main() -> None:
    if len(sys.argv) < 3:
        raise SystemExit("usage: python -m marina.applet_validator MODULE_ROOT MODULE[:FACTORY] ...")
    module_root = Path(sys.argv[1]).resolve()
    sys.path.insert(0, str(module_root.parent))
    for entrypoint in sys.argv[2:]:
        module_path, separator, factory_name = entrypoint.partition(":")
        module = importlib.import_module(f"{module_root.name}.{module_path}")
        target = factory_name if separator else "main"
        if not callable(getattr(module, target)):
            raise TypeError(f"{target} is not callable")
        migration = getattr(module, "migrate", None)
        if migration is not None and not callable(migration):
            raise TypeError("migrate is not callable")


if __name__ == "__main__":
    main()
