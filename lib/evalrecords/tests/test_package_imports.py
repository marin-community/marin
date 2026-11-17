# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import importlib
import pkgutil

import evalrecords


def test_every_module_imports_with_only_the_declared_dependencies():
    """This scope installs the package without ``marin`` or a training library, so a module
    that imports one of them fails here."""
    for module in pkgutil.iter_modules(evalrecords.__path__, prefix="evalrecords."):
        importlib.import_module(module.name)
