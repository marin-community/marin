# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Levanter: a JAX library for training foundation models.

Import from the defining submodule (``levanter.trainer``, ``levanter.models``,
``levanter.optim``, ``levanter.data``, ``levanter.tracker``, ...) rather than the package. This
module deliberately imports nothing: eagerly importing the submodules here made ``import
levanter`` pull in the whole library, so a change to any one module selected nearly every test.
"""

__version__ = "1.2"
