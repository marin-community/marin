# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: F821

import os

import lit.formats

config.name = "Shuttle MLIR"
config.test_format = lit.formats.ShTest(execute_external=True)
config.suffixes = [".mlir"]
config.excludes = ["Inputs"]
config.test_source_root = os.path.dirname(__file__)
config.test_exec_root = os.path.dirname(__file__)
