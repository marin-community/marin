# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import os
from pathlib import Path

import draccus
import pytest


def _tpu_is_available() -> bool:
    """A Cloud TPU exposes its chips through /dev/vfio (the signal the GrugMoE e2e checks too)."""
    vfio = Path("/dev/vfio")
    return vfio.is_dir() and any(vfio.iterdir())


# Skip a test that needs a real TPU when none is present, so pointing pytest at the file on a
# TPU host runs it while a CPU run reports a clean skip.
skip_if_no_tpu = pytest.mark.skipif(not _tpu_is_available(), reason="no TPU available")


def check_load_config(config_class: type, config_file: str) -> None:
    """
    Attempt to load and parse a configuration file using a specified config class.

    Args:
        config_class (Type): The configuration class to use for parsing.
        config_file (str): Path to the configuration file to be parsed.

    Raises:
        Exception: If the configuration file fails to parse.
    """
    try:
        draccus.parse(config_class, config_file, args=[])
    except Exception as e:
        raise Exception(f"failed to parse {config_file}") from e


def skip_if_module_missing(module: str):
    def try_import_module(module):
        try:
            __import__(module)
        except ImportError:
            return False
        else:
            return True

    return pytest.mark.skipif(not try_import_module(module), reason=f"{module} not installed")


def skip_in_ci(fn_or_msg):
    if isinstance(fn_or_msg, str):

        def decorator(fn):
            return pytest.mark.skipif("CI" in os.environ, reason=fn_or_msg)(fn)

        return decorator

    return pytest.mark.skipif("CI" in os.environ, reason="skipped in CI")(fn_or_msg)
