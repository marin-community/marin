# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import os
from pathlib import Path

import draccus
import pytest


def _tpu_is_available() -> bool:
    """Whether a Cloud TPU is present, via its numbered VFIO IOMMU groups (/dev/vfio/0, ...).

    Match only a numbered group, not any /dev/vfio entry: the bare /dev/vfio/vfio control device is
    present wherever the vfio module is loaded (e.g. GitHub ubuntu-latest runners) with no TPU bound,
    so ``any(iterdir())`` false-positives there. Numbered groups are the real-TPU signal the iris
    worker and the GrugMoE e2e key off.
    """
    vfio = Path("/dev/vfio")
    return vfio.is_dir() and any(p.name.isdigit() for p in vfio.iterdir())


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
