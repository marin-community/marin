# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import glob
import os
from collections.abc import Callable

import draccus
import pytest
from marin.utils import rebase_file_path


def parameterize_with_configs(pattern: str, config_path: str | None = None) -> Callable:
    """
    A decorator to parameterize a test function with configuration files.

    Args:
        pattern (str): A glob pattern to match configuration files.
        config_path (Optional[str]): The base path to look for config files.
                                    If None, uses "../config" relative to the test file.

    Returns:
        Callable: A pytest.mark.parametrize decorator that provides config files to the test function.

    """
    test_path = os.path.dirname(os.path.abspath(__file__))
    if config_path is None:
        config_path = os.path.join(test_path, "..", "config")
    configs = glob.glob(os.path.join(config_path, "**", pattern), recursive=True)
    return pytest.mark.parametrize("config_file", configs, ids=lambda x: f"{os.path.basename(x)}")


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


# rebase_file_path is pure string manipulation (os.path.relpath + string ops),
# so these tests use string paths directly rather than materialising files via tmp_path.
_REBASE_BASE_IN = "/in"
_REBASE_BASE_OUT = "/out"


@pytest.mark.parametrize(
    ("rel_path", "kwargs", "expected_rel"),
    [
        pytest.param(
            "nested/sample.parquet",
            {"new_extension": ".parquet", "old_extension": ".parquet"},
            "nested/sample.parquet",
            id="matching_extension",
        ),
        pytest.param(
            "sample.jsonl.gz",
            {"new_extension": ".parquet", "old_extension": ".jsonl.gz"},
            "sample.parquet",
            id="compound_old_extension_replaced",
        ),
        pytest.param(
            "noext",
            {"new_extension": ".txt"},
            "noext.txt",
            id="no_dot_appends_new_extension",
        ),
    ],
)
def test_rebase_file_path(rel_path, kwargs, expected_rel):
    file_path = os.path.join(_REBASE_BASE_IN, rel_path)
    rebased = rebase_file_path(_REBASE_BASE_IN, file_path, _REBASE_BASE_OUT, **kwargs)
    assert rebased == os.path.join(_REBASE_BASE_OUT, expected_rel)


@pytest.mark.parametrize(
    ("rel_path", "kwargs", "match"),
    [
        pytest.param(
            "sample.parquet",
            {"old_extension": ".parquet"},
            "old_extension requires new_extension",
            id="old_without_new",
        ),
        pytest.param(
            "sample.jsonl",
            {"new_extension": ".parquet", "old_extension": ".jsonl.gz"},
            "does not end with old_extension",
            id="mismatched_old_extension",
        ),
    ],
)
def test_rebase_file_path_raises(rel_path, kwargs, match):
    file_path = os.path.join(_REBASE_BASE_IN, rel_path)
    with pytest.raises(ValueError, match=match):
        rebase_file_path(_REBASE_BASE_IN, file_path, _REBASE_BASE_OUT, **kwargs)
