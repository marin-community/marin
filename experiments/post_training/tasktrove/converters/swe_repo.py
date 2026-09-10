# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Helpers shared by the SWE-bench-shaped converters (``swe_patched``, ``swe_trusted_paths``)."""

import json

PLUGIN = "pytest-json-report"


def test_ids(value: object) -> list[str]:
    """``FAIL_TO_PASS``/``PASS_TO_PASS`` as a list of node ids; the field is sometimes a JSON-encoded string."""
    if isinstance(value, str):
        decoded: object = json.loads(value) if value.strip() else []
    else:
        decoded = value if value is not None else []
    if not isinstance(decoded, list):
        raise ValueError(f"expected a list of test ids, got {type(decoded).__name__}")
    return [str(v) for v in decoded]


def ensure_pytest_json_report(dockerfile: str, conda_lines: tuple[str, ...] = ()) -> str:
    """Add the ``pytest-json-report`` plugin the ``pytest`` mode needs into the repo's own Python.

    Installed at image build time because grading runs with no network. An existing
    ``RUN ... pip install ... pytest`` line gets the plugin appended, so it lands in the same
    interpreter; otherwise a conda repo installs it inside its activated env and a plain repo
    through ``pip`` or ``pip3``.
    """
    if PLUGIN in dockerfile:
        return dockerfile
    lines = dockerfile.splitlines()
    for index, line in enumerate(lines):
        tokens = line.split()
        if tokens[:1] == ["RUN"] and "pip" in tokens and "install" in tokens and "pytest" in tokens:
            lines[index] = line + f" {PLUGIN}"
            return "\n".join(lines) + "\n"
    if conda_lines:
        activate = " && ".join(conda_lines)
        install = f'RUN bash -lc "{activate} && pip install --no-cache-dir {PLUGIN}"\n'
    else:
        install = f"RUN (pip install --no-cache-dir {PLUGIN} || pip3 install --no-cache-dir {PLUGIN})\n"
    return dockerfile.rstrip("\n") + "\n" + install
