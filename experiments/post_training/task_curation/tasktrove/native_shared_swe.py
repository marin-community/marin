# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Language boundaries for the SWE-rebench source command and parser contract."""

import hashlib
import json
import re
from pathlib import Path

from taskcompendium.integrity import NativeSharedRunner
from taskcompendium.native_integrity import FAMILIES, LANGUAGES
from taskcompendium.runtime import shared_pytest


def native_runner(files: dict[str, bytes]) -> NativeSharedRunner:
    """Bind a source-declared language and exact command to a supported family."""
    config = json.loads(files["tests/config.json"])
    family = LANGUAGES[config["language"]]
    profile = FAMILIES[family]
    install = config["install_config"]
    if isinstance(install, str):
        install = json.loads(install)
    command = install["test_cmd"]
    # Require the exact command boundary used by the source's normalized script.
    source = files["tests/legacy_test.sh"].decode()
    if f"{command} > /logs/verifier/test_output.log 2>&1" not in source:
        raise ValueError("Source test command differs from its normalized runner")
    commit = config["base_commit"]
    if re.fullmatch(r"[0-9a-f]{7,40}", commit) is None:
        raise ValueError("Native SHARED requires a pinned source commit")
    return NativeSharedRunner(
        family=family,
        command=command,
        toolchain_commands=profile.tools,
        config_patterns=profile.configs,
        dependency_paths=profile.dependencies,
        startup_patterns=profile.hooks,
        environment_keys=profile.environment,
        runner_paths=profile.runner_paths,
        trusted_commit=commit,
        shared_controller_sha256=hashlib.sha256(Path(shared_pytest.__file__).read_bytes()).hexdigest(),
    )
