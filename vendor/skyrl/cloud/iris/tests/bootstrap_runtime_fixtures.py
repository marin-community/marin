# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import os
from pathlib import Path
import shutil
import subprocess
import sys

FIXTURES = Path(__file__).with_name("bootstrap_fixtures")


def fake_frozen_runtime(
    tmp_path: Path, *, managed_python_pin_is_unresolvable: bool = False
) -> tuple[Path, dict[str, str]]:
    environment = tmp_path / "runtime"
    subprocess.run([sys.executable, "-m", "venv", "--without-pip", str(environment)], check=True)
    site_packages = next((environment / "lib").glob("python*/site-packages"))

    (site_packages / "nvidia/cu13/lib").mkdir(parents=True)
    shutil.copytree(FIXTURES / "site-packages", site_packages, dirs_exist_ok=True)

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    (fake_bin / "python3.12").symlink_to(sys.executable)
    uv = fake_bin / "uv"
    shutil.copyfile(FIXTURES / "uv.sh", uv)
    uv.chmod(0o755)
    process_environment = os.environ | {"PATH": f"{fake_bin}:{os.environ['PATH']}"}
    if managed_python_pin_is_unresolvable:
        process_environment["EXPECTED_SYSTEM_PYTHON"] = str(fake_bin / "python3.12")
    return environment, process_environment
