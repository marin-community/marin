"""Delegate native cache preparation to an isolated, frozen Marin CPU runtime."""

import json
import os
import subprocess
from pathlib import Path

from rigging.config_discovery import find_project_root

from marinskyrl.native_environment_control import CONTROL_KEY


def main() -> None:
    workspace = find_project_root(Path(__file__))
    if workspace is None:
        raise RuntimeError("Native cache preparation requires the shipped Marin workspace")
    profile = json.loads(os.environ[CONTROL_KEY])
    environment = dict(os.environ)
    for name in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV", CONTROL_KEY):
        environment.pop(name, None)
    environment["UV_PROJECT_ENVIRONMENT"] = str(Path(profile["venv_alias"]).parent / "marin-cache-runtime")
    uv_version = (workspace / "vendor/skyrl/.uv-version").read_text().strip()
    command = [
        "uv", "tool", "run", "--from", f"uv=={uv_version}",
        "--python", "==3.12.*", "--no-python-downloads", "uv", "run",
        "--project", str(workspace), "--frozen", "--package", "marin-core",
        "--extra", "cpu", "--no-default-groups", "--python", "==3.12.*", "--no-python-downloads",
        "python", "-m", "experiments.post_training.long_loop.native_environment_cache",
    ]
    subprocess.run(command, input=json.dumps(profile), text=True, env=environment, cwd=workspace, check=True)


if __name__ == "__main__":
    main()
