# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Sideload a self-built jax-cuda13-pjrt wheel into the worker env at task start.

The campaign env installs the locked PJRT wheel from pyproject; measurement arms that test an
adhoc PJRT build replace exactly that one package before the task command runs. Everything else
in the env stays as locked, so the arm differs from its control by the single wheel.
"""


def pjrt_wheel_install_script(wheel_url: str) -> str:
    """Worker setup script fetching ``wheel_url`` (a ``.whl`` or a directory of one) and
    reinstalling jax-cuda13-pjrt from it.

    The URL is any fsspec-readable location; workers read object storage with their own
    credentials, so an s3 research prefix works without presigning.
    """
    return f"""set -e
: "${{IRIS_WORKDIR:?}}"
: "${{IRIS_VENV:?}}"
wheel_dir="$IRIS_WORKDIR/.marin-adhoc-pjrt"
rm -rf "$wheel_dir"
mkdir -p "$wheel_dir"
echo 'downloading adhoc PJRT wheel'
"$IRIS_VENV/bin/python" - <<'PY'
import os
from pathlib import Path

import fsspec

wheel_url = {wheel_url!r}
wheel_dir = Path(os.environ["IRIS_WORKDIR"]) / ".marin-adhoc-pjrt"
filesystem, remote_path = fsspec.core.url_to_fs(wheel_url)
if remote_path.endswith(".whl"):
    matches = [remote_path]
else:
    matches = filesystem.glob(remote_path.rstrip("/") + "/*.whl")
if not matches:
    raise FileNotFoundError(f"no .whl under {{wheel_url}}")
for match in matches:
    filesystem.get(match, str(wheel_dir / match.rsplit("/", 1)[1]))
    print("fetched", match)
PY
echo 'reinstalling jax-cuda13-pjrt from the adhoc wheel'
uv pip install --python "$IRIS_VENV/bin/python" --no-deps --reinstall "$wheel_dir"/*.whl
"$IRIS_VENV/bin/python" -c \\
  "from importlib.metadata import version; print('sideloaded jax-cuda13-pjrt', version('jax-cuda13-pjrt'))"
"""
