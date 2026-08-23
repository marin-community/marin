# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Install a self-built PJRT wheel on the train tasks.

Research scaffolding for the ragged EP64 tuning loop, not for the production branch. EP64 on
a ragged all-to-all cannot start on a stock runtime: XLA fixes ``MultiGpuBarrierKernel::kMaxPeers``
at 32 and takes the barrier fast path whenever the LSA domain covers the clique, so a 64-device
NVL72 clique writes 64 peer slots into a 32-slot allocation and every rank dies with
``CUDA_ERROR_ILLEGAL_ADDRESS`` before the first step (openxla/xla#47283). Every arm in the loop
therefore runs on a wheel that raises the cap.

Set ``MARIN_PJRT_WHEEL`` on the launcher job to an object-storage URL and the train tasks install
it over the stock nightly set. Leave it unset and nothing changes.
"""

import os
import sys
from collections.abc import Sequence

from iris.cluster.setup_scripts import cuda_toolchain_setup_script, default_setup_script, wants_gpu_extra

PJRT_WHEEL_ENV = "MARIN_PJRT_WHEEL"

_NIGHTLY = "https://us-python.pkg.dev/ml-oss-artifacts-published/jax-public-nightly-artifacts-registry"
# The siblings must track the jax revision the PJRT wheel was built from: the plugin carries an
# exact `==` pin on the pjrt version and the two share an ABI.
_STOCK_SIBLINGS: tuple[str, ...] = (
    f"{_NIGHTLY}/jax/jax-0.11.2.dev20260821-py3-none-any.whl",
    f"{_NIGHTLY}/jaxlib/jaxlib-0.11.2.dev20260821-cp312-cp312-manylinux_2_27_aarch64.whl",
    f"{_NIGHTLY}/jax-cuda13-plugin/jax_cuda13_plugin-0.11.2.dev20260821-cp312-cp312-manylinux_2_27_aarch64.whl",
)
_STOCK_PJRT = f"{_NIGHTLY}/jax-cuda13-pjrt/jax_cuda13_pjrt-0.11.2.dev20260821-py3-none-manylinux_2_27_aarch64.whl"


def _install_script(wheel_url: str) -> str:
    jax_url, jaxlib_url, plugin_url = _STOCK_SIBLINGS
    # Full dependency resolution with the with-cuda extra. Installing the stock set with --no-deps
    # leaves the venv's baseline nvidia-* runtime in place, and with those the one-shot ragged path
    # never engages: LsaSize() comes back empty and the thunk falls back with INVALID_ARGUMENT.
    stock = (
        f'"jax @ {jax_url}" "jaxlib @ {jaxlib_url}" '
        f'"jax-cuda13-plugin[with-cuda] @ {plugin_url}" "jax-cuda13-pjrt @ {_STOCK_PJRT}"'
    )
    return f"""set -e
: "${{IRIS_WORKDIR:?}}"
: "${{IRIS_VENV:?}}"
wheel_dir="$IRIS_WORKDIR/.ra2a-pjrt"
rm -rf "$wheel_dir"
mkdir -p "$wheel_dir"
echo 'downloading self-built PJRT wheel'
"$IRIS_VENV/bin/python" - <<'PY'
import os
from pathlib import Path

import fsspec

wheel_url = {wheel_url!r}
wheel_dir = Path(os.environ["IRIS_WORKDIR"]) / ".ra2a-pjrt"
filesystem, remote_path = fsspec.core.url_to_fs(wheel_url)
filesystem.get(remote_path, str(wheel_dir / remote_path.rsplit("/", 1)[1]))
PY
echo 'installing stock nightly set with dependencies'
uv pip install --python "$IRIS_VENV/bin/python" --reinstall {stock}
echo 'overlaying self-built PJRT'
uv pip install --python "$IRIS_VENV/bin/python" --no-deps --reinstall "$wheel_dir"/*.whl
"$IRIS_VENV/bin/python" - <<'PY'
from importlib.metadata import version

for pkg in ("jax", "jax-cuda13-pjrt", "nvidia-nccl-cu13", "nvidia-nvshmem-cu13"):
    print(pkg, version(pkg))
PY
"""


def pjrt_wheel_setup_scripts(*, extras: Sequence[str]) -> tuple[str, ...]:
    """Full setup script list ending in the ``MARIN_PJRT_WHEEL`` install, or none when unset.

    Passing ``setup_scripts`` replaces iris's default setup rather than adding to it, so the
    default uv sync and the CUDA toolchain staging have to be rebuilt here in the order
    ``EnvironmentSpec.to_proto`` would have used. Getting this wrong leaves no venv at all: the
    first attempt shipped the wheel script alone and every rank died on a missing
    ``$IRIS_VENV/bin/python``.

    The wheel install goes last, after the toolchain staging, matching the ordering these wheels
    have been validated under.
    """
    wheel_url = os.environ.get(PJRT_WHEEL_ENV)
    if not wheel_url:
        return ()
    extras = list(extras)
    scripts = [
        default_setup_script(
            extras=extras,
            python_version=f"{sys.version_info.major}.{sys.version_info.minor}",
        )
    ]
    if wants_gpu_extra(extras):
        scripts.append(cuda_toolchain_setup_script())
    scripts.append(_install_script(wheel_url))
    return tuple(scripts)
