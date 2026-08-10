# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The startup verifier must accept the install metadata ``uvx`` actually writes."""

import contextlib
import dataclasses
import functools
import hashlib
import http.server
import json
import os
import subprocess
import sys
import threading
import zipfile
from collections.abc import Iterator
from pathlib import Path
from unittest import mock

import marin.inference.vllm_server as vllm_server
import pytest
from marin.external_dependencies import VLLM_GPU_RELEASE, VllmGpuWheel
from marin.inference.vllm_release import vllm_gpu_wheel_for_architecture, vllm_gpu_wheel_provenance
from marin.inference.vllm_wheel_entrypoint import deep_gemm_cuda_environment, installed_wheel_url_matches

VERSION = VLLM_GPU_RELEASE.version
H100_WHEEL = vllm_gpu_wheel_for_architecture(VLLM_GPU_RELEASE, "x86_64")
GB200_WHEEL = vllm_gpu_wheel_for_architecture(VLLM_GPU_RELEASE, "aarch64")
PROBE_DISTRIBUTION = "marinuvxprobe"
_LOOPBACK_HOSTS = "127.0.0.1,localhost,::1"
_READ_DIRECT_URL = (
    "import importlib.metadata\n"
    f"print(importlib.metadata.distribution({PROBE_DISTRIBUTION!r}).read_text('direct_url.json'))\n"
)


def _uvx_direct_url(url: str) -> dict[str, object]:
    """PEP 610 record ``uvx`` writes for a remote direct reference: URL only, no digest."""
    return {"url": url, "archive_info": {}}


def _provenance(wheel: VllmGpuWheel) -> dict[str, object]:
    return json.loads(json.dumps(dataclasses.asdict(vllm_gpu_wheel_provenance(VLLM_GPU_RELEASE, wheel))))


def _write_vllm_package(root: Path) -> None:
    """Write an importable ``vllm`` package whose CLI records the argv it was handed."""
    package = root / "vllm"
    cli = package / "entrypoints" / "cli"
    cli.mkdir(parents=True)
    for init_file in (package / "__init__.py", package / "entrypoints" / "__init__.py", cli / "__init__.py"):
        init_file.write_text("")
    (package / "_C.py").write_text("EXTENSION_SENTINEL = True\n")
    (cli / "main.py").write_text(
        "import json\n"
        "import os\n"
        "import sys\n"
        "from pathlib import Path\n"
        "\n"
        "def main():\n"
        "    Path(os.environ['FAKE_VLLM_MARKER']).write_text(json.dumps(sys.argv[1:]))\n"
    )


def _write_installed_wheel(
    root: Path,
    *,
    direct_url: dict[str, object],
    compute_capability: tuple[int, int],
) -> None:
    """Write what ``uvx`` leaves behind: the package, its PEP 610 record, and an importable torch."""
    _write_vllm_package(root)
    metadata = root / f"vllm-{VERSION}.dist-info"
    metadata.mkdir()
    (metadata / "METADATA").write_text(f"Metadata-Version: 2.4\nName: vllm\nVersion: {VERSION}\n")
    (metadata / "direct_url.json").write_text(json.dumps(direct_url))
    (root / "torch.py").write_text(
        "class cuda:\n" "    @staticmethod\n" f"    def get_device_capability(): return {compute_capability!r}\n"
    )


def _run_entrypoint(
    tmp_path: Path,
    *,
    wheel: VllmGpuWheel = H100_WHEEL,
    direct_url: dict[str, object] | None = None,
    compute_capability: tuple[int, int] = (9, 0),
    python_path: tuple[Path, ...] = (),
):
    install_root = tmp_path / "install"
    install_root.mkdir()
    _write_installed_wheel(
        install_root,
        direct_url=_uvx_direct_url(wheel.url) if direct_url is None else direct_url,
        compute_capability=compute_capability,
    )
    with mock.patch("platform.machine", return_value=wheel.architecture):
        command = vllm_server.IsolatedCudaVllm(source=vllm_server.VllmType.MARIN_FORK).command()
    bootstrap_index = command.index("-c")
    wrapped_command = command[bootstrap_index + 2 :]
    marker = tmp_path / "cli.json"
    environment = dict(os.environ)
    environment.update(
        {
            "FAKE_VLLM_MARKER": str(marker),
            "PYTHONPATH": os.pathsep.join(str(path) for path in (*python_path, install_root)),
        }
    )
    return (
        subprocess.run(
            [
                sys.executable,
                wrapped_command[0],
                json.dumps(_provenance(wheel)),
                "serve",
                "test/model",
            ],
            capture_output=True,
            text=True,
            env=environment,
            check=False,
        ),
        marker,
    )


def _build_wheel(directory: Path) -> Path:
    """Write a dependency-free wheel that ``uvx`` can install from a URL."""
    dist_info = f"{PROBE_DISTRIBUTION}-1.0.0.dist-info"
    wheel_path = directory / f"{PROBE_DISTRIBUTION}-1.0.0-py3-none-any.whl"
    with zipfile.ZipFile(wheel_path, "w") as archive:
        archive.writestr(f"{PROBE_DISTRIBUTION}/__init__.py", "")
        archive.writestr(
            f"{dist_info}/METADATA",
            f"Metadata-Version: 2.1\nName: {PROBE_DISTRIBUTION}\nVersion: 1.0.0\n",
        )
        archive.writestr(
            f"{dist_info}/WHEEL",
            "Wheel-Version: 1.0\nGenerator: test\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
        )
        archive.writestr(f"{dist_info}/RECORD", "")
    return wheel_path


@contextlib.contextmanager
def _serve_directory(directory: Path) -> Iterator[str]:
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(directory))
    with http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            yield f"http://127.0.0.1:{server.server_address[1]}"
        finally:
            server.shutdown()
            thread.join(timeout=10)


def test_verifier_accepts_the_record_a_real_uvx_install_writes(tmp_path):
    """Feed the verifier's URL check a PEP 610 record from a real hashed ``uvx`` install."""
    wheel_path = _build_wheel(tmp_path)
    digest = hashlib.sha256(wheel_path.read_bytes()).hexdigest()

    with _serve_directory(tmp_path) as base_url:
        url = f"{base_url}/{wheel_path.name}"
        result = subprocess.run(
            [
                "uvx",
                "--quiet",
                "--python",
                sys.executable,
                "--from",
                f"{PROBE_DISTRIBUTION} @ {url}#sha256={digest}",
                "python",
                "-c",
                _READ_DIRECT_URL,
            ],
            capture_output=True,
            text=True,
            # A configured HTTP_PROXY would otherwise take the loopback wheel URL.
            env={**os.environ, "NO_PROXY": _LOOPBACK_HOSTS, "no_proxy": _LOOPBACK_HOSTS},
            check=False,
        )

    assert result.returncode == 0, result.stderr
    recorded = json.loads(result.stdout)
    assert installed_wheel_url_matches(recorded, url)
    # No digest survives the install, which is why startup verification cannot check one.
    assert digest not in json.dumps(recorded)
    # The fake installs below stand in for this record; keep them honest about its shape.
    assert recorded == _uvx_direct_url(url)


def test_deep_gemm_cuda_environment_uses_packaged_nvcc_and_cccl(tmp_path):
    nvidia_root = tmp_path / "site-packages" / "nvidia"
    compiler_root = nvidia_root / "cu13"
    (compiler_root / "bin").mkdir(parents=True)
    (compiler_root / "bin" / "nvcc").touch()
    cccl_root = nvidia_root / "cuda_cccl" / "include"
    (cccl_root / "nv").mkdir(parents=True)
    (cccl_root / "nv" / "target").touch()
    (cccl_root / "cuda" / "std").mkdir(parents=True)
    (cccl_root / "cuda" / "std" / "type_traits").touch()

    environment = deep_gemm_cuda_environment((nvidia_root,), tmp_path / "runtime")

    compatibility_include = tmp_path / "runtime" / "marin-deep-gemm-cuda" / "include"
    assert environment == {
        "CUDA_HOME": str(compiler_root),
        "DG_JIT_USE_NVRTC": "0",
        "NVCC_PREPEND_FLAGS": f"-I{compatibility_include} -I{cccl_root}",
    }
    assert (compatibility_include / "cccl").resolve() == cccl_root.resolve()


def test_deep_gemm_cuda_environment_rejects_missing_compiler(tmp_path):
    nvidia_root = tmp_path / "site-packages" / "nvidia"
    cccl_root = nvidia_root / "cuda_cccl" / "include"
    (cccl_root / "nv").mkdir(parents=True)
    (cccl_root / "nv" / "target").touch()
    (cccl_root / "cuda" / "std").mkdir(parents=True)
    (cccl_root / "cuda" / "std" / "type_traits").touch()

    with pytest.raises(RuntimeError, match="Expected one packaged CUDA compiler root"):
        deep_gemm_cuda_environment((nvidia_root,), tmp_path / "runtime")


@pytest.mark.parametrize(
    ("wheel", "compute_capability"),
    [(H100_WHEEL, (9, 0)), (GB200_WHEEL, (10, 0))],
    ids=["h100", "gb200"],
)
def test_wheel_entrypoint_verifies_extension_and_records_provenance(tmp_path, wheel, compute_capability):
    result, marker = _run_entrypoint(tmp_path, wheel=wheel, compute_capability=compute_capability)

    assert result.returncode == 0, result.stderr
    records = {
        sentinel: json.loads(payload)
        for line in result.stdout.splitlines()
        for sentinel, _, payload in (line.partition("="),)
    }
    assert records["MARIN_VLLM_WHEEL_SELECTED"] == _provenance(wheel)
    assert records["MARIN_VLLM_WHEEL_VERIFIED"] == {
        **_provenance(wheel),
        "compute_capability": f"{compute_capability[0]}.{compute_capability[1]}",
        "extension_path": str(tmp_path / "install" / "vllm" / "_C.py"),
    }
    assert json.loads(marker.read_text()) == ["serve", "test/model"]


def _assert_refused_before_cli(result, marker) -> None:
    assert result.returncode != 0
    selected_record = result.stdout.splitlines()[0]
    sentinel, _, payload = selected_record.partition("=")
    assert sentinel == "MARIN_VLLM_WHEEL_SELECTED"
    assert json.loads(payload) == _provenance(H100_WHEEL)
    assert not marker.exists()


@pytest.mark.parametrize(
    ("direct_url", "compute_capability"),
    [
        (_uvx_direct_url(GB200_WHEEL.url), (9, 0)),
        (None, (8, 0)),
    ],
    ids=["other-architecture-wheel", "unsupported-gpu"],
)
def test_wheel_entrypoint_fails_before_cli_for_unverified_install(tmp_path, direct_url, compute_capability):
    result, marker = _run_entrypoint(tmp_path, direct_url=direct_url, compute_capability=compute_capability)

    _assert_refused_before_cli(result, marker)


def test_wheel_entrypoint_rejects_an_extension_shadowing_the_verified_wheel(tmp_path):
    """A vLLM earlier on ``PYTHONPATH`` supplies the extension while the promoted metadata still reads."""
    shadow_root = tmp_path / "shadow"
    shadow_root.mkdir()
    _write_vllm_package(shadow_root)

    result, marker = _run_entrypoint(tmp_path, python_path=(shadow_root,))

    _assert_refused_before_cli(result, marker)
    assert f"vllm._C loaded outside the verified distribution: {shadow_root / 'vllm' / '_C.py'}" in result.stderr
