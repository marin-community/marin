# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import json
import os
import subprocess
import sys
from pathlib import Path

import marin.inference.vllm_server as vllm_server
import pytest
from marin.external_dependencies import VLLM_GPU_RELEASE
from marin.inference.vllm_release import vllm_gpu_wheel_for_architecture, vllm_gpu_wheel_provenance

VERSION = VLLM_GPU_RELEASE.version
WHEEL = vllm_gpu_wheel_for_architecture(VLLM_GPU_RELEASE, "x86_64")


@dataclasses.dataclass(frozen=True)
class DirectUrlHashes:
    archive_sha256: str | None
    fragment_sha256: str | None


ARCHIVE_HASH = DirectUrlHashes(archive_sha256=WHEEL.sha256, fragment_sha256=None)
FRAGMENT_HASH = DirectUrlHashes(archive_sha256=None, fragment_sha256=WHEEL.sha256)
BOTH_HASHES = DirectUrlHashes(archive_sha256=WHEEL.sha256, fragment_sha256=WHEEL.sha256)
NO_HASH = DirectUrlHashes(archive_sha256=None, fragment_sha256=None)


def _provenance() -> dict[str, object]:
    return json.loads(json.dumps(dataclasses.asdict(vllm_gpu_wheel_provenance(VLLM_GPU_RELEASE, WHEEL))))


def _write_fake_vllm(
    tmp_path: Path,
    *,
    version: str = VERSION,
    include_extension: bool = True,
    direct_url_hashes: DirectUrlHashes = ARCHIVE_HASH,
    compute_capability: tuple[int, int] = (9, 0),
) -> None:
    package = tmp_path / "vllm"
    cli = package / "entrypoints" / "cli"
    cli.mkdir(parents=True)
    for init_file in (package / "__init__.py", package / "entrypoints" / "__init__.py", cli / "__init__.py"):
        init_file.write_text("")
    if include_extension:
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
    metadata = tmp_path / f"vllm-{version}.dist-info"
    metadata.mkdir()
    (metadata / "METADATA").write_text(f"Metadata-Version: 2.4\nName: vllm\nVersion: {version}\n")
    archive_info = {}
    if direct_url_hashes.archive_sha256 is not None:
        archive_info["hashes"] = {"sha256": direct_url_hashes.archive_sha256}
    wheel_url = WHEEL.url
    if direct_url_hashes.fragment_sha256 is not None:
        wheel_url = f"{wheel_url}#sha256={direct_url_hashes.fragment_sha256}"
    direct_url = {"archive_info": archive_info, "url": wheel_url}
    (metadata / "direct_url.json").write_text(json.dumps(direct_url))
    (tmp_path / "torch.py").write_text(
        "class cuda:\n" "    @staticmethod\n" f"    def get_device_capability(): return {compute_capability!r}\n"
    )


def _run_entrypoint(
    tmp_path: Path,
    *,
    version: str = VERSION,
    include_extension: bool = True,
    direct_url_hashes: DirectUrlHashes = ARCHIVE_HASH,
    compute_capability: tuple[int, int] = (9, 0),
):
    _write_fake_vllm(
        tmp_path,
        version=version,
        include_extension=include_extension,
        direct_url_hashes=direct_url_hashes,
        compute_capability=compute_capability,
    )
    command = vllm_server.IsolatedCudaVllm(source=vllm_server.VllmType.MARIN_FORK).command()
    bootstrap_index = command.index("-c")
    wrapped_command = command[bootstrap_index + 2 :]
    marker = tmp_path / "cli.json"
    environment = dict(os.environ)
    environment.update(
        {
            "FAKE_VLLM_MARKER": str(marker),
            "PYTHONPATH": str(tmp_path),
        }
    )
    return (
        subprocess.run(
            [
                sys.executable,
                *wrapped_command[1:4],
                json.dumps(_provenance()),
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


def test_wheel_entrypoint_verifies_extension_and_records_provenance(tmp_path):
    result, marker = _run_entrypoint(tmp_path)

    assert result.returncode == 0, result.stderr
    records = {
        sentinel: json.loads(payload)
        for line in result.stdout.splitlines()
        for sentinel, _, payload in (line.partition("="),)
    }
    assert records["MARIN_VLLM_WHEEL_SELECTED"] == _provenance()
    assert records["MARIN_VLLM_WHEEL_VERIFIED"] == {
        **_provenance(),
        "compute_capability": "9.0",
        "extension_path": str(tmp_path / "vllm" / "_C.py"),
    }
    assert json.loads(marker.read_text()) == ["serve", "test/model"]


@pytest.mark.parametrize("direct_url_hashes", [NO_HASH, FRAGMENT_HASH, BOTH_HASHES])
def test_wheel_entrypoint_accepts_supported_hash_provenance(tmp_path, direct_url_hashes):
    result, marker = _run_entrypoint(tmp_path, direct_url_hashes=direct_url_hashes)

    assert result.returncode == 0, result.stderr
    assert marker.exists()


@pytest.mark.parametrize(
    ("version", "include_extension", "direct_url_hashes", "compute_capability"),
    [
        ("0.0.0.dev0+wrong", True, ARCHIVE_HASH, (9, 0)),
        (VERSION, False, ARCHIVE_HASH, (9, 0)),
        (VERSION, True, DirectUrlHashes(archive_sha256="0" * 64, fragment_sha256=None), (9, 0)),
        (VERSION, True, ARCHIVE_HASH, (8, 0)),
        (
            VERSION,
            True,
            DirectUrlHashes(archive_sha256=WHEEL.sha256, fragment_sha256="0" * 64),
            (9, 0),
        ),
    ],
)
def test_wheel_entrypoint_fails_before_cli_for_unverified_install(
    tmp_path, version, include_extension, direct_url_hashes, compute_capability
):
    result, marker = _run_entrypoint(
        tmp_path,
        version=version,
        include_extension=include_extension,
        direct_url_hashes=direct_url_hashes,
        compute_capability=compute_capability,
    )

    assert result.returncode != 0
    selected_record = result.stdout.splitlines()[0]
    sentinel, _, payload = selected_record.partition("=")
    assert sentinel == "MARIN_VLLM_WHEEL_SELECTED"
    assert json.loads(payload) == _provenance()
    assert not marker.exists()
