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
from marin.inference.vllm_release import MARIN_VLLM_GPU_RELEASE

VERSION = MARIN_VLLM_GPU_RELEASE.version
WHEEL = MARIN_VLLM_GPU_RELEASE.wheel_for_architecture("x86_64")


def _provenance() -> dict[str, object]:
    return json.loads(json.dumps(dataclasses.asdict(MARIN_VLLM_GPU_RELEASE.provenance(WHEEL))))


def _write_fake_vllm(
    tmp_path: Path,
    *,
    version: str = VERSION,
    include_extension: bool = True,
    wheel_sha256: str = WHEEL.sha256,
    compute_capability: tuple[int, int] = (9, 0),
    hash_in_url_fragment: bool = False,
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
    direct_url = {
        "archive_info": {} if hash_in_url_fragment else {"hashes": {"sha256": wheel_sha256}},
        "url": f"{WHEEL.url}#sha256={wheel_sha256}" if hash_in_url_fragment else WHEEL.url,
    }
    (metadata / "direct_url.json").write_text(json.dumps(direct_url))
    (tmp_path / "torch.py").write_text(
        "class cuda:\n" "    @staticmethod\n" f"    def get_device_capability(): return {compute_capability!r}\n"
    )


def _run_entrypoint(
    tmp_path: Path,
    *,
    version: str = VERSION,
    include_extension: bool = True,
    wheel_sha256: str = WHEEL.sha256,
    compute_capability: tuple[int, int] = (9, 0),
    hash_in_url_fragment: bool = False,
):
    _write_fake_vllm(
        tmp_path,
        version=version,
        include_extension=include_extension,
        wheel_sha256=wheel_sha256,
        compute_capability=compute_capability,
        hash_in_url_fragment=hash_in_url_fragment,
    )
    entrypoint = Path(vllm_server.__file__).with_name("vllm_wheel_entrypoint.py")
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
            [sys.executable, str(entrypoint), json.dumps(_provenance()), "serve", "test/model"],
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


def test_wheel_entrypoint_accepts_uv_hash_fragment_provenance(tmp_path):
    result, marker = _run_entrypoint(tmp_path, hash_in_url_fragment=True)

    assert result.returncode == 0, result.stderr
    assert marker.exists()


@pytest.mark.parametrize(
    ("version", "include_extension", "wheel_sha256", "compute_capability"),
    [
        ("0.0.0.dev0+wrong", True, WHEEL.sha256, (9, 0)),
        (VERSION, False, WHEEL.sha256, (9, 0)),
        (VERSION, True, "0" * 64, (9, 0)),
        (VERSION, True, WHEEL.sha256, (8, 0)),
    ],
)
def test_wheel_entrypoint_fails_before_cli_for_unverified_install(
    tmp_path, version, include_extension, wheel_sha256, compute_capability
):
    result, marker = _run_entrypoint(
        tmp_path,
        version=version,
        include_extension=include_extension,
        wheel_sha256=wheel_sha256,
        compute_capability=compute_capability,
    )

    assert result.returncode != 0
    selected_record = result.stdout.splitlines()[0]
    sentinel, _, payload = selected_record.partition("=")
    assert sentinel == "MARIN_VLLM_WHEEL_SELECTED"
    assert json.loads(payload) == _provenance()
    assert not marker.exists()
