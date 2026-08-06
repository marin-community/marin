# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
import os
import subprocess
import sys
from pathlib import Path

import marin.inference.vllm_server as vllm_server
import pytest

VERSION = "0.0.0.dev20260805+marin.fa50698a9a30.cu129"
PROVENANCE = {
    "release_tag": "marin-vllm-gpu-20260805-fa50698a9a30",
    "sm_targets": ["9.0"],
    "source_commit": "fa50698a9a303f7282aa0e969f35717703de4911",
    "version": VERSION,
    "wheel_sha256": "d4e5d6e19da49c0f1dd030bd14d3ab795a10b8f1185c55162ae5daf6745c98eb",
    "wheel_url": "https://example.invalid/vllm.whl",
}


def _write_fake_vllm(
    tmp_path: Path,
    *,
    version: str = VERSION,
    include_extension: bool = True,
    wheel_sha256: str = PROVENANCE["wheel_sha256"],
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
    (metadata / "direct_url.json").write_text(
        json.dumps(
            {
                "archive_info": {"hashes": {"sha256": wheel_sha256}},
                "url": PROVENANCE["wheel_url"],
            }
        )
    )
    (tmp_path / "torch.py").write_text(
        "class cuda:\n"
        "    @staticmethod\n"
        f"    def get_device_capability(): return {compute_capability!r}\n"
    )


def _run_entrypoint(
    tmp_path: Path,
    *,
    version: str = VERSION,
    include_extension: bool = True,
    wheel_sha256: str = PROVENANCE["wheel_sha256"],
    compute_capability: tuple[int, int] = (9, 0),
):
    _write_fake_vllm(
        tmp_path,
        version=version,
        include_extension=include_extension,
        wheel_sha256=wheel_sha256,
        compute_capability=compute_capability,
    )
    entrypoint = Path(vllm_server.__file__).with_name("vllm_wheel_entrypoint.py")
    marker = tmp_path / "cli.json"
    environment = dict(os.environ)
    environment.update(
        {
            "FAKE_VLLM_MARKER": str(marker),
            "MARIN_VLLM_WHEEL_PROVENANCE": json.dumps(PROVENANCE),
            "PYTHONPATH": str(tmp_path),
        }
    )
    return (
        subprocess.run(
            [sys.executable, str(entrypoint), "serve", "test/model"],
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
    assert records["MARIN_VLLM_WHEEL_SELECTED"] == PROVENANCE
    assert records["MARIN_VLLM_WHEEL_VERIFIED"] == {
        **PROVENANCE,
        "compute_capability": "9.0",
        "extension_path": str(tmp_path / "vllm" / "_C.py"),
    }
    assert json.loads(marker.read_text()) == ["serve", "test/model"]


@pytest.mark.parametrize(
    ("version", "include_extension", "wheel_sha256", "compute_capability"),
    [
        ("0.0.0.dev0+wrong", True, PROVENANCE["wheel_sha256"], (9, 0)),
        (VERSION, False, PROVENANCE["wheel_sha256"], (9, 0)),
        (VERSION, True, "0" * 64, (9, 0)),
        (VERSION, True, PROVENANCE["wheel_sha256"], (8, 0)),
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
    assert json.loads(payload) == PROVENANCE
    assert not marker.exists()
