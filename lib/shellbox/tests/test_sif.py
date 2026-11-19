# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import os
import shutil
import sys
from pathlib import Path

import pytest
from shellbox.image import PreparedImage, RegistryImage, prepared_image_config
from shellbox.sif import (
    OciRuntime,
    OciTools,
    filesystem_entries,
    oci_from_filesystem,
    require_equal_filesystems,
    sif_runtime,
)

pytestmark = pytest.mark.skipif(
    sys.platform != "linux", reason="SIF filesystem proofs require Linux inode/xattr semantics"
)


def test_filesystem_proof_rejects_lost_hardlinks_with_identical_content(tmp_path: Path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "data").write_bytes(b"original payload")
    os.link(source / "data", source / "alias")
    (source / "reference").symlink_to("data")
    shutil.copytree(source, tmp_path / "copy", symlinks=True)
    copy = tmp_path / "copy"

    with pytest.raises(ValueError, match="data"):
        require_equal_filesystems(filesystem_entries(source), filesystem_entries(copy))

    (copy / "data").unlink()
    os.link(copy / "alias", copy / "data")
    metadata = source.stat()
    os.utime(copy, ns=(metadata.st_atime_ns, metadata.st_mtime_ns))
    require_equal_filesystems(filesystem_entries(source), filesystem_entries(copy))


@pytest.mark.parametrize("change", ["bytes", "permissions", "symlink"])
def test_filesystem_proof_detects_content_and_permission_changes(tmp_path: Path, change: str):
    source = tmp_path / "source"
    source.mkdir()
    (source / "payload").write_bytes(b"same-size original")
    (source / "reference").symlink_to("payload")
    baseline = filesystem_entries(source)
    if change == "bytes":
        (source / "payload").write_bytes(b"same-size modified")
    elif change == "permissions":
        (source / "payload").chmod(0o755)
    else:
        (source / "reference").unlink()
        (source / "reference").symlink_to("different")
    with pytest.raises(ValueError, match=r"payload|reference"):
        require_equal_filesystems(baseline, filesystem_entries(source))


def test_oci_roundtrip_preserves_files_and_source_runtime(tmp_path: Path):
    umoci, tar = shutil.which("umoci"), shutil.which("tar")
    if umoci is None or tar is None:
        pytest.skip("The offline OCI roundtrip requires umoci and GNU tar")
    source = tmp_path / "source"
    source.mkdir()
    (source / "work").mkdir()
    (source / "work" / "executable").write_bytes(b"#!/bin/sh\nexit 0\n")
    (source / "work" / "executable").chmod(0o751)
    os.setxattr(source / "work" / "executable", "user.fidelity", b"retained image attribute")
    os.link(source / "work" / "executable", source / "alias")
    (source / "link").symlink_to("work/executable")
    runtime = OciRuntime(
        ("LANG=C.UTF-8", "VALUE=with spaces"), ("/work/executable",), (), "/work", "", {"origin": "SIF"}
    )

    converted = oci_from_filesystem(source, runtime, tmp_path / "converted", OciTools(Path(tar), Path(umoci)))

    image = PreparedImage(converted.layout, converted.manifest_digest, RegistryImage("offline-fixture"))
    assert prepared_image_config(image)["config"] == {
        "Env": ["LANG=C.UTF-8", "VALUE=with spaces"],
        "Entrypoint": ["/work/executable"],
        "Cmd": [],
        "WorkingDir": "/work",
        "User": "",
        "Labels": {"origin": "SIF"},
    }
    restored = converted.layout.parent / "verification" / "rootfs"
    assert (restored / "work" / "executable").read_bytes() == b"#!/bin/sh\nexit 0\n"
    assert (restored / "work" / "executable").stat().st_mode & 0o7777 == 0o751
    assert (restored / "alias").stat().st_ino == (restored / "work" / "executable").stat().st_ino
    assert (restored / "link").readlink() == Path("work/executable")
    assert os.getxattr(restored / "work" / "executable", "user.fidelity") == b"retained image attribute"


def test_sif_environment_uses_image_prompt_and_excludes_host_exports(tmp_path: Path, monkeypatch):
    bubblewrap, busybox = shutil.which("bwrap"), shutil.which("busybox")
    if bubblewrap is None or busybox is None:
        pytest.skip("The isolated SIF environment test requires Bubblewrap and busybox-static")
    root = tmp_path / "root"
    for directory in ["bin", "usr/bin", "proc", "dev", "tmp", "task_file", ".singularity.d/env"]:
        (root / directory).mkdir(parents=True, exist_ok=True)
    shutil.copyfile(busybox, root / "bin/busybox")
    (root / "bin/busybox").chmod(0o755)
    (root / "bin/sh").symlink_to("busybox")
    (root / "usr/bin/env").symlink_to("../../bin/busybox")
    (root / ".singularity.d/runscript").write_text('#!/bin/sh\ncd /task_file\nexec "$@"\n')
    (root / ".singularity.d/labels.json").write_text('{"origin":"fixture"}')
    (root / ".singularity.d/env/90-environment.sh").write_text('export SOURCE_VALUE="with spaces"\n')
    (root / ".singularity.d/env/99-base.sh").write_text('if [ -z "${PS1:-}" ]; then PS1="Apptainer> "; fi\nexport PS1\n')
    monkeypatch.setenv("HOST_ONLY", "must-not-enter-image")
    monkeypatch.setenv("PS1", "host prompt")

    runtime = sif_runtime(root, Path(bubblewrap), metadata_timeout=60)

    assert runtime.environment == ("PS1=Apptainer> ", "SOURCE_VALUE=with spaces")
    assert runtime.working_directory == "/task_file"
