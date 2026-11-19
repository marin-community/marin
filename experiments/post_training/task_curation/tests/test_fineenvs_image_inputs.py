# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import io
import runpy
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path, PurePosixPath

import pytest
from finestore.rl.status_taxonomy import DataRestriction
from marin.execution.artifact import ArtifactRecord, result_type_name, write_record
from shellbox.sif import OciTools, filesystem_entries

from experiments.post_training.task_curation.datasets.fineenvs.artifacts import GroupedInputImage
from experiments.post_training.task_curation.datasets.fineenvs.image_inputs import (
    NativeInputFile,
    NativeInputSet,
    derive_group_image,
    install_native_inputs,
    require_base_preserved,
)


def native_package(tmp_path: Path, payload: bytes) -> tuple[Path, NativeInputFile]:
    package = tmp_path / "native.tar.gz"
    path = "setup_files/inputs/coordinates.csv"
    timestamp = "1700000000.123456789"
    with tarfile.open(package, "w:gz", format=tarfile.PAX_FORMAT) as archive:
        for name, data in ((path, payload), ("tests/grader.py", b"private original grader")):
            member = tarfile.TarInfo(name)
            member.size = len(data)
            member.mode = 0o640
            member.pax_headers = {"mtime": timestamp}
            archive.addfile(member, io.BytesIO(data))
    return package, NativeInputFile(
        path=path,
        bytes=len(payload),
        mode=0o640,
        mtime=timestamp,
        type="0",
        sha256=hashlib.sha256(payload).hexdigest(),
    )


def test_large_public_input_keeps_native_content_mode_and_timestamp(tmp_path):
    payload = b"native-coordinate-input\n" * 100_000
    assert len(payload) > 1_000_000
    package, file = native_package(tmp_path, payload)
    original_archive_hash = hashlib.sha256(package.read_bytes()).hexdigest()
    root = tmp_path / "rootfs"
    root.mkdir()

    install_native_inputs(root, package, (file,), PurePosixPath("setup_files/inputs"))

    target = root / file.path
    assert target.read_bytes() == payload
    assert target.stat().st_mode & 0o7777 == 0o640
    assert target.stat().st_mtime_ns == 1_700_000_000_123_456_789
    assert not (root / "tests/grader.py").exists()
    assert hashlib.sha256(package.read_bytes()).hexdigest() == original_archive_hash


def test_native_input_checksum_mismatch_stops_image_preparation(tmp_path):
    package, file = native_package(tmp_path, b"original input")
    root = tmp_path / "rootfs"
    root.mkdir()
    wrong = file.model_copy(update={"sha256": hashlib.sha256(b"different input").hexdigest()})

    with pytest.raises(ValueError, match="Native input checksum changed"):
        install_native_inputs(root, package, (wrong,), PurePosixPath("setup_files/inputs"))


def test_grouped_image_proof_loads_after_producer_entrypoint_exits(tmp_path):
    producer = runpy.run_module(
        "experiments.post_training.task_curation.datasets.fineenvs.images", run_name="fineenvs_images_cli_alias"
    )
    image = producer["GroupedInputImage"](
        data_restriction=DataRestriction.OPEN,
        path=str(tmp_path / "image"),
        group=0,
        base_image="fixture@sha256:" + "a" * 64,
        input_signatures=("b" * 64,),
        manifest_digest="sha256:" + "c" * 64,
        tree_sha256="d" * 64,
        provenance_uri="s3://fixture/provenance.json",
        provenance_sha256="e" * 64,
    )
    write_record(
        ArtifactRecord(output_path=image.path, result_type=result_type_name(type(image)), result=image.result_payload())
    )

    assert GroupedInputImage.raw_load(image.path) == image


@pytest.mark.skipif(sys.platform != "linux", reason="OCI filesystem verification requires Linux inode semantics")
def test_input_image_rejects_changes_to_existing_base_files(tmp_path):
    root = tmp_path / "rootfs"
    root.mkdir()
    base = root / "base-command"
    base.write_bytes(b"unchanged base executable")
    base.chmod(0o755)
    package, file = native_package(tmp_path, b"public data")
    original = filesystem_entries(root)
    install_native_inputs(root, package, (file,), PurePosixPath("setup_files/inputs"))
    added = filesystem_entries(root)
    require_base_preserved(original, added, (file.path,))

    base.chmod(0o644)
    with pytest.raises(ValueError, match="changed a base-image entry: base-command"):
        require_base_preserved(original, filesystem_entries(root), (file.path,))


@pytest.mark.skipif(
    sys.platform != "linux" or shutil.which("umoci") is None,
    reason="OCI round-trip requires Linux and the umoci executable",
)
def test_derived_oci_image_round_trips_large_inputs_and_original_base(tmp_path):
    executable = shutil.which("umoci")
    assert executable is not None
    umoci = Path(executable)
    layout = tmp_path / "base-layout"
    subprocess.run([str(umoci), "init", "--layout", str(layout)], check=True)
    subprocess.run([str(umoci), "new", "--image", f"{layout}:source"], check=True)
    original_bundle = tmp_path / "original"
    subprocess.run([str(umoci), "unpack", "--rootless", "--image", f"{layout}:source", str(original_bundle)], check=True)
    base_file = original_bundle / "rootfs/base-command"
    base_file.write_bytes(b"original base executable")
    base_file.chmod(0o755)
    subprocess.run([str(umoci), "repack", "--image", f"{layout}:source", str(original_bundle)], check=True)
    first = tmp_path / "first"
    first.mkdir()
    package, file = native_package(first, b"original-public-data\n" * 100_000)
    second = tmp_path / "second"
    second.mkdir()
    other_package, other_file = native_package(second, b"another input set\n" * 100_000)
    inputs = NativeInputSet(package, (file,))
    other_inputs = NativeInputSet(other_package, (other_file,))
    tar = shutil.which("tar")
    assert tar is not None

    derive_group_image(layout, (inputs, other_inputs), tmp_path / "derived", OciTools(Path(tar), umoci))

    verified = tmp_path / "derived/verified/rootfs"
    target = verified / inputs.image_paths[0]
    assert target.read_bytes() == b"original-public-data\n" * 100_000
    assert (verified / other_inputs.image_paths[0]).read_bytes() == b"another input set\n" * 100_000
    assert not (verified / file.path).exists()
    assert not (verified / "tests/grader.py").exists()
    assert (verified / "base-command").read_bytes() == b"original base executable"
    assert (verified / "base-command").stat().st_mode & 0o7777 == 0o755
    assert target.stat().st_mtime_ns == 1_700_000_000_123_456_789
    assert target.stat().st_mode & 0o7777 == 0o640
