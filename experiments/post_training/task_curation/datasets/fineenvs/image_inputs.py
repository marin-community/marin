# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Preserve native FineEnvs public input files in a derived OCI image."""

import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import tarfile
from dataclasses import dataclass, replace
from decimal import Decimal
from pathlib import Path, PurePosixPath
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator
from shellbox.sif import (
    FilesystemEntry,
    OciRuntime,
    OciTools,
    file_sha256,
    filesystem_entries,
    oci_from_filesystem,
    require_equal_filesystems,
)

INPUT_PREFIX = PurePosixPath("setup_files/inputs")
INERT_PREFIX = PurePosixPath("opt/fineenvs-inputs")
BASE_TAG = "source"
IMAGE_TAG = "image"


class NativeInputFile(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    path: str
    bytes: int = Field(ge=0)
    mode: int = Field(ge=0, le=0o7777)
    mtime: str
    type: Literal["0"]
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def _native_regular_file(self) -> "NativeInputFile":
        path = PurePosixPath(self.path)
        if (
            path.is_absolute()
            or ".." in path.parts
            or path.as_posix() != self.path
            or not path.is_relative_to(INPUT_PREFIX)
            or path == INPUT_PREFIX
        ):
            raise ValueError(f"Invalid native public input: {self.path}")
        Decimal(self.mtime)
        return self

    @property
    def mtime_ns(self) -> int:
        return int(Decimal(self.mtime) * 1_000_000_000)


def input_signature(files: tuple[NativeInputFile, ...]) -> str:
    records = [file.model_dump(mode="json") for file in sorted(files, key=lambda file: file.path)]
    return hashlib.sha256(json.dumps(records, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def install_native_inputs(
    root: Path, package: Path, files: tuple[NativeInputFile, ...], destination_prefix: PurePosixPath
) -> None:
    """Stream only the declared public inputs, preserving native bytes, modes and times."""
    if destination_prefix.is_absolute() or ".." in destination_prefix.parts:
        raise ValueError(f"Input image destination must be root-relative: {destination_prefix}")
    wanted = {file.path: file for file in files}
    if not wanted or len(wanted) != len(files):
        raise ValueError("Native input set is empty or repeats a file")
    seen = set()
    with tarfile.open(package, "r|*") as archive:
        for member in archive:
            name = member.name.removeprefix("./")
            if member.isdir() or not PurePosixPath(name).is_relative_to(INPUT_PREFIX):
                continue
            if name not in wanted or name in seen or not member.isfile():
                raise ValueError(f"Native input member differs from the frozen inventory: {name}")
            expected = wanted[name]
            mtime = str(member.pax_headers.get("mtime", member.mtime))
            if (member.size, member.mode, Decimal(mtime)) != (
                expected.bytes,
                expected.mode,
                Decimal(expected.mtime),
            ):
                raise ValueError(f"Native input metadata changed: {name}")
            destination = root / destination_prefix / PurePosixPath(name).relative_to(INPUT_PREFIX)
            if any(parent.is_symlink() for parent in (destination, *destination.parents)):
                raise ValueError(f"Native input path traverses an image symlink: {name}")
            if destination.exists():
                raise ValueError(f"Native input would replace a base-image entry: {name}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            source = archive.extractfile(member)
            assert source is not None
            with destination.open("xb") as output:
                shutil.copyfileobj(source, output, length=1024 * 1024)
            if destination.stat().st_size != expected.bytes or file_sha256(destination) != expected.sha256:
                raise ValueError(f"Native input checksum changed: {name}")
            destination.chmod(expected.mode)
            os.utime(destination, ns=(expected.mtime_ns, expected.mtime_ns))
            seen.add(name)
    if seen != wanted.keys():
        raise ValueError(f"Native package lacks frozen inputs: {sorted(wanted.keys() - seen)}")


def require_base_preserved(
    before: tuple[FilesystemEntry, ...], after: tuple[FilesystemEntry, ...], input_paths: tuple[str, ...]
) -> None:
    """Permit only new input paths and parent-directory timestamps to differ."""
    original = {entry.path: entry for entry in before}
    derived = {entry.path: entry for entry in after}
    parents = {parent.as_posix() for name in input_paths for parent in PurePosixPath(name).parents}
    additions = set(input_paths) | parents
    if set(derived) - set(original) != additions - set(original) or set(original) - set(derived):
        raise ValueError("Derived image changed paths outside the native public input set")
    for name, entry in original.items():
        actual = derived[name]
        if entry.kind == stat.S_IFDIR and name in parents:
            actual = replace(actual, mtime_ns=entry.mtime_ns)
        if actual != entry:
            raise ValueError(f"Derived image changed a base-image entry: {name}")


@dataclass(frozen=True)
class OciImageDocuments:
    manifest: dict
    config: dict
    manifest_digest: str


def _oci_document(layout: Path, digest: str) -> dict:
    if re.fullmatch(r"sha256:[0-9a-f]{64}", digest) is None:
        raise ValueError(f"Invalid OCI content digest: {digest}")
    path = layout / "blobs/sha256" / digest.removeprefix("sha256:")
    if "sha256:" + file_sha256(path) != digest:
        raise ValueError(f"OCI document checksum changed: {digest}")
    return json.loads(path.read_bytes())


def image_documents(layout: Path, tag: str) -> OciImageDocuments:
    """Read the exact tagged manifest and config from a local OCI layout."""
    index = json.loads((layout / "index.json").read_bytes())
    matches = [
        row for row in index["manifests"] if row.get("annotations", {}).get("org.opencontainers.image.ref.name") == tag
    ]
    if len(matches) != 1:
        raise ValueError(f"OCI layout lacks exactly one {tag!r} tag")
    digest = matches[0]["digest"]
    manifest = _oci_document(layout, digest)
    return OciImageDocuments(manifest, _oci_document(layout, manifest["config"]["digest"]), digest)


def _store_document(layout: Path, document: dict, media_type: str) -> dict:
    data = json.dumps(document, sort_keys=True, separators=(",", ":")).encode()
    digest = hashlib.sha256(data).hexdigest()
    (layout / "blobs/sha256" / digest).write_bytes(data)
    return {"mediaType": media_type, "digest": f"sha256:{digest}", "size": len(data)}


def _append_input_layer(layout: Path, root: Path, input_paths: tuple[str, ...], output: Path, tools: OciTools) -> None:
    # umoci repack rounds fractional mtimes; the maintained PAX writer preserves them.
    names = set(input_paths)
    names.update(parent.as_posix() for name in input_paths for parent in PurePosixPath(name).parents)
    delta = output / "input-layer-root"
    delta.mkdir()
    for name in sorted(names):
        source, target = root / name, delta / name
        if source.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        else:
            shutil.copy2(source, target)
        os.chown(target, source.stat().st_uid, source.stat().st_gid)
    # Copy directory metadata after creating children, including the root directory.
    for name in sorted(names, reverse=True):
        shutil.copystat(root / name, delta / name)
    layer = oci_from_filesystem(delta, OciRuntime((), (), (), "", "", {}), output / "input-layer", tools)
    added = image_documents(layer.layout, IMAGE_TAG)
    original = image_documents(layout, BASE_TAG)
    (descriptor,) = added.manifest["layers"]
    blob = descriptor["digest"].removeprefix("sha256:")
    shutil.copyfile(layer.layout / "blobs/sha256" / blob, layout / "blobs/sha256" / blob)
    config = {
        **original.config,
        "rootfs": {
            **original.config["rootfs"],
            "diff_ids": [
                *original.config["rootfs"]["diff_ids"],
                layer.diff_id,
            ],
        },
    }
    config["history"] = [*config.get("history", []), {"created_by": "FineEnvs frozen public inputs"}]
    config_descriptor = _store_document(layout, config, original.manifest["config"]["mediaType"])
    manifest = {**original.manifest, "config": config_descriptor, "layers": [*original.manifest["layers"], descriptor]}
    manifest_descriptor = _store_document(layout, manifest, "application/vnd.oci.image.manifest.v1+json")
    index_path = layout / "index.json"
    index = json.loads(index_path.read_bytes())
    index["manifests"].append({**manifest_descriptor, "annotations": {"org.opencontainers.image.ref.name": IMAGE_TAG}})
    index_path.write_text(json.dumps(index, sort_keys=True, separators=(",", ":")))


@dataclass(frozen=True)
class DerivedFilesystem:
    layout: Path
    manifest_digest: str
    input_signatures: tuple[str, ...]
    entries: tuple[FilesystemEntry, ...]


@dataclass(frozen=True)
class NativeInputSet:
    package: Path
    files: tuple[NativeInputFile, ...]

    @property
    def signature(self) -> str:
        return input_signature(self.files)

    @property
    def image_prefix(self) -> PurePosixPath:
        return INERT_PREFIX / self.signature

    @property
    def image_paths(self) -> tuple[str, ...]:
        prefix = self.image_prefix
        return tuple((prefix / PurePosixPath(file.path).relative_to(INPUT_PREFIX)).as_posix() for file in self.files)


def derive_group_image(
    base_layout: Path, input_sets: tuple[NativeInputSet, ...], output: Path, tools: OciTools
) -> DerivedFilesystem:
    """Add SHA-keyed public input sets while preserving the base filesystem and runtime."""
    signatures = tuple(inputs.signature for inputs in input_sets)
    if not signatures or len(set(signatures)) != len(signatures):
        raise ValueError("Input image group must contain distinct nonempty input sets")
    output.mkdir(parents=True, exist_ok=False)
    layout = output / "oci"
    shutil.copytree(base_layout, layout)
    original_image = image_documents(layout, BASE_TAG)
    base_config = original_image.config
    bundle = output / "modified"
    subprocess.run(
        [str(tools.umoci), "unpack", "--rootless", "--image", f"{layout}:{BASE_TAG}", str(bundle)], check=True
    )
    root = bundle / "rootfs"
    original = filesystem_entries(root)
    for inputs in input_sets:
        install_native_inputs(root, inputs.package, inputs.files, inputs.image_prefix)
    input_paths = tuple(name for inputs in input_sets for name in inputs.image_paths)
    expected = filesystem_entries(root)
    require_base_preserved(original, expected, input_paths)
    _append_input_layer(layout, root, input_paths, output, tools)
    image = image_documents(layout, IMAGE_TAG)
    config = image.config
    if config.get("config") != base_config.get("config") or (config["architecture"], config["os"]) != (
        base_config["architecture"],
        base_config["os"],
    ):
        raise ValueError("Derived image changed the base runtime configuration")
    verification = output / "verified"
    subprocess.run(
        [str(tools.umoci), "unpack", "--rootless", "--image", f"{layout}:{IMAGE_TAG}", str(verification)], check=True
    )
    observed = filesystem_entries(verification / "rootfs")
    require_equal_filesystems(expected, observed)
    if image.manifest["layers"][:-1] != original_image.manifest["layers"]:
        raise ValueError("Derived image did not preserve the original base layers")
    if config["rootfs"]["diff_ids"][:-1] != base_config["rootfs"]["diff_ids"]:
        raise ValueError("Derived image changed the base filesystem layer identities")
    return DerivedFilesystem(layout, image.manifest_digest, signatures, observed)
