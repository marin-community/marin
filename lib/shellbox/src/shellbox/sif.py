# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Preserve a pristine SIF filesystem in a verified, single-layer OCI layout."""

import gzip
import hashlib
import json
import os
import re
import shlex
import shutil
import stat
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

from shellbox.image import OCI_TAG

LAYER_MEDIA_TYPE = "application/vnd.oci.image.layer.v1.tar+gzip"
CONFIG_MEDIA_TYPE = "application/vnd.oci.image.config.v1+json"
MANIFEST_MEDIA_TYPE = "application/vnd.oci.image.manifest.v1+json"
ENVIRONMENT_SEPARATOR = b"\x00SIF_ENVIRONMENT_AFTER\x00"


@dataclass(frozen=True)
class SifTools:
    siftool: Path
    unsquashfs: Path


@dataclass(frozen=True)
class OciTools:
    tar: Path
    umoci: Path


@dataclass(frozen=True)
class OciRuntime:
    environment: tuple[str, ...]
    entrypoint: tuple[str, ...]
    command: tuple[str, ...]
    working_directory: str
    user: str
    labels: dict[str, str]

    def process_config(self) -> dict[str, object]:
        return {
            "Env": list(self.environment),
            "Entrypoint": list(self.entrypoint),
            "Cmd": list(self.command),
            "WorkingDir": self.working_directory,
            "User": self.user,
            "Labels": self.labels,
        }


@dataclass(frozen=True)
class FilesystemEntry:
    path: str
    kind: int
    mode: int
    uid: int
    gid: int
    mtime_ns: int
    content_sha256: str | None
    link_target: str | None
    hardlink_to: str | None
    device: int | None
    xattrs: dict[str, str]


@dataclass(frozen=True)
class SifFilesystem:
    root: Path
    sif_sha256: str
    partition_id: int
    partition_sha256: str
    descriptor_listing: str


@dataclass(frozen=True)
class OciFilesystem:
    layout: Path
    manifest_digest: str
    config_digest: str
    diff_id: str
    tree_sha256: str
    entries: tuple[FilesystemEntry, ...]


def _environment(data: bytes) -> dict[str, str]:
    result = {}
    for item in data.split(b"\x00"):
        if not item:
            continue
        key, value = item.split(b"=", 1)
        result[key.decode()] = value.decode()
    return result


def sif_runtime(root: Path, bubblewrap: Path, metadata_timeout: float) -> OciRuntime:
    """Resolve image environment scripts in an isolated, read-only guest and preserve its runscript."""
    runscript = root / ".singularity.d" / "runscript"
    commands = [
        shlex.split(line)
        for line in runscript.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if len(commands) != 2 or commands[0][:1] != ["cd"] or commands[1] != ["exec", "$@"]:
        raise ValueError("SIF runscript needs an explicit working-directory binding")
    if len(commands[0]) != 2 or not commands[0][1].startswith("/"):
        raise ValueError("SIF working directory is not an absolute literal path")
    working_directory = commands[0][1]
    if not (root / working_directory.removeprefix("/")).is_dir():
        raise ValueError("SIF working directory is absent from the filesystem")
    script = (
        "unset PS1; /usr/bin/env -0; printf '\\000SIF_ENVIRONMENT_AFTER\\000'; "
        'for script in /.singularity.d/env/*.sh; do . "$script" >&2; done; /usr/bin/env -0'
    )
    resolved = subprocess.run(
        [
            str(bubblewrap),
            "--unshare-all",
            "--new-session",
            "--die-with-parent",
            "--ro-bind",
            str(root),
            "/",
            "--proc",
            "/proc",
            "--dev",
            "/dev",
            "--tmpfs",
            "/tmp",
            "--clearenv",
            "--chdir",
            "/",
            "--",
            "/bin/sh",
            "-ec",
            script,
        ],
        check=True,
        capture_output=True,
        timeout=metadata_timeout,
    )
    before_bytes, after_bytes = resolved.stdout.split(ENVIRONMENT_SEPARATOR)
    before, after = _environment(before_bytes), _environment(after_bytes)
    # Shell bootstrap variables such as PWD are runtime state, rather than image exports.
    environment = tuple(f"{key}={value}" for key, value in sorted(after.items()) if before.get(key) != value)
    labels = json.loads((root / ".singularity.d" / "labels.json").read_text())
    if any(not isinstance(key, str) or not isinstance(value, str) for key, value in labels.items()):
        raise ValueError("SIF image labels must be strings")
    return OciRuntime(environment, ("/.singularity.d/runscript",), (), working_directory, "", labels)


def file_sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def filesystem_entries(root: Path) -> tuple[FilesystemEntry, ...]:
    """Inventory content, permissions, links, ownership and extended attributes without following links."""
    if sys.platform != "linux":
        raise ValueError("SIF filesystem verification requires Linux inode and extended-attribute semantics")
    paths = [root]
    for directory, children, files in os.walk(root, followlinks=False):
        paths.extend(Path(directory) / name for name in children + files)
    inodes: dict[tuple[int, int], str] = {}
    entries = []
    for path in sorted(paths, key=lambda p: p.relative_to(root).as_posix()):
        name = path.relative_to(root).as_posix()
        metadata = path.lstat()
        kind = stat.S_IFMT(metadata.st_mode)
        if kind == stat.S_IFSOCK:
            raise ValueError(f"OCI cannot preserve a filesystem socket: {name}")
        content = file_sha256(path) if stat.S_ISREG(metadata.st_mode) else None
        target = os.readlink(path) if stat.S_ISLNK(metadata.st_mode) else None
        hardlink = None
        if stat.S_ISREG(metadata.st_mode) and metadata.st_nlink > 1:
            inode = (metadata.st_dev, metadata.st_ino)
            hardlink = inodes.get(inode)
            inodes.setdefault(inode, name)
        attributes = {
            key: hashlib.sha256(os.getxattr(path, key, follow_symlinks=False)).hexdigest()
            for key in sorted(os.listxattr(path, follow_symlinks=False))
        }
        entries.append(
            FilesystemEntry(
                name,
                kind,
                stat.S_IMODE(metadata.st_mode),
                metadata.st_uid,
                metadata.st_gid,
                metadata.st_mtime_ns,
                content,
                target,
                hardlink,
                metadata.st_rdev if stat.S_ISCHR(metadata.st_mode) or stat.S_ISBLK(metadata.st_mode) else None,
                attributes,
            )
        )
    return tuple(entries)


def require_equal_filesystems(source: tuple[FilesystemEntry, ...], converted: tuple[FilesystemEntry, ...]) -> None:
    before = {entry.path: entry for entry in source}
    after = {entry.path: entry for entry in converted}
    changed = sorted(path for path in before.keys() | after.keys() if before.get(path) != after.get(path))
    if changed:
        raise ValueError(f"SIF to OCI changed {len(changed)} filesystem entries: {changed[:20]}")


def unpack_sif(image: Path, expected_sha256: str, output: Path, tools: SifTools, processors: int) -> SifFilesystem:
    """Extract the digest-verified primary amd64 Squashfs partition into a fresh directory."""
    digest = file_sha256(image)
    if digest != expected_sha256:
        raise ValueError("SIF digest differs from the pinned input")
    output.mkdir(parents=True, exist_ok=False)
    listing = subprocess.check_output([str(tools.siftool), "list", str(image)], text=True)
    partitions = re.findall(r"^\s*(\d+)\s*\|.*FS \(Squashfs/\*System/(?:amd64|x86_64)\)", listing, re.MULTILINE)
    if len(partitions) != 1:
        raise ValueError(f"Expected one primary amd64 Squashfs partition: {listing}")
    partition_id = int(partitions[0])
    partition = output / "rootfs.squashfs"
    with partition.open("xb") as stream:
        subprocess.run([str(tools.siftool), "dump", str(partition_id), str(image)], stdout=stream, check=True)
    root = output / "rootfs"
    subprocess.run(
        [str(tools.unsquashfs), "-no-progress", "-processors", str(processors), "-dest", str(root), str(partition)],
        check=True,
        capture_output=True,
    )
    return SifFilesystem(root, digest, partition_id, file_sha256(partition), listing)


def _json_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def _json_blob(layout: Path, value: object, media_type: str) -> dict[str, object]:
    data = _json_bytes(value)
    digest = hashlib.sha256(data).hexdigest()
    (layout / "blobs" / "sha256" / digest).write_bytes(data)
    return {"mediaType": media_type, "digest": f"sha256:{digest}", "size": len(data)}


def oci_from_filesystem(root: Path, runtime: OciRuntime, output: Path, tools: OciTools) -> OciFilesystem:
    """Build a fresh OCI layout, then require its unpacked filesystem to equal the source."""
    original = filesystem_entries(root)
    output.mkdir(parents=True, exist_ok=False)
    layer = output / "filesystem.tar"
    subprocess.run(
        [
            str(tools.tar),
            "--sort=name",
            "--format=pax",
            "--pax-option=exthdr.name=%d/PaxHeaders/%f,delete=atime,delete=ctime",
            "--numeric-owner",
            "--xattrs",
            "--acls",
            "--selinux",
            "-cpf",
            str(layer),
            "-C",
            str(root),
            ".",
        ],
        check=True,
        capture_output=True,
    )
    diff_id = f"sha256:{file_sha256(layer)}"
    layout = output / "oci"
    blobs = layout / "blobs" / "sha256"
    blobs.mkdir(parents=True)
    compressed = output / "filesystem.tar.gz"
    with layer.open("rb") as source, compressed.open("xb") as destination:
        with gzip.GzipFile(filename="", mode="wb", fileobj=destination, mtime=0) as stream:
            shutil.copyfileobj(source, stream)
    layer_digest = file_sha256(compressed)
    compressed.rename(blobs / layer_digest)
    layer_descriptor = {
        "mediaType": LAYER_MEDIA_TYPE,
        "digest": f"sha256:{layer_digest}",
        "size": (blobs / layer_digest).stat().st_size,
    }
    config = _json_blob(
        layout,
        {
            "architecture": "amd64",
            "os": "linux",
            "config": runtime.process_config(),
            "rootfs": {"type": "layers", "diff_ids": [diff_id]},
        },
        CONFIG_MEDIA_TYPE,
    )
    manifest = _json_blob(
        layout,
        {"schemaVersion": 2, "mediaType": MANIFEST_MEDIA_TYPE, "config": config, "layers": [layer_descriptor]},
        MANIFEST_MEDIA_TYPE,
    )
    (layout / "oci-layout").write_bytes(_json_bytes({"imageLayoutVersion": "1.0.0"}))
    (layout / "index.json").write_bytes(
        _json_bytes(
            {
                "schemaVersion": 2,
                "manifests": [{**manifest, "annotations": {"org.opencontainers.image.ref.name": OCI_TAG}}],
            }
        )
    )
    unpacked = output / "verification"
    subprocess.run(
        [str(tools.umoci), "unpack", "--image", f"{layout}:{OCI_TAG}", str(unpacked)],
        check=True,
        capture_output=True,
    )
    require_equal_filesystems(original, filesystem_entries(unpacked / "rootfs"))
    tree_sha = hashlib.sha256(_json_bytes([asdict(entry) for entry in original])).hexdigest()
    return OciFilesystem(layout, str(manifest["digest"]), str(config["digest"]), diff_id, tree_sha, original)
