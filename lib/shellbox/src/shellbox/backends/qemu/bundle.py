# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Stage a QEMU guest bundle from runtime assets and an optional OCI image."""

import argparse
import gzip
import hashlib
import importlib.resources
import json
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path, PurePosixPath


def _guest_file(name: str):
    return importlib.resources.files("shellbox.backends.qemu").joinpath(f"guest/{name}")


def guest_code_id() -> str:
    """Identify guest-side code included in a staged bundle."""
    digest = hashlib.sha256()
    for name in ("disk-init", "init", "pty-agent.c"):
        digest.update(_guest_file(name).read_bytes())
    return digest.hexdigest()


def _compile_pty_agent(output: Path) -> None:
    with importlib.resources.as_file(_guest_file("pty-agent.c")) as source:
        subprocess.run(
            ["cc", "-static", "-Os", "-s", "-Wall", "-Wextra", "-o", str(output), str(source), "-lutil"],
            check=True,
            capture_output=True,
        )


def _initramfs(root: Path, output: Path) -> None:
    paths = [str(path.relative_to(root)) for path in root.rglob("*")]
    cpio = subprocess.run(
        ["cpio", "-o", "-H", "newc"],
        input="\n".join(paths).encode() + b"\n",
        cwd=root,
        capture_output=True,
        check=True,
    )
    output.write_bytes(gzip.compress(cpio.stdout, compresslevel=1))


def _boot_initramfs(busybox: Path, output: Path) -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        (root / "bin").mkdir()
        shutil.copy2(busybox, root / "bin/busybox")
        shutil.copy2(_guest_file("disk-init"), root / "init")
        (root / "init").chmod(0o755)
        for name in ("dev", "proc", "sys"):
            (root / name).mkdir()
        _initramfs(root, output)


def _layer_owners(oci_layout: Path, layers: list[dict]) -> dict[str, tuple[int, int]]:
    owners: dict[str, tuple[int, int]] = {}
    for layer in layers:
        if "zstd" in layer["mediaType"]:
            raise ValueError("Zstandard OCI layers are unsupported by this staging tool")
        algorithm, layer_hash = layer["digest"].split(":", 1)
        additions: dict[str, tuple[int, int]] = {}
        removals: list[str] = []
        with tarfile.open(oci_layout / "blobs" / algorithm / layer_hash, "r:*") as archive:
            for member in archive:
                path = PurePosixPath(member.name)
                if path.is_absolute() or ".." in path.parts:
                    raise ValueError(f"Invalid OCI layer path: {member.name}")
                name = path.name
                if name == ".wh..wh..opq":
                    removals.append("" if str(path.parent) == "." else f"{path.parent}/")
                elif name.startswith(".wh."):
                    removals.append(str(path.parent / name.removeprefix(".wh.")))
                else:
                    additions[str(path)] = (member.uid, member.gid)
        for removed in removals:
            if not removed:
                owners.clear()
                continue
            owners = {
                path: owner
                for path, owner in owners.items()
                if path != removed and not path.startswith(removed.rstrip("/") + "/")
            }
        owners.update(additions)
    return owners


def _rootfs_tar(root: Path, owners: dict[str, tuple[int, int]], output: Path) -> None:
    with tarfile.open(output, "w", format=tarfile.PAX_FORMAT) as archive:
        for path in root.rglob("*"):
            name = str(path.relative_to(root))
            info = archive.gettarinfo(str(path), arcname=name)
            info.uid, info.gid = owners.get(name, (0, 0))
            info.uname = ""
            info.gname = ""
            if info.isfile():
                with path.open("rb") as stream:
                    archive.addfile(info, stream)
            else:
                archive.addfile(info)


def _oci_disk(
    oci_layout: Path,
    oci_tag: str,
    umoci: Path,
    busybox: Path,
    disk_size_mb: int,
    output: Path,
    task_dockerfile: Path | None,
    image_reference: str | None,
) -> None:
    with tempfile.TemporaryDirectory() as temporary:
        bundle = Path(temporary) / "unpacked"
        subprocess.run(
            [str(umoci), "unpack", "--rootless", "--image", f"{oci_layout}:{oci_tag}", str(bundle)],
            check=True,
        )
        config = json.loads((bundle / "config.json").read_text())
        process = config["process"]
        if process["user"]["uid"] != 0 or process["user"]["gid"] != 0:
            raise ValueError("OCI images with a non-root default user are unsupported")
        descriptor = json.loads((bundle / "umoci.json").read_text())["from_descriptor_path"]["descriptor_walk"][-1]
        algorithm, manifest_hash = descriptor["digest"].split(":", 1)
        manifest = json.loads((oci_layout / "blobs" / algorithm / manifest_hash).read_text())
        config_algorithm, config_hash = manifest["config"]["digest"].split(":", 1)
        image_config = json.loads((oci_layout / "blobs" / config_algorithm / config_hash).read_text())
        if image_config["os"] != "linux" or image_config["architecture"] != "amd64":
            raise ValueError("QEMU guest requires a Linux amd64 OCI image")
        owners = _layer_owners(oci_layout, manifest["layers"])
        root = bundle / "rootfs"
        if (root / "harbor").exists() or not (root / "bin/sh").exists():
            raise ValueError("OCI image must have /bin/sh and must not contain /harbor")
        (root / "harbor").mkdir()
        shutil.copy2(busybox, root / "harbor/busybox")
        shutil.copy2(_guest_file("init"), root / "harbor/init")
        _compile_pty_agent(root / "harbor/pty-agent")
        (root / "harbor/init").chmod(0o755)
        for name in ("workspace", "logs", "tmp", "dev", "proc", "sys"):
            (root / name).mkdir(exist_ok=True)
        disk = output / "rootfs.ext4"
        with disk.open("wb") as stream:
            stream.truncate(disk_size_mb * 1024 * 1024)
        rootfs_tar = Path(temporary) / "rootfs.tar"
        _rootfs_tar(root, owners, rootfs_tar)
        subprocess.run(["mkfs.ext4", "-q", "-F", "-d", str(rootfs_tar), str(disk)], check=True, capture_output=True)
        metadata = {
            "env": process.get("env", []),
            "cwd": process.get("cwd", "/"),
            "oci_tag": oci_tag,
            "manifest_digest": descriptor["digest"],
        }
        if task_dockerfile is not None:
            metadata["dockerfile_sha256"] = hashlib.sha256(task_dockerfile.read_bytes()).hexdigest()
        if image_reference is not None:
            metadata["image_reference"] = image_reference
        (output / "image.json").write_text(json.dumps(metadata, indent=2) + "\n")
        _boot_initramfs(busybox, output / "initramfs.cpio.gz")


def _minimal_initramfs(busybox: Path, output: Path) -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        (root / "bin").mkdir()
        shutil.copy2(busybox, root / "bin/busybox")
        applets = subprocess.run(
            [str(busybox), "--list"], capture_output=True, text=True, check=True
        ).stdout.splitlines()
        for applet in applets:
            if applet != "busybox":
                (root / "bin" / applet).symlink_to("busybox")
        (root / "harbor").mkdir()
        shutil.copy2(busybox, root / "harbor/busybox")
        shutil.copy2(_guest_file("init"), root / "init")
        _compile_pty_agent(root / "harbor/pty-agent")
        (root / "init").chmod(0o755)
        for name in ("dev", "proc", "sys", "workspace", "tmp", "logs"):
            (root / name).mkdir()
        _initramfs(root, output)


def stage_bundle(
    qemu: Path,
    kernel: Path,
    busybox: Path,
    firmware: Path,
    libraries: Path,
    output: Path,
    oci_layout: Path | None = None,
    oci_tag: str | None = None,
    umoci: Path | None = None,
    disk_size_mb: int | None = None,
    task_dockerfile: Path | None = None,
    image_reference: str | None = None,
) -> None:
    """Stage QEMU plus a minimal guest or one unpacked OCI image."""
    if not (firmware / "bios-microvm.bin").is_file():
        raise FileNotFoundError(firmware / "bios-microvm.bin")
    output.mkdir(parents=True, exist_ok=True)
    shutil.copy2(qemu, output / "qemu-system-x86_64")
    shutil.copy2(kernel, output / "vmlinuz")
    shutil.copytree(firmware, output / "firmware", dirs_exist_ok=True)
    shutil.copytree(libraries, output / "lib", dirs_exist_ok=True)
    if oci_layout is None:
        if any(value is not None for value in (oci_tag, umoci, disk_size_mb, task_dockerfile, image_reference)):
            raise ValueError("OCI options require --oci-layout")
        _minimal_initramfs(busybox, output / "initramfs.cpio.gz")
        return
    if oci_tag is None or umoci is None or disk_size_mb is None:
        raise ValueError("OCI staging requires --oci-tag, --umoci, and --disk-size-mb")
    _oci_disk(oci_layout, oci_tag, umoci, busybox, disk_size_mb, output, task_dockerfile, image_reference)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("qemu", "kernel", "busybox", "firmware", "libraries", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--oci-layout", type=Path)
    parser.add_argument("--oci-tag")
    parser.add_argument("--umoci", type=Path)
    parser.add_argument("--disk-size-mb", type=int)
    parser.add_argument("--task-dockerfile", type=Path)
    parser.add_argument("--image-reference")
    args = parser.parse_args()
    stage_bundle(
        args.qemu,
        args.kernel,
        args.busybox,
        args.firmware,
        args.libraries,
        args.output,
        args.oci_layout,
        args.oci_tag,
        args.umoci,
        args.disk_size_mb,
        args.task_dockerfile,
        args.image_reference,
    )


if __name__ == "__main__":
    main()
