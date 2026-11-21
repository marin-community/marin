# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Download the pinned xprof-rs binary before bundling the Iris service."""

import hashlib
import io
import lzma
import platform
import tarfile
import tempfile
import urllib.request
from pathlib import Path

from infra.xprof.release import XPROF_RS_ARCH, XPROF_RS_COMPRESSED_PATH, XPROF_RS_SHA256, XPROF_RS_VERSION


def main() -> None:
    if platform.system() != "Linux" or platform.machine() != XPROF_RS_ARCH:
        raise RuntimeError(f"The pinned xprof-rs release requires {XPROF_RS_ARCH} Linux")

    archive_name = f"xprof-rs-{XPROF_RS_VERSION}-{XPROF_RS_ARCH}-linux.tar.gz"
    url = f"https://github.com/Locamage/xprof-rs/releases/download/{XPROF_RS_VERSION}/{archive_name}"
    with urllib.request.urlopen(url, timeout=60) as response:
        archive = response.read()
    digest = hashlib.sha256(archive).hexdigest()
    if digest != XPROF_RS_SHA256:
        raise ValueError(f"xprof-rs archive SHA-256 mismatch: {digest}")

    member_name = f"{archive_name.removesuffix('.tar.gz')}/xprof-rs"
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as package:
        member = package.getmember(member_name)
        if not member.isfile():
            raise ValueError(f"xprof-rs release member is not a file: {member_name}")
        source = package.extractfile(member)
        if source is None:
            raise ValueError(f"xprof-rs release member is empty: {member_name}")
        destination = Path(XPROF_RS_COMPRESSED_PATH)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as temporary:
            temporary.write(lzma.compress(source.read(), preset=6))
            temporary_path = Path(temporary.name)
        temporary_path.replace(destination)


if __name__ == "__main__":
    main()
