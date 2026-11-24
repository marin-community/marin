# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Read digest-pinned files from the published ScienceAgentBench release."""

import hashlib
import http.cookiejar
import io
import subprocess
import urllib.request
import zipfile
from pathlib import Path
from typing import BinaryIO

ARCHIVE_URL = (
    "https://buckeyemailosu-my.sharepoint.com/:u:/g/personal/chen_8336_osu_edu/"
    "IQB870QrmuqwS5Ck33cHpJfkAUGOoyYcEbxZm7oyarjcHAY?download=1"
)
ARCHIVE_SIZE = 1769478786
ARCHIVE_SHA256 = "46e715d3b2196d459d2dff52aa487f506a95ec44b44262e82208d086ea879610"
READ_AHEAD = 512 * 1024


class Archive(io.RawIOBase):
    """Read only the byte ranges requested by ZipFile from the publisher's release."""

    def __init__(self):
        self.position = 0
        self.block_start = 0
        self.block = b""
        self.opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar()))

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        self.position = (0, self.position, ARCHIVE_SIZE)[whence] + offset
        if self.position < 0:
            raise OSError("Invalid archive offset")
        return self.position

    def read(self, size=-1):
        remaining = max(0, ARCHIVE_SIZE - self.position)
        size = remaining if size < 0 else min(size, remaining)
        if not size:
            return b""
        if size > 4 * 1024 * 1024:
            raise ValueError("Unexpectedly large ZIP directory or compressed member read")
        offset = self.position - self.block_start
        if offset >= 0 and offset + size <= len(self.block):
            self.position += size
            return self.block[offset : offset + size]
        length = min(remaining, max(size, READ_AHEAD))
        end = self.position + length - 1
        request = urllib.request.Request(ARCHIVE_URL, headers={"Range": f"bytes={self.position}-{end}"})
        with self.opener.open(request, timeout=60) as response:
            if (
                response.status != 206
                or response.headers["Content-Range"] != f"bytes {self.position}-{end}/{ARCHIVE_SIZE}"
            ):
                raise ValueError("Publisher did not return the requested archive range")
            block = response.read(length + 1)
        if len(block) != length:
            raise OSError("Incomplete archive range")
        self.block_start, self.block = self.position, block
        self.position += size
        return block[:size]


def copy_member(archive: zipfile.ZipFile, member: str, digest: str, output: BinaryIO, size: int) -> None:
    """Stream one pinned archive member, checking its length and digest."""
    info = archive.getinfo(member)
    if info.is_dir() or info.file_size != size:
        raise ValueError(f"Unexpected member size for {member}")
    checksum = hashlib.sha256()
    with archive.open(info, pwd=b"scienceagentbench") as source:
        while chunk := source.read(1024 * 1024):
            checksum.update(chunk)
            output.write(chunk)
    if checksum.hexdigest() != digest:
        raise ValueError(f"Unexpected contents for {member}")


def download_release(path: Path) -> None:
    """Verify the entire release before using it to stage large inputs."""
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar()))
    digest = hashlib.sha256()
    size = 0
    with opener.open(ARCHIVE_URL, timeout=120) as response, path.open("wb") as output:
        while block := response.read(1024 * 1024):
            size += len(block)
            if size > ARCHIVE_SIZE:
                raise ValueError("Publisher archive exceeds pinned size")
            digest.update(block)
            output.write(block)
    if size != ARCHIVE_SIZE or digest.hexdigest() != ARCHIVE_SHA256:
        raise ValueError("Publisher archive integrity mismatch")


def copy_local_member(path: Path, member: str, digest: str, output: BinaryIO, size: int) -> None:
    """Use native ZIP decryption for large files, verifying before sandbox upload."""
    if any(ch in member for ch in "*?[]\\"):
        with zipfile.ZipFile(path) as archive:
            copy_member(archive, member, digest, output, size)
        return
    checksum = hashlib.sha256()
    copied = 0
    with subprocess.Popen(["unzip", "-p", "-P", "scienceagentbench", str(path), member], stdout=subprocess.PIPE) as proc:
        assert proc.stdout is not None
        with proc.stdout as stream:
            while chunk := stream.read(1024 * 1024):
                copied += len(chunk)
                if copied > size:
                    raise ValueError(f"Unexpected member size for {member}")
                checksum.update(chunk)
                output.write(chunk)
        if proc.wait() or copied != size or checksum.hexdigest() != digest:
            raise ValueError(f"Unexpected contents for {member}")
