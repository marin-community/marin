# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import fsspec
import pytest

from marin.utils import fsspec_copy_path_into_dir, fsspec_copyfile_between_fs


@pytest.fixture
def memfs():
    fs = fsspec.filesystem("memory")
    fs.store.clear()
    yield fs
    fs.store.clear()


def test_fsspec_copyfile_between_fs_memory_to_local(memfs, tmp_path):
    memfs.makedirs("/src", exist_ok=True)
    with memfs.open("/src/file.txt", "wb") as f:
        f.write(b"hello")

    local_fs = fsspec.filesystem("file")
    dst = tmp_path / "out.txt"
    fsspec_copyfile_between_fs(fs_in=memfs, src="/src/file.txt", fs_out=local_fs, dst=str(dst))
    assert dst.read_bytes() == b"hello"


def test_fsspec_copy_path_into_dir_file_memory_to_local_autofs(memfs, tmp_path):
    memfs.makedirs("/src", exist_ok=True)
    with memfs.open("/src/file.txt", "wb") as f:
        f.write(b"hello")

    dst_dir = tmp_path / "dst"
    fsspec_copy_path_into_dir(src_path="memory://src/file.txt", dst_path=str(dst_dir))
    assert (dst_dir / "file.txt").read_bytes() == b"hello"


def test_fsspec_copy_path_into_dir_dir_memory_to_local_with_fs_in(memfs, tmp_path):
    memfs.makedirs("/srcdir/sub", exist_ok=True)
    with memfs.open("/srcdir/a.txt", "wb") as f:
        f.write(b"a")
    with memfs.open("/srcdir/sub/b.txt", "wb") as f:
        f.write(b"b")

    dst_dir = tmp_path / "dst"
    fsspec_copy_path_into_dir(src_path="memory://srcdir", dst_path=str(dst_dir), fs_in=memfs)
    assert (dst_dir / "a.txt").read_bytes() == b"a"
    assert (dst_dir / "sub" / "b.txt").read_bytes() == b"b"


def test_fsspec_copy_path_into_dir_dir_local_to_memory_autofs(memfs, tmp_path):
    src_dir = tmp_path / "src"
    (src_dir / "sub").mkdir(parents=True)
    (src_dir / "a.txt").write_bytes(b"a")
    (src_dir / "sub" / "b.txt").write_bytes(b"b")

    fsspec_copy_path_into_dir(src_path=str(src_dir), dst_path="memory://dst")
    assert memfs.cat("/dst/a.txt") == b"a"
    assert memfs.cat("/dst/sub/b.txt") == b"b"


def test_fsspec_copy_path_into_dir_file_local_to_memory_with_fs_out(memfs, tmp_path):
    src_file = tmp_path / "file.bin"
    src_file.write_bytes(b"\x00\x01\x02")

    fsspec_copy_path_into_dir(src_path=str(src_file), dst_path="memory://dst2", fs_out=memfs)
    assert memfs.cat("/dst2/file.bin") == b"\x00\x01\x02"


def test_fsspec_copy_path_into_dir_file_local_to_local(tmp_path):
    src_file = tmp_path / "src.bin"
    src_file.write_bytes(b"\x03\x04")

    dst_dir = tmp_path / "dst"
    fsspec_copy_path_into_dir(src_path=str(src_file), dst_path=str(dst_dir))
    assert (dst_dir / "src.bin").read_bytes() == b"\x03\x04"


def test_fsspec_copy_path_into_dir_dir_local_to_local(tmp_path):
    src_dir = tmp_path / "srcdir"
    (src_dir / "sub").mkdir(parents=True)
    (src_dir / "a.txt").write_bytes(b"a")
    (src_dir / "sub" / "b.txt").write_bytes(b"b")

    dst_dir = tmp_path / "dst"
    fsspec_copy_path_into_dir(src_path=str(src_dir), dst_path=str(dst_dir))
    assert (dst_dir / "a.txt").read_bytes() == b"a"
    assert (dst_dir / "sub" / "b.txt").read_bytes() == b"b"
