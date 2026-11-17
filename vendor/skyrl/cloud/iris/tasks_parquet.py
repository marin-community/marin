"""Extract Harbor task directories from a parquet dataset (one row per task).

Parquet schema:
- ``path``: str (relative path from the base dir to the task dir)
- ``task_binary``: binary (tar archive bytes)

Only the extraction (parquet -> directory) path is included here; task detection
relies on the presence of an ``instruction.md`` inside each task directory.
"""

from __future__ import annotations

import io
import os
import shutil
import tarfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path, PurePosixPath
from typing import Sequence

from tqdm import tqdm

# Optional heavy import; a clear error is raised if it is missing when used.
try:
    import pyarrow.parquet as pq
except ImportError:  # pragma: no cover - import-time availability varies
    pq = None  # type: ignore[assignment]


TASK_MARKER = "instruction.md"


def _require_pyarrow() -> object:
    if pq is None:  # type: ignore[truthy-bool]
        raise RuntimeError("pyarrow is required. Please install it: pip install pyarrow")
    return pq  # type: ignore[return-value]


def find_tasks(
    base: Path,
    recursive: bool = False,
    depth: int | None = None,
    marker: str | Sequence[str] = TASK_MARKER,
) -> list[Path]:
    """Find task directories containing the marker file.

    - recursive: walk the entire subtree and collect any directory containing the
      marker; do not descend further under a detected task dir.
    - depth: fixed search depth relative to base (1 = direct children). Ignored
      when ``recursive`` is True.
    """
    if isinstance(marker, str):
        markers = (marker,)
    else:
        markers = tuple(marker)

    def has_marker(path: Path) -> bool:
        return any((path / m).is_file() for m in markers)

    if recursive:
        out: list[Path] = []
        for root, dirs, _files in os.walk(base):
            root_path = Path(root)
            if has_marker(root_path):
                out.append(root_path)
                dirs[:] = []  # do not search within a detected task dir
        return sorted(out)

    if depth is None:
        depth = 1
    if depth < 0:
        raise ValueError("depth must be >= 0")

    current_level: list[Path] = [base]
    for _ in range(depth):
        next_level: list[Path] = []
        for p in current_level:
            if not p.is_dir():
                continue
            for child in p.iterdir():
                if child.is_dir():
                    next_level.append(child)
        current_level = next_level

    return sorted(p for p in current_level if has_marker(p))


def _is_within(base: Path, target: Path) -> bool:
    try:
        return os.path.commonpath([str(base.resolve()), str(target.resolve())]) == str(base.resolve())
    except Exception:
        return False


def _sanitize_tar_member_name(name: str) -> str:
    # Remove leading slashes, collapse to posix, strip ".." components.
    p = PurePosixPath(name)
    parts = [part for part in p.parts if part not in ("..", ".", "")]
    while parts and parts[0] == "/":
        parts.pop(0)
    return str(PurePosixPath(*parts))


def safe_extract_tar(archive_bytes: bytes, dest_dir: Path) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    buf = io.BytesIO(archive_bytes)
    with tarfile.open(fileobj=buf, mode="r:*") as tf:
        for member in tf.getmembers():
            member_name = _sanitize_tar_member_name(member.name)
            if not member_name or member_name.endswith("/"):
                (dest_dir / member_name).mkdir(parents=True, exist_ok=True)
                continue
            if ".snapshot" in PurePosixPath(member_name).parts:
                # Skip snapshot metadata that can be read-only on shared filesystems.
                continue
            target = (dest_dir / member_name).resolve()
            if not _is_within(dest_dir, target):
                raise RuntimeError(f"Unsafe path in archive: {member.name}")
            target.parent.mkdir(parents=True, exist_ok=True)
            # Extract regular files only; skip devices/symlinks for safety.
            if member.isfile():
                with tf.extractfile(member) as src:  # type: ignore[assignment]
                    if src is None:
                        continue
                    with open(target, "wb") as dst:
                        dst.write(src.read())
            elif member.isdir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                continue


def _process_parquet_row(args: tuple[int, str, bytes, Path, str]) -> Path | None:
    """Process a single parquet row in a worker process.

    Returns the target_dir if extracted, or None if skipped. Raises on error to
    propagate back to the main process.
    """
    i, rel_path, data, base, on_exist = args

    if not isinstance(rel_path, str):
        raise RuntimeError(f"Row {i}: 'path' must be a string")
    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise RuntimeError(f"Row {i}: 'task_binary' must be bytes")

    safe_rel = PurePosixPath(rel_path)
    parts = [p for p in safe_rel.parts if p not in ("..", "")]
    rel_norm = Path(*parts)
    target_dir = (base / rel_norm).resolve()
    if not _is_within(base, target_dir):
        raise RuntimeError(f"Unsafe target path: {rel_path}")

    if target_dir.exists():
        if on_exist == "skip":
            return None
        if on_exist == "error":
            raise FileExistsError(f"Target exists: {target_dir}")
        if on_exist == "overwrite":
            if target_dir.is_dir():
                shutil.rmtree(target_dir)
            else:
                target_dir.unlink()
        else:
            raise ValueError("on_exist must be one of: skip, overwrite, error")

    safe_extract_tar(bytes(data), target_dir)
    return target_dir


def from_parquet(
    parquet_path: str,
    base: str,
    on_exist: str = "error",
    max_workers: int = 10,
    batch_size: int = 256,
) -> list[Path]:
    """Extract tasks from a parquet file to a directory, in parallel batches.

    Reads the parquet in row-group batches to avoid loading the entire file (which
    can contain large binary blobs) into memory at once.
    """
    pq_mod = _require_pyarrow()

    pf = pq_mod.ParquetFile(parquet_path)
    total_rows = pf.metadata.num_rows

    schema_names = [f.name for f in pf.schema_arrow]
    if "path" not in schema_names or "task_binary" not in schema_names:
        raise RuntimeError("Parquet must have columns: 'path', 'task_binary'")

    base_path = Path(base).resolve()
    written: list[Path] = []
    row_offset = 0

    with tqdm(total=total_rows, desc="Extracting tasks") as pbar:
        for batch in pf.iter_batches(batch_size=batch_size, columns=["path", "task_binary"]):
            path_col = batch.column("path").to_pylist()
            data_col = batch.column("task_binary").to_pylist()

            tasks_args = [
                (row_offset + i, rel_path, data, base_path, on_exist)
                for i, (rel_path, data) in enumerate(zip(path_col, data_col))
            ]

            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(_process_parquet_row, args) for args in tasks_args]
                for future in as_completed(futures):
                    result = future.result()
                    if result is not None:
                        written.append(result)
                    pbar.update(1)

            row_offset += len(path_col)
            del path_col, data_col, tasks_args

    return written
