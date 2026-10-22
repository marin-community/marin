from asyncio import to_thread
import errno
from pathlib import Path
import shutil
import tempfile

from huggingface_hub import snapshot_download


def _materialize_snapshot(snapshot: Path) -> Path:
    """Copy a Hub snapshot into a directory that contains no cache symlinks."""
    destination = snapshot.parent.parent / "harbor-materialized" / snapshot.name
    if destination.is_dir():
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent) as staging_root:
        staged = Path(staging_root) / snapshot.name
        shutil.copytree(snapshot, staged)
        try:
            staged.rename(destination)
        except OSError as error:
            concurrent_copy_completed = (
                error.errno in {errno.EEXIST, errno.ENOTEMPTY} and destination.is_dir()
            )
            if not concurrent_copy_completed:
                raise

    return destination


def _download_dataset_snapshot(
    repo_id: str, revision: str | None, cache_dir: Path | None, overwrite: bool
) -> Path:
    snapshot = Path(
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            revision=revision,
            cache_dir=cache_dir,
            force_download=overwrite,
        )
    )
    return _materialize_snapshot(snapshot)


async def download_dataset_snapshot(
    repo_id: str, revision: str | None, cache_dir: Path | None, overwrite: bool
) -> Path:
    """Download and materialize a Hugging Face dataset repository."""
    return await to_thread(
        _download_dataset_snapshot,
        repo_id,
        revision,
        cache_dir,
        overwrite,
    )
