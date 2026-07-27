# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Stage object-store Harbor task directories at Harbor's local-path boundary."""

from pathlib import Path

from rigging.filesystem import StoragePath, TreeTransferMode, copy_tree


def materialize_harbor_dataset(
    dataset: str,
    workdir: Path,
    *,
    task_limit: int | None,
) -> Path | None:
    """Return a local Harbor task directory, or ``None`` for a registry-backed dataset.

    Existing local directories are returned unchanged. An fsspec URI is treated as
    a tree of Harbor tasks and only the selected task directories are copied locally.
    Bare values remain Harbor registry names.
    """
    if dataset.startswith("hf://"):
        raise ValueError(
            "Harbor datasets may not be read directly from Hugging Face; "
            "resolve the immutable repository revision as an evaluator artifact first"
        )

    dataset_path = Path(dataset).expanduser()
    source = StoragePath(dataset)
    if source.is_remote:
        task_directories = sorted(
            (
                child
                for child in source.ls()
                if child.isdir() and (child / "task.toml").isfile() and (child / "environment").isdir()
            ),
            key=lambda path: path.name,
        )
        selected = task_directories[:task_limit] if task_limit is not None else task_directories
        if not selected:
            raise ValueError(f"no Harbor task directories found at {dataset}")
        local_root = workdir / "harbor_dataset"
        local_root.mkdir(parents=True, exist_ok=True)
        for task_directory in selected:
            copy_tree(
                task_directory,
                StoragePath(str(local_root / task_directory.name)),
                mode=TreeTransferMode.RESUME,
            )
        return local_root

    if not dataset_path.exists():
        return None
    if not dataset_path.is_dir():
        raise ValueError(f"Harbor dataset path must be a directory, got: {dataset_path}")
    return dataset_path
