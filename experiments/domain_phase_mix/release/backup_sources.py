# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Back up the dataset export's local inputs to a private Hugging Face dataset, or show what a backup would send.

`uv run python -m experiments.domain_phase_mix.release.backup_sources [--repo Calvin-Xu/mariner-data-sources]
[--dry-run | --confirm]`

The export records every file it read in SOURCES.json (under paths.PRIVATE). Git pins a source when this repository
tracks it at the build's commit with the same content. The other sources exist only on the machine that ran the
export: the locally excluded reference_outputs/ and .agents/ files, and the mixture-selection checkout, which has no
remote. The backup holds those files under their repository's directory name and relative path (marin/...,
mixture-selection/...), with MANIFEST.csv listing every source's size, SHA-256, whether git pins it, and the tables
that read it.

The backup is for us, not part of the release: readers work from the public dataset alone. A dry run, the default,
makes no network call. `--confirm` creates the repository as private, or finds it, refuses one that is not private,
and sends the files in one commit, which also deletes the files that the build no longer reads.
"""

import argparse
import csv
import dataclasses
import hashlib
import io
import json
import subprocess
from pathlib import Path

from huggingface_hub import CommitOperationAdd, CommitOperationDelete, HfApi

from experiments.domain_phase_mix.release import paths
from experiments.domain_phase_mix.release.export_dataset import SOURCES

REPO_ID = "Calvin-Xu/mariner-data-sources"
MANIFEST = "MANIFEST.csv"
CARD = "README.md"
CONTEXT = "(context)"  # the tables column of a file that the export reads for every table
CARD_TEXT = """# MARINER dataset export sources (private)

The local files that `experiments/domain_phase_mix/release/export_dataset.py` (Marin branch
`calvin/swarm-olmo3-regmix-test`) read for the dataset build at commit {commit}, built {built_at}: {backed} of its
{total} sources, {size:.1f} MB. Files under `marin/` are untracked research outputs of that repository; files under
`mixture-selection/` come from a local repository without a remote. `MANIFEST.csv` lists all {total} sources with
their size and SHA-256. The {pinned} that git tracks at the commit with the same content (`pinned_by_git`) are not
copied here.

This is a backup of internal intermediates, not part of the paper's release, which is the public dataset alone.
Do not make this repository public.
"""


@dataclasses.dataclass(frozen=True)
class Source:
    """One file the export read: its place in the backup, its content, and the tables that read it."""

    path: Path
    backup_path: str
    size: int
    sha256: str
    pinned_by_git: bool
    tables: tuple[str, ...]


def git(root: Path, *args: str, stdin: str | None = None) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *args], input=stdin, capture_output=True, text=True, check=True
    ).stdout


def readers(record: dict) -> dict[Path, list[str]]:
    """Each source's absolute path and the tables that read it, from the export's SOURCES.json record."""
    tables: dict[Path, list[str]] = {Path(p): [CONTEXT] for p in record["context"]}
    for table, files in record["tables"].items():
        for p in files:
            tables.setdefault(Path(p), []).append(table)
    return tables


def git_pinned(files: list[Path], commit: str) -> set[Path]:
    """The files of this repository that git tracks at `commit` with the content they have on disk."""
    relative = {f: f.resolve().relative_to(paths.REPO) for f in files}
    tracked = {}
    for entry in git(paths.REPO, "ls-tree", "-r", "-z", commit).split("\0")[:-1]:
        meta, name = entry.split("\t", 1)
        tracked[name] = meta.split()[2]
    on_disk = git(paths.REPO, "hash-object", "--stdin-paths", stdin="\n".join(str(r) for r in relative.values()))
    return {f for f, blob in zip(files, on_disk.split(), strict=True) if tracked.get(str(relative[f])) == blob}


def backup_path(path: Path) -> str:
    """`<repository directory>/<path in the repository>`; raises for a file outside every git repository."""
    root = Path(git(path.parent, "rev-parse", "--show-toplevel").strip())
    return f"{root.name}/{path.resolve().relative_to(root)}"


def sources(record: dict) -> list[Source]:
    tables = readers(record)
    for path in tables:
        paths.require(path)
    in_repo = [p for p in tables if p.resolve().is_relative_to(paths.REPO)]
    pinned = git_pinned(in_repo, record["commit"])
    return [
        Source(
            path=path,
            backup_path=backup_path(path),
            size=path.stat().st_size,
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            pinned_by_git=path in pinned,
            tables=tuple(sorted(tables[path])),
        )
        for path in sorted(tables)
    ]


def manifest(files: list[Source]) -> bytes:
    out = io.StringIO()
    writer = csv.writer(out, lineterminator="\n")
    writer.writerow(["path", "bytes", "sha256", "pinned_by_git", "tables"])
    for f in files:
        writer.writerow([f.backup_path, f.size, f.sha256, f.pinned_by_git, ";".join(f.tables)])
    return out.getvalue().encode()


def card(record: dict, files: list[Source]) -> bytes:
    backed = [f for f in files if not f.pinned_by_git]
    text = CARD_TEXT.format(
        commit=record["commit"][:12],
        built_at=record["built_at"],
        backed=len(backed),
        total=len(files),
        size=sum(f.size for f in backed) / 1e6,
        pinned=len(files) - len(backed),
    )
    return text.encode()


def publish(api: HfApi, repo: str, record: dict, files: list[Source]) -> str:
    """Send the unpinned sources, the manifest and the card to the private repository `repo` in one commit, deleting
    the files that the build no longer reads; return the commit's id."""
    api.create_repo(repo, repo_type="dataset", private=True, exist_ok=True)
    if api.repo_info(repo, repo_type="dataset").private is not True:
        raise ValueError(f"{repo} is not private; refusing to upload")
    backed = {f.backup_path: f.path for f in files if not f.pinned_by_git}
    stale = [
        f for f in api.list_repo_files(repo, repo_type="dataset") if f not in {*backed, MANIFEST, CARD, ".gitattributes"}
    ]
    operations: list[CommitOperationAdd | CommitOperationDelete] = [
        CommitOperationAdd(path_in_repo=name, path_or_fileobj=path) for name, path in sorted(backed.items())
    ]
    operations += [
        CommitOperationAdd(path_in_repo=MANIFEST, path_or_fileobj=manifest(files)),
        CommitOperationAdd(path_in_repo=CARD, path_or_fileobj=card(record, files)),
    ]
    operations += [CommitOperationDelete(path_in_repo=f) for f in sorted(stale)]
    message = f"Sources of the build from commit {record['commit'][:12]}"
    return api.create_commit(repo, operations, commit_message=message, repo_type="dataset").oid


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repo", default=REPO_ID)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="the default: list the backup without network calls")
    mode.add_argument("--confirm", action="store_true", help="send the files to the private repository")
    args = parser.parse_args(argv)
    record = json.loads((paths.PRIVATE / SOURCES).read_text())
    files = sources(record)
    backed = [f for f in files if not f.pinned_by_git]
    by_repository: dict[str, list[Source]] = {}
    for f in backed:
        by_repository.setdefault(f.backup_path.split("/", 1)[0], []).append(f)
    print(f"build: commit {record['commit']}, built {record['built_at']}")
    print(f"{len(files)} sources, {len(files) - len(backed)} pinned by git, {len(backed)} to back up:")
    for repository, group in sorted(by_repository.items()):
        print(f"  {repository}/: {len(group)} files, {sum(f.size for f in group) / 1e6:.1f} MB")
    if not args.confirm:
        print(f"dry run, nothing sent; --confirm uploads to {args.repo} (private)")
        return
    oid = publish(HfApi(), args.repo, record, files)
    print(f"uploaded to https://huggingface.co/datasets/{args.repo} at commit {oid}")


if __name__ == "__main__":
    main()
