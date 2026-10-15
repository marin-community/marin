# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Upload the dataset build to its private Hugging Face repository, or show what an upload would send.

`uv run python -m experiments.domain_phase_mix.release.upload [--build release_build] [--repo Calvin-Xu/mariner-data]
[--card DRAFT] [--sections DIR] [--registry CSV] [--dry-run | --confirm]`

Every run writes the dataset card into the build as README.md: a YAML header (the fields of the Olmix dataset card's
header, the language and one config per CSV table, with the size category from MANIFEST.json's row count), then the
prose of the card draft in the paper folder, whose own YAML header, if any, it drops. An upload sends exactly the
manifest's tables, MANIFEST.json, figures_tables.csv and README.md. The run stops when the build root holds any other
entry except fits/ (the inputs and outputs of the archived-swarm refits, which it skips), when the files under data/
differ from the manifest's tables in name, size or SHA-256, when the manifest records consistency problems, or when the
privacy scan of the files it would send finds anything. It then prints the file list, the build commit, the paper
audit's summary line and the size.

A dry run, the default, stops there without any network call. `--confirm` creates the repository as private, or finds
it, refuses one that is not private, and sends the files in one commit, which also deletes the repository's data/
files that the build no longer holds.
"""

import argparse
import dataclasses
import hashlib
import json
from pathlib import Path

import yaml
from huggingface_hub import CommitOperationAdd, CommitOperationDelete, HfApi

from experiments.domain_phase_mix.release import paths, scan_upload, verify_paper_numbers

REPO_ID = "Calvin-Xu/mariner-data"
CARD_DRAFT = paths.DRIVE / "revision_notes/20261014_release_dataset/README.md"
SECTIONS = paths.DRIVE / "sections"
MANIFEST = "MANIFEST.json"
CARD = "README.md"
DATA = "data"
TOP_LEVEL = (MANIFEST, "figures_tables.csv", CARD)  # sent with every file under data/
SKIPPED = ("fits",)  # regenerate_mixture_selection_data's refit inputs and outputs; never sent
CONFIG_FORMAT = "csv"  # the only table format the card lists as configs; see card_configs
# The Hub's size categories by row count: each name applies below its bound, and "n>1T" above the last.
SIZE_CATEGORIES = (
    (10**3, "n<1K"),
    (10**4, "1K<n<10K"),
    (10**5, "10K<n<100K"),
    (10**6, "100K<n<1M"),
    (10**7, "1M<n<10M"),
    (10**8, "10M<n<100M"),
    (10**9, "100M<n<1B"),
    (10**10, "1B<n<10B"),
    (10**11, "10B<n<100B"),
    (10**12, "100B<n<1T"),
)


@dataclasses.dataclass(frozen=True)
class Upload:
    """A checked build and what an upload of it sends."""

    build: Path
    manifest: dict
    files: tuple[str, ...]  # relative to the build, in the order sent
    skipped: tuple[str, ...]  # top-level entries left out
    audit: str  # verify_paper_numbers' summary line for the build


def card_configs(manifest: dict) -> str:
    """The card's YAML `configs:` block: one config per CSV table, named after it, whose train split is its file.

    `datasets` (4.8.5) reads every config of a repository with the one builder that the first config's files select,
    so a Parquet table listed beside CSV tables fails to load. The card's prose lists the Parquet tables as files.
    """
    configs = [
        {"config_name": t["table"], "data_files": [{"split": "train", "path": t["path"]}]}
        for t in manifest["tables"]
        if t["format"] == CONFIG_FORMAT
    ]
    return yaml.safe_dump({"configs": configs}, sort_keys=False)


def size_category(rows: int) -> str:
    for bound, name in SIZE_CATEGORIES:
        if rows < bound:
            return name
    return "n>1T"


def build_card(manifest: dict, draft: str) -> str:
    """The dataset card: a YAML header generated from the manifest, then the draft's prose without its own header."""
    # The fields of the Olmix dataset card's header (huggingface.co/datasets/allenai/olmix) in its order, and the
    # language.
    header = {
        "license": "apache-2.0",
        "task_categories": ["other"],
        "language": ["en"],
        "pretty_name": "MARINER Data",
        "size_categories": [size_category(sum(t["rows"] for t in manifest["tables"]))],
        "tags": ["data-mixing", "language-models", "pretraining", "mixture-optimization"],
    }
    yaml_block = yaml.safe_dump(header, sort_keys=False) + card_configs(manifest)
    return f"---\n{yaml_block}---\n\n{_prose(draft).strip()}\n"


def _prose(draft: str) -> str:
    """The draft without the YAML header it may open with; a table's `| --- |` rule opens none."""
    if not draft.startswith("---\n"):
        return draft
    end = draft.find("\n---\n", 3)
    if end < 0:
        raise ValueError("the card draft opens a YAML header that it never closes")
    return draft[end + len("\n---\n") :]


def upload_files(build: Path, manifest: dict) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """The files an upload sends, relative to `build`, and the top-level entries it skips.

    Raises when the build root holds another entry than data/, the top-level files and fits/, or when the files under
    data/ differ from the manifest's tables in name, size or SHA-256.
    """
    entries = {p.name for p in build.iterdir()}
    unknown = sorted(entries - {DATA, *TOP_LEVEL, *SKIPPED})
    if unknown:
        raise ValueError(f"{build} holds entries that an upload neither sends nor skips: {unknown}")
    tables = sorted(t["path"] for t in manifest["tables"])
    on_disk = sorted(p.relative_to(build).as_posix() for p in (build / DATA).rglob("*") if p.is_file())
    if on_disk != tables:
        unlisted, absent = sorted(set(on_disk) - set(tables)), sorted(set(tables) - set(on_disk))
        raise ValueError(f"data/ differs from the manifest: files it does not list {unlisted}, tables absent {absent}")
    changed = [t["path"] for t in manifest["tables"] if _size_and_digest(build / t["path"]) != (t["bytes"], t["sha256"])]
    if changed:
        raise ValueError(f"tables changed after the manifest recorded their size and SHA-256: {changed}")
    return (*tables, *TOP_LEVEL), tuple(sorted(entries & set(SKIPPED)))


def _size_and_digest(path: Path) -> tuple[int, str]:
    content = path.read_bytes()
    return len(content), hashlib.sha256(content).hexdigest()


def audit_summary(build: Path, sections: Path, registry: Path) -> str:
    """The summary line of `verify_paper_numbers` for the build: differences, untraced cells and registered skips."""
    differences, untraced = verify_paper_numbers.audit(build, paths.require(sections), registry)
    unresolved = sum(u.kind in (verify_paper_numbers.UNREGISTERED, "untraced") for u in untraced)
    return f"{len(differences)} differences, {unresolved} untraced cells, {len(untraced) - unresolved} registered skips"


def prepare(build: Path, card: Path, sections: Path, registry: Path) -> Upload:
    """Check the build, write its card and scan the files an upload sends; makes no network call."""
    manifest = json.loads((build / MANIFEST).read_text())
    problems = manifest["consistency_problems"]
    if problems:
        raise ValueError(f"the manifest records {len(problems)} consistency problems, the first: {problems[0]}")
    files, skipped = upload_files(build, manifest)
    (build / CARD).write_text(build_card(manifest, paths.require(card).read_text()))
    hits = scan_upload.scan(build, include=files)
    if hits:
        shown = "\n".join(scan_upload.hit_line(hit, build) for hit in hits[: scan_upload.SHOWN])
        raise ValueError(f"{len(hits)} privacy scan hits in the files an upload sends:\n{shown}")
    return Upload(build, manifest, files, skipped, audit_summary(build, sections, registry))


def publish(api: HfApi, repo: str, upload: Upload) -> None:
    """Send the upload's files to the private dataset repository `repo` in one commit.

    The commit also deletes the repository's data/ files that the upload does not send, so data/ mirrors the build.
    """
    api.create_repo(repo, repo_type="dataset", private=True, exist_ok=True)
    if api.repo_info(repo, repo_type="dataset").private is not True:
        raise ValueError(f"{repo} is not private; refusing to upload")
    stale = [
        f for f in api.list_repo_files(repo, repo_type="dataset") if f.startswith(f"{DATA}/") and f not in upload.files
    ]
    operations: list[CommitOperationAdd | CommitOperationDelete] = [
        CommitOperationAdd(path_in_repo=f, path_or_fileobj=upload.build / f) for f in upload.files
    ]
    operations += [CommitOperationDelete(path_in_repo=f) for f in sorted(stale)]
    message = f"Build from commit {upload.manifest['commit'][:12]}"
    api.create_commit(repo, operations, commit_message=message, repo_type="dataset")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--build", type=Path, default=paths.BUILD)
    parser.add_argument("--repo", default=REPO_ID)
    parser.add_argument("--card", type=Path, default=CARD_DRAFT, help="the card draft; its prose follows the header")
    parser.add_argument("--sections", type=Path, default=SECTIONS, help="the paper's sections/ folder, for the audit")
    parser.add_argument("--registry", type=Path, default=verify_paper_numbers.REGISTRY, help="the audit's registry")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="the default: check, scan and list, without network calls")
    mode.add_argument("--confirm", action="store_true", help="send the files to the private repository")
    args = parser.parse_args(argv)
    upload = prepare(args.build, args.card, args.sections, args.registry)
    sizes = {name: (args.build / name).stat().st_size for name in upload.files}
    for name, size in sizes.items():
        print(f"{size:>12,}  {name}")
    for name in upload.skipped:
        print(f"{'skipped':>12}  {name}/")
    manifest = upload.manifest
    rows = sum(t["rows"] for t in manifest["tables"])
    print(f"build: commit {manifest['commit']}, built {manifest['built_at']}")
    print(f"paper audit: {upload.audit}")
    summary = (
        f"{len(upload.files)} files ({len(manifest['tables'])} tables, {rows:,} rows), {sum(sizes.values()) / 1e6:.1f} "
        f"MB, 0 privacy scan hits, for {args.repo} (private)"
    )
    if not args.confirm:
        print(f"dry run, nothing sent: {summary}; --confirm uploads")
        return
    print(f"uploading {summary}")
    publish(HfApi(), args.repo, upload)
    print(f"uploaded to https://huggingface.co/datasets/{args.repo}")


if __name__ == "__main__":
    main()
