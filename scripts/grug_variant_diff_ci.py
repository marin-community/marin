# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Generate Grug variant comparison reports for newly added variants in a PR."""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from scripts.grug_dir_diff import (
    DEFAULT_EXTENSIONS,
    build_directory_diff_report,
    collect_files,
    line_change_counts,
    parse_extensions,
    read_text_lines,
)

GRUG_ROOT = Path("experiments/grug")
IGNORED_VARIANTS = frozenset({"__pycache__"})


@dataclass(frozen=True)
class VariantMatch:
    """Best-match result for one newly added variant directory."""

    variant: str
    closest_variant: str
    distance_score: int


def list_variants_at_ref(
    *,
    ref: str,
    grug_root: Path = GRUG_ROOT,
    strict: bool = True,
) -> set[str]:
    """Return direct subdirectory names under ``grug_root`` for a git ref."""
    tree_spec = f"{ref}:{grug_root.as_posix()}"

    completed = subprocess.run(
        ["git", "ls-tree", tree_spec],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        if strict:
            error = completed.stderr.strip() or f"git ls-tree failed for {tree_spec}"
            raise RuntimeError(error)
        return set()

    variants: set[str] = set()
    for line in completed.stdout.splitlines():
        # Format: <mode> <type> <sha>\t<name>
        metadata, _, name = line.partition("\t")
        parts = metadata.split()
        if len(parts) < 2:
            continue
        object_type = parts[1]
        if object_type != "tree":
            continue
        if name == "base" or name in IGNORED_VARIANTS:
            continue
        if name.startswith("."):
            continue
        variants.add(name)

    return variants


def materialize_grug_tree_at_ref(*, ref: str, destination_root: Path, grug_root: Path = GRUG_ROOT) -> Path:
    """Export ``grug_root`` from ``ref`` into ``destination_root`` and return the extracted path."""
    destination_root.mkdir(parents=True, exist_ok=True)
    archive_process = subprocess.Popen(
        ["git", "archive", "--format=tar", ref, grug_root.as_posix()],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if archive_process.stdout is None:
        raise RuntimeError("git archive did not produce stdout")

    untar_process = subprocess.run(
        ["tar", "-x", "-C", str(destination_root)],
        stdin=archive_process.stdout,
        capture_output=True,
        text=True,
        check=False,
    )
    archive_process.stdout.close()
    archive_stderr = archive_process.stderr.read().decode("utf-8", errors="replace")
    archive_returncode = archive_process.wait()

    if archive_returncode != 0:
        error = archive_stderr.strip() or f"git archive failed for ref={ref}"
        raise RuntimeError(error)
    if untar_process.returncode != 0:
        error = untar_process.stderr.strip() or "tar extraction failed"
        raise RuntimeError(error)

    extracted = destination_root / grug_root
    if not extracted.is_dir():
        raise RuntimeError(f"Expected extracted directory at {extracted}")
    return extracted


def _line_cache_get(path: Path, cache: dict[Path, list[str]]) -> list[str]:
    if path not in cache:
        cache[path] = read_text_lines(path)
    return cache[path]


def directory_distance(
    *,
    left_dir: Path,
    right_dir: Path,
    extensions: tuple[str, ...] = DEFAULT_EXTENSIONS,
) -> int:
    """Compute a simple line-delta distance between two code directories."""
    left_files = collect_files(left_dir, extensions=extensions, include_all_files=False)
    right_files = collect_files(right_dir, extensions=extensions, include_all_files=False)
    all_paths = set(left_files) | set(right_files)

    line_cache: dict[Path, list[str]] = {}
    distance = 0

    for rel_path in all_paths:
        left_path = left_files.get(rel_path)
        right_path = right_files.get(rel_path)

        if left_path is None:
            distance += len(_line_cache_get(right_path, line_cache))
            continue

        if right_path is None:
            distance += len(_line_cache_get(left_path, line_cache))
            continue

        left_lines = _line_cache_get(left_path, line_cache)
        right_lines = _line_cache_get(right_path, line_cache)
        if left_lines == right_lines:
            continue

        added, deleted = line_change_counts(left_lines, right_lines)
        distance += added + deleted

    return distance


def find_closest_variant(
    *,
    variant_dir: Path,
    candidate_dirs: dict[str, Path],
    extensions: tuple[str, ...] = DEFAULT_EXTENSIONS,
) -> VariantMatch:
    """Find the closest existing variant directory by line-delta distance."""
    if not candidate_dirs:
        raise ValueError("No candidate variants available for comparison")

    best_variant: str | None = None
    best_score: int | None = None

    for variant_name, candidate_dir in sorted(candidate_dirs.items()):
        score = directory_distance(
            left_dir=candidate_dir,
            right_dir=variant_dir,
            extensions=extensions,
        )
        if best_score is None or score < best_score:
            best_variant = variant_name
            best_score = score

    return VariantMatch(
        variant=variant_dir.name,
        closest_variant=best_variant,
        distance_score=best_score,
    )


def build_manifest(
    *,
    base_sha: str,
    head_sha: str,
    output_dir: Path,
    context_lines: int,
    extensions: tuple[str, ...] = DEFAULT_EXTENSIONS,
) -> dict[str, object]:
    """Build diff reports for newly added variants and return manifest JSON data."""
    base_variants = list_variants_at_ref(ref=base_sha)
    head_variants = list_variants_at_ref(ref=head_sha)
    added_variants = sorted(head_variants - base_variants)

    reports: list[dict[str, object]] = []
    with tempfile.TemporaryDirectory(prefix="grug-base-snapshot-") as tmpdir:
        snapshot_root = Path(tmpdir)
        base_grug_root = materialize_grug_tree_at_ref(
            ref=base_sha,
            destination_root=snapshot_root,
            grug_root=GRUG_ROOT,
        )
        candidate_names = sorted(base_variants | {"base"})
        candidate_dirs = {name: base_grug_root / name for name in candidate_names if (base_grug_root / name).is_dir()}

        for variant_name in added_variants:
            variant_dir = GRUG_ROOT / variant_name
            if not variant_dir.is_dir():
                continue

            closest = find_closest_variant(
                variant_dir=variant_dir,
                candidate_dirs=candidate_dirs,
                extensions=extensions,
            )

            report_dir = output_dir / variant_name
            index_path, entries = build_directory_diff_report(
                left_dir=candidate_dirs[closest.closest_variant],
                right_dir=variant_dir,
                output_dir=report_dir,
                left_label=f"experiments/grug/{closest.closest_variant}",
                right_label=f"experiments/grug/{variant_name}",
                extensions=extensions,
                include_all_files=False,
                show_unchanged=False,
                context_lines=context_lines,
            )

            status_counts = {
                "changed": sum(1 for entry in entries if entry.status == "changed"),
                "added": sum(1 for entry in entries if entry.status == "added"),
                "removed": sum(1 for entry in entries if entry.status == "removed"),
                "unchanged": sum(1 for entry in entries if entry.status == "unchanged"),
            }

            reports.append(
                {
                    "variant": variant_name,
                    "closest_variant": closest.closest_variant,
                    "distance_score": closest.distance_score,
                    "report_relpath": index_path.relative_to(output_dir).as_posix(),
                    "status_counts": status_counts,
                }
            )

    return {
        "base_sha": base_sha,
        "head_sha": head_sha,
        "added_variants": added_variants,
        "report_count": len(reports),
        "reports": reports,
    }


def parse_args() -> argparse.Namespace:
    """Parse command-line args."""
    parser = argparse.ArgumentParser(
        description=(
            "Detect newly added Grug variants and build HTML diff reports " "against the most similar existing variant."
        )
    )
    parser.add_argument("--base-sha", required=True, help="Base commit SHA")
    parser.add_argument("--head-sha", required=True, help="Head commit SHA")
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for generated reports and manifest",
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        required=True,
        help="Path to write manifest JSON",
    )
    parser.add_argument(
        "--extensions",
        default=",".join(DEFAULT_EXTENSIONS),
        help="Comma-separated extensions to compare",
    )
    parser.add_argument(
        "--context-lines",
        type=int,
        default=3,
        help="Context lines for inline diff rendering",
    )
    return parser.parse_args()


def main() -> int:
    """CLI entrypoint."""
    args = parse_args()
    if args.context_lines < 0:
        raise ValueError("--context-lines must be >= 0")

    extensions = parse_extensions(args.extensions)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.manifest_path.parent.mkdir(parents=True, exist_ok=True)

    manifest = build_manifest(
        base_sha=args.base_sha,
        head_sha=args.head_sha,
        output_dir=args.output_dir,
        context_lines=args.context_lines,
        extensions=extensions,
    )

    args.manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    report_count = manifest["report_count"]
    print(f"Wrote manifest: {args.manifest_path.resolve()}")
    print(f"Generated reports: {report_count}")
    for report in manifest["reports"]:
        print(
            f"- {report['variant']} vs {report['closest_variant']} "
            f"(score={report['distance_score']}) -> {report['report_relpath']}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
