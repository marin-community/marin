# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""
Bulk-clone unique ``(repo, base_commit)`` pairs from SWE-rebench V2 in
parallel via Zephyr and upload tarballs to GCS for use by the SWE-ZERO
rollout workers.

Output layout:
    gs://<bucket>/swe_zero/repo_cache/<org>__<repo>/<short_sha>.tar.gz
    gs://<bucket>/swe_zero/repo_cache/manifest-<shard>-of-<total>.jsonl

The manifest contains one row per (repo, base_commit) with the GCS path,
size, and a list of instance_ids that share this checkout.

Usage (locally, hits the Iris-backed Zephyr cluster automatically):
    uv run python experiments/swe_zero/clone_repos.py \
        --language python \
        --instances core-gatech-group__serpent-tools-21,core-gatech-group__serpent-tools-272 \
        --max-workers 32

    # Or pre-clone every repo for an entire experiment scope:
    uv run python experiments/swe_zero/clone_repos.py \
        --language python --num-repos 10 --num-prs-per-repo 10
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import fsspec
from zephyr import Dataset, ZephyrContext

from experiments.swe_zero.data_loader import SWERebenchV2Loader
from experiments.swe_zero.worktree import DEFAULT_GCS_CACHE_ROOT, gcs_cache_key

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [%(name)s] %(message)s")
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CloneTask:
    """One unique (repo, base_commit) pair plus the instances that share it."""

    repo: str
    base_commit: str
    instance_ids: tuple[str, ...]
    gcs_root: str


def _gcs_exists(path: str) -> bool:
    fs, _ = fsspec.core.url_to_fs(path)
    return fs.exists(path)


def clone_one(task: CloneTask) -> dict:
    """Clone one (repo, base_commit) pair, tar.gz it, and upload to GCS.

    This is the function Zephyr distributes across workers. It's idempotent:
    if the GCS object already exists, we skip the clone entirely.
    """
    gcs_path = gcs_cache_key(task.repo, task.base_commit, task.gcs_root)
    record = {
        "repo": task.repo,
        "base_commit": task.base_commit,
        "instance_ids": list(task.instance_ids),
        "gcs_path": gcs_path,
        "size_bytes": 0,
        "duration_s": 0.0,
        "skipped": False,
        "error": None,
    }

    if _gcs_exists(gcs_path):
        record["skipped"] = True
        fs, _ = fsspec.core.url_to_fs(gcs_path)
        try:
            record["size_bytes"] = fs.size(gcs_path)
        except Exception:
            pass
        return record

    start = time.monotonic()
    tmpdir = Path(tempfile.mkdtemp(prefix="swe_zero_clone_"))
    try:
        url = f"https://github.com/{task.repo}.git"
        subprocess.run(["git", "init", "--quiet"], cwd=tmpdir, check=True)
        subprocess.run(["git", "remote", "add", "origin", url], cwd=tmpdir, check=True)
        subprocess.run(
            ["git", "fetch", "--quiet", "--depth=1", "origin", task.base_commit],
            cwd=tmpdir,
            check=True,
            timeout=600,
        )
        subprocess.run(["git", "checkout", "--quiet", "FETCH_HEAD"], cwd=tmpdir, check=True)

        # Drop .git to keep the tarball small — we don't need history at the
        # rollout site, only the working tree.
        shutil.rmtree(tmpdir / ".git", ignore_errors=True)

        with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp_tar:
            tarball = Path(tmp_tar.name)
        try:
            # tar with absolute prefix `repo/` so extraction with --strip-components=1
            # lands files at the worktree root cleanly.
            tar_cmd = [
                "tar",
                "czf",
                str(tarball),
                "-C",
                str(tmpdir.parent),
                f"--transform=s,^{tmpdir.name},repo,",
                tmpdir.name,
            ]
            subprocess.run(tar_cmd, check=True)
            size = tarball.stat().st_size
            with open(tarball, "rb") as src, fsspec.open(gcs_path, "wb") as dst:
                shutil.copyfileobj(src, dst, length=8 * 1024 * 1024)
            record["size_bytes"] = size
        finally:
            tarball.unlink(missing_ok=True)
    except subprocess.CalledProcessError as e:
        record["error"] = f"{e.cmd}: returncode={e.returncode}"
    except subprocess.TimeoutExpired as e:
        record["error"] = f"{e.cmd}: timed out after {e.timeout}s"
    except Exception as e:
        record["error"] = repr(e)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
        record["duration_s"] = round(time.monotonic() - start, 2)

    return record


def _build_clone_tasks_from_loader(
    loader: SWERebenchV2Loader,
    *,
    instances: list[str] | None,
    num_repos: int | None,
    num_prs_per_repo: int | None,
    seed: int,
    gcs_root: str,
) -> list[CloneTask]:
    """Resolve the requested PR set into unique (repo, base_commit) clone tasks."""
    selected_records = []
    if instances:
        for iid in instances:
            selected_records.append(loader.get(iid))
    elif num_repos is not None and num_prs_per_repo is not None:
        repos = loader.sample_repos(n=num_repos, min_prs=num_prs_per_repo, seed=seed)
        for repo in repos:
            for pr in loader.sample_prs(repo, n=num_prs_per_repo, seed=seed):
                selected_records.append(pr)
    else:
        raise ValueError("Provide either --instances or both --num-repos and --num-prs-per-repo")

    by_pair: dict[tuple[str, str], list[str]] = {}
    for pr in selected_records:
        by_pair.setdefault((pr.repo, pr.base_commit), []).append(pr.instance_id)

    return [
        CloneTask(repo=repo, base_commit=commit, instance_ids=tuple(ids), gcs_root=gcs_root)
        for (repo, commit), ids in sorted(by_pair.items())
    ]


def main():
    parser = argparse.ArgumentParser(description="Bulk-clone SWE-rebench V2 repos via Zephyr")
    parser.add_argument("--language", default="python", help="Filter SWE-rebench V2 by language")
    parser.add_argument(
        "--instances", default=None, help="Comma-separated instance IDs (overrides --num-repos/--num-prs-per-repo)"
    )
    parser.add_argument("--num-repos", type=int, default=None)
    parser.add_argument("--num-prs-per-repo", type=int, default=None)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--gcs-root", default=DEFAULT_GCS_CACHE_ROOT)
    parser.add_argument("--max-workers", type=int, default=32)
    parser.add_argument(
        "--manifest-path",
        default=None,
        help="Where to write the JSONL manifest (defaults to <gcs_root>/manifest-...jsonl)",
    )
    args = parser.parse_args()

    instances = [s.strip() for s in args.instances.split(",")] if args.instances else None

    logger.info("Loading SWE-rebench V2 (language=%s)", args.language)
    loader = SWERebenchV2Loader(language_filter=args.language)

    tasks = _build_clone_tasks_from_loader(
        loader,
        instances=instances,
        num_repos=args.num_repos,
        num_prs_per_repo=args.num_prs_per_repo,
        seed=args.seed,
        gcs_root=args.gcs_root,
    )
    logger.info(
        "Built %d clone tasks (covering %d instances)",
        len(tasks),
        sum(len(t.instance_ids) for t in tasks),
    )
    if not tasks:
        logger.warning("Nothing to clone, exiting")
        return

    manifest_path = args.manifest_path or f"{args.gcs_root}/manifest-{int(time.time())}.jsonl"
    logger.info("Manifest will be written to %s", manifest_path)

    ctx = ZephyrContext(name="swe-zero-clone-repos", max_workers=args.max_workers)
    pipeline = Dataset.from_list(tasks).map(clone_one)
    results = list(ctx.execute(pipeline))

    n_ok = sum(1 for r in results if not r.get("error"))
    n_skipped = sum(1 for r in results if r.get("skipped"))
    n_err = sum(1 for r in results if r.get("error"))
    total_size_mb = sum(r.get("size_bytes", 0) for r in results) / (1024 * 1024)
    logger.info("Done: ok=%d, skipped=%d, error=%d, total=%.1f MB", n_ok, n_skipped, n_err, total_size_mb)

    with fsspec.open(manifest_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    logger.info("Wrote manifest with %d entries to %s", len(results), manifest_path)

    if n_err:
        logger.warning("First few errors:")
        for r in results:
            if r.get("error"):
                logger.warning("  %s @ %s: %s", r["repo"], r["base_commit"][:12], r["error"])
                n_err -= 1
                if n_err == 0:
                    break


if __name__ == "__main__":
    main()
