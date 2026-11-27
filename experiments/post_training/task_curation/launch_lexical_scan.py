# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Launch the lexical scan for the staged conversion-release populations.

The scan consumes the staged populations manifest, rebuilds the DF masks for
the new cohort set (no frozen drop sets: the new populations are not in the
recorded artifact), measures against the pinned shared bloom, and lands the
lexical report artifact. Interactive priority per the operator directive.

The artifact directory is keyed by source and run —
``lexical-scan/<source-slug>/<artifact-version>/`` — because a population
version is not unique to one source: two lanes staging the same population
version would overwrite each other's reports there (the 2026-11-27 cross-lane
incident, where a crosscodeeval scan replaced the OpenSWE lane's lexical
report). The launcher refuses to write into a directory that already holds
another source's plan or report, and every plan it writes self-identifies its
source through a ``source_slug`` row.
"""

import argparse
import subprocess
from pathlib import Path

from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath

BLOOM = "s3://marin-us-east-02a/marin/datakit/bloom/_combined_fixed_eca06d42"
EVAL_CORPUS_VERSION = "aa-index-v4.1.1-v3"
LEXICAL_SCAN_ROOT = "s3://marin-us-east-02a/marin/rl-data/atlas/lexical/text-cohort"
PLAN_FILES = ("lexical-report.json", "plan.tsv")


def artifact_dir(source_slug: str, artifact_version: str, *, root: str = LEXICAL_SCAN_ROOT) -> str:
    """The source- and run-keyed directory holding one lexical scan's artifacts."""
    return f"{root.rstrip('/')}/lexical-scan/{source_slug}/{artifact_version}"


def artifact_paths(source_slug: str, artifact_version: str, *, root: str = LEXICAL_SCAN_ROOT) -> dict[str, str]:
    """The four artifact URIs of one lexical scan: plan, populations, source ids, report."""
    for segment in (source_slug, artifact_version):
        if not segment or "/" in segment:
            raise ValueError(f"source slug and artifact version must be single path segments: {segment!r}")
    directory = artifact_dir(source_slug, artifact_version, root=root)
    return {
        "artifact_dir": directory,
        "plan": f"{directory}/plan.tsv",
        "populations": f"{directory}/populations.json",
        "source_ids": f"{directory}/source-ids.json",
        "report": f"{directory}/lexical-report.json",
    }


def build_plan_text(
    paths: dict[str, str], populations_manifest: str, source_ids: str, artifact_version: str, source_slug: str
) -> str:
    """The scan plan; its ``source_slug`` row lets later runs refuse foreign directories."""
    return (
        f"source_slug\t{source_slug}\n"
        f"populations\t{populations_manifest}\n"
        f"source_ids\t{source_ids}\n"
        f"populations_uri\t{paths['populations']}\n"
        f"source_ids_uri\t{paths['source_ids']}\n"
        f"report_uri\t{paths['report']}\n"
        f"artifact_version\t{artifact_version}\n"
        f"bloom\t{BLOOM}\n"
        f"eval_corpus_version\t{EVAL_CORPUS_VERSION}\n"
    )


def plan_source_slug(plan_text: str) -> str | None:
    """The plan's ``source_slug`` value, or None when the plan predates the keying."""
    for line in plan_text.splitlines():
        if line.startswith("source_slug\t"):
            return line.split("\t", 1)[1].strip()
    return None


def refuse_foreign_artifacts(found: set[str], plan_text: str | None, source_slug: str, directory: str) -> None:
    """Refuse to write into a directory that already holds another source's report or plan.

    A directory holding our own plan (the same ``source_slug`` row) is ours to re-run. A
    directory holding an unkeyed plan, a report without a plan, or a plan naming a different
    source is somebody else's: writing there would repeat the cross-lane overwrite.
    """
    if not found:
        return
    if "lexical-report.json" in found and "plan.tsv" not in found:
        raise SystemExit(
            f"refusing to write into {directory}: it already holds a lexical-report.json with no "
            "plan.tsv, so another lane's unkeyed run owns it; choose a new artifact version"
        )
    recorded = plan_source_slug(plan_text) if plan_text is not None else None
    if recorded is None:
        raise SystemExit(
            f"refusing to write into {directory}: its plan.tsv predates the source keying (no "
            "source_slug row), so it may hold another lane's report; choose a new artifact version"
        )
    if recorded != source_slug:
        raise SystemExit(
            f"refusing to write into {directory}: its plan.tsv names source_slug {recorded!r}, "
            f"not {source_slug!r}; choose a new artifact version"
        )


def existing_artifacts(directory: str) -> tuple[set[str], str | None]:
    """Which plan files already occupy the directory, and its plan.tsv text if readable."""
    configure_coreweave_s3()
    found: set[str] = set()
    plan_text = None
    for name in PLAN_FILES:
        path = StoragePath(f"{directory}/{name}")
        try:
            data = path.read_bytes()
        except FileNotFoundError:
            continue
        found.add(name)
        if name == "plan.tsv":
            plan_text = data.decode()
    return found, plan_text


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", required=True, help="The landed marin revision to run at.")
    parser.add_argument(
        "--populations-manifest", required=True, help="The staged populations manifest (populations.json)."
    )
    parser.add_argument("--source-ids", required=True, help="The staged source-ids.json.")
    parser.add_argument("--population-version", required=True, help="The staged population version.")
    parser.add_argument("--artifact-version", required=True, help="The lexical artifact version.")
    parser.add_argument(
        "--source-slug",
        required=True,
        help="The source keying the artifact directory (lexical-scan/<source-slug>/<artifact-version>/).",
    )
    parser.add_argument("--job-name", default="atlas-taskspec-lexical-scan")
    parser.add_argument("--max-workers", default="16")
    parser.add_argument("--cpu", default="16")
    parser.add_argument("--memory", default="128GB")
    parser.add_argument("--disk", default="300GB")
    parser.add_argument("--timeout", default="259200", help="Job timeout in seconds (default: three days).")
    args = parser.parse_args()

    paths = artifact_paths(args.source_slug, args.artifact_version)
    directory = paths["artifact_dir"]
    found, existing_plan = existing_artifacts(directory)
    refuse_foreign_artifacts(found, existing_plan, args.source_slug, directory)

    plan_local = Path("build") / "lexical-scan-plan.tsv"
    plan_local.parent.mkdir(exist_ok=True)
    plan_local.write_text(
        build_plan_text(paths, args.populations_manifest, args.source_ids, args.artifact_version, args.source_slug)
    )
    subprocess.run(
        ["uv", "run", "--no-sync", "fsutil", "cp", str(plan_local), paths["plan"]],
        check=True,
        stdout=subprocess.DEVNULL,
    )
    subprocess.run(
        ["uv", "run", "--no-sync", "fsutil", "cp", args.populations_manifest, paths["populations"]],
        check=True,
        stdout=subprocess.DEVNULL,
    )
    subprocess.run(
        ["uv", "run", "--no-sync", "fsutil", "cp", args.source_ids, paths["source_ids"]],
        check=True,
        stdout=subprocess.DEVNULL,
    )

    script_uri = (
        "https://raw.githubusercontent.com/marin-community/marin/"
        f"{args.revision}/experiments/post_training/task_curation/run_lexical_scan.sh"
    )
    command = (
        f"mkdir -p /workspace && curl -fsSL {script_uri} -o /workspace/run_lexical_scan.sh && "
        f"bash /workspace/run_lexical_scan.sh {args.revision} {paths['populations']} {paths['source_ids']} "
        f"{args.population_version} {args.artifact_version} {args.max_workers} {paths['report']}"
    )
    args_list = [
        "uv",
        "run",
        "--no-sync",
        "iris",
        "--cluster=marin",
        "job",
        "run",
        "--no-wait",
        "--target-cluster=cw-us-east-02a",
        "--no-sync",
        "--secret",
        "HF_TOKEN",
        "env:HF_TOKEN",
        "--secret",
        "CW_KEY_ID",
        "env:CW_KEY_ID",
        "--secret",
        "CW_KEY_SECRET",
        "env:CW_KEY_SECRET",
        "--cpu",
        args.cpu,
        "--memory",
        args.memory,
        "--disk",
        args.disk,
        "--enable-extra-resources",
        "--priority",
        "interactive",
        "--timeout",
        args.timeout,
        "--max-retries",
        "0",
        "--job-name",
        args.job_name,
        "--",
        "bash",
        "-c",
        command,
    ]
    completed = subprocess.run(args_list)
    if completed.returncode:
        raise SystemExit(completed.returncode)
    print(f"lexical scan job submitted: /benfeuer/{args.job_name}")
    print(f"report will land at: {paths['report']}")


if __name__ == "__main__":
    main()
