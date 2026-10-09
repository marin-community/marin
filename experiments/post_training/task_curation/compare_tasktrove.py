# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compare retained TaskTrove conversions with a fully isolated, pinned legacy converter."""

import hashlib
import importlib.metadata
import json
import re
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path
from typing import Any

import click
import verifyit
from taskcompendium.harbor import snapshots
from taskcompendium.harbor.compare import ParityReport, Tolerance
from taskcompendium.harbor.export import UnsupportedHarborTask, harbor_record
from taskcompendium.harbor.records import NormalizedIndex
from taskcompendium.harbor.snapshots import TaskSnapshot, snapshot_from_dict, task_snapshot
from taskcompendium.models import AnswerType, TaskSpec

from experiments.post_training.task_curation.datasets.tasktrove.archives import TASKTROVE_REPO
from experiments.post_training.task_curation.images.build import BASE_IMAGE
from experiments.post_training.task_curation.pipeline import HfSource
from experiments.post_training.task_curation.source import RlDataSource
from experiments.post_training.task_curation.sources import all_sources

FROZEN_PATHS = (
    "experiments/post_training/tasktrove",
    "lib/taskcompendium/src/taskcompendium",
    "lib/verifyit/src/verifyit",
)
ROW_METADATA = ("family", "template_id", "converter", "mode", "dockerfile_id", "language", "tags", "has_solution")


def file_identity(path: Path) -> dict[str, Any]:
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": digest}


def frozen_checkout(repository: Path, revision: str, destination: Path) -> dict[str, str]:
    """Extract only the pinned reference implementation, including its shared helpers."""
    if re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise ValueError("The baseline must be a full commit SHA")
    resolved = subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", revision + "^{commit}"], text=True
    ).strip()
    if resolved != revision:
        raise ValueError("Baseline revision did not resolve exactly")
    destination.mkdir()
    with subprocess.Popen(
        ["git", "-C", str(repository), "archive", revision, *FROZEN_PATHS], stdout=subprocess.PIPE
    ) as process:
        assert process.stdout is not None
        with tarfile.open(fileobj=process.stdout, mode="r|") as archive:
            archive.extractall(destination, filter="data")
        if process.wait() != 0:
            raise RuntimeError("Failed to extract frozen reference")
    return {
        path: (
            subprocess.check_output(["git", "-C", str(repository), "rev-parse", f"{revision}:{path}"], text=True).strip()
        )
        for path in FROZEN_PATHS
    }


def candidate_snapshot(row: dict[str, Any], source: RlDataSource, config: str, grader_image: str) -> TaskSnapshot:
    if row["task_json"] is None:
        return task_snapshot(
            config, row["original_path"], row["normalization_reason"], detail=row["normalization_detail"] or ""
        )
    task = TaskSpec.model_validate_json(row["task_json"])
    try:
        record = harbor_record(
            row,
            family=source.info.family,
            grader_image=None if task.answer_type == AnswerType.WORKSPACE_STATE else grader_image,
            fallback_actor_image=BASE_IMAGE,
        )
    except UnsupportedHarborTask as error:
        return task_snapshot(config, row["original_path"], "lowering_rejection", detail=str(error))
    if record.source != config or record.path != row["original_path"]:
        raise ValueError("Export changed the original source/path identity")
    return task_snapshot(
        record.source,
        record.path,
        "converted",
        task_binary=record.task_binary,
        solution_binary=record.solution_binary,
        metadata={key: getattr(record, key) for key in ROW_METADATA},
    )


def compare_source(
    source: RlDataSource,
    normalized: Path,
    raw: Path,
    reference: Path,
    revision: str,
    output: Path,
    grader_image: str,
    tolerances: frozenset[Tolerance],
) -> dict[str, Any]:
    """Stream one exact source population and persist every matching or mismatching task."""
    pipeline = source.pipeline
    assert pipeline is not None and isinstance(pipeline.source, HfSource)
    if len(pipeline.source.files) != 1:
        raise ValueError(f"Expected one pinned TaskTrove file for {source.name}")
    config = pipeline.source.files[0].split("/", 1)[0]
    raw_file = raw / pipeline.source.files[0]
    manifest = json.loads((normalized / "manifest.json").read_text())
    if (manifest["source"], manifest["source_dataset"], manifest["source_revision"]) != (
        source.name,
        pipeline.source.repo,
        pipeline.source.revision,
    ):
        raise ValueError(f"Normalized input provenance does not match {source.name}")
    output.mkdir()
    provenance = {
        "source": source.name,
        "original_source": config,
        "baseline_revision": revision,
        "baseline_layer": "convert_one plus pure check_spec/check_dockerfile/check_gold_leak/check_shape",
        "not_executed": ["grading", "model_review", "deduplication", "release_cap"],
        "candidate_layer": "normalized TaskSpec followed by current Harbor export",
        "normalization_manifest": manifest,
        "manifest_file": file_identity(normalized / "manifest.json"),
        "raw_input": file_identity(raw_file),
        "normalized_inputs": [file_identity(path) for path in sorted((normalized / "normalize").glob("*.parquet"))],
        "tolerances": sorted(tolerances),
        "runtime_verified": False,
    }
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    started = time.monotonic()
    worker = Path(__file__).with_name("tasktrove_reference.py")
    assert snapshots.__file__ is not None
    command = [
        sys.executable,
        str(worker),
        "--checkout",
        str(reference),
        "--snapshots-module",
        snapshots.__file__,
        "--source",
        config,
        "--input",
        str(raw_file),
        "--revision",
        revision,
    ]
    with (
        tempfile.TemporaryDirectory(prefix="index-", dir=output) as temporary,
        NormalizedIndex(normalized / "normalize", Path(temporary) / "rows.sqlite") as index,
        ParityReport(output / "parity.sqlite") as report,
        (output / "reference.stderr").open("w") as stderr,
        subprocess.Popen(command, stdout=subprocess.PIPE, stderr=stderr, text=True, cwd=reference) as process,
    ):
        assert process.stdout is not None
        try:
            for line in process.stdout:
                baseline = snapshot_from_dict(json.loads(line))
                row = index.get(baseline.path)
                current = candidate_snapshot(row, source, config, grader_image) if row is not None else None
                report.add(baseline, current, tolerances=tolerances)
                if report.counts["tasks"] % 1000 == 0:
                    click.echo(json.dumps({"source": source.name, **report.summary()["counts"]}))
            if process.wait() != 0:
                raise RuntimeError(f"Frozen reference failed; see {output / 'reference.stderr'}")
            for row in index.unmatched():
                report.add(None, candidate_snapshot(row, source, config, grader_image), tolerances=tolerances)
        finally:
            if process.poll() is None:
                process.terminate()
                process.wait()
        summary = {**provenance, **report.summary(), "elapsed_seconds": time.monotonic() - started}
        candidate_rows = sum(count for key, count in report.counts.items() if key.startswith("candidate:"))
        if candidate_rows != manifest["input_rows"]:
            raise ValueError(
                f"Candidate row census disagrees with manifest: {candidate_rows} != {manifest['input_rows']}"
            )
        (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


@click.command(help=__doc__)
@click.option(
    "--normalized-root",
    "normalized_roots",
    multiple=True,
    required=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Source directory or parent containing source manifests; later roots override earlier ones.",
)
@click.option("--raw-root", required=True, type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--baseline-repository", required=True, type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--baseline-revision", required=True, help="Full legacy converter commit SHA.")
@click.option("--source", "selected", multiple=True, help="Pipeline name; default is all retained TaskTrove sources.")
@click.option(
    "--grader-image", required=True, help="Explicit digest-pinned image used when lowering non-repository graders."
)
@click.option("--allow", "allowed", multiple=True, type=click.Choice([value.value for value in Tolerance]))
@click.option("--output-root", required=True, type=click.Path(file_okay=False, path_type=Path))
def main(
    normalized_roots: tuple[Path, ...],
    raw_root: Path,
    baseline_repository: Path,
    baseline_revision: str,
    selected: tuple[str, ...],
    grader_image: str,
    allowed: tuple[str, ...],
    output_root: Path,
) -> None:
    catalog = {
        source.name: source
        for source in all_sources().values()
        if source.pipeline is not None
        and isinstance(source.pipeline.source, HfSource)
        and source.pipeline.source.repo == TASKTROVE_REPO
    }
    unknown = set(selected) - catalog.keys()
    if unknown:
        raise click.UsageError(f"Not retained TaskTrove sources: {sorted(unknown)}")
    names = sorted(set(selected) if selected else catalog)
    inputs = {}
    for root in normalized_roots:
        manifests = (
            [root / "manifest.json"] if (root / "manifest.json").is_file() else sorted(root.glob("*/manifest.json"))
        )
        for path in manifests:
            inputs[json.loads(path.read_text())["source"]] = path.parent.resolve()
    missing = set(names) - inputs.keys()
    if missing:
        raise click.UsageError(f"Missing normalized inputs: {sorted(missing)}")
    output_root = output_root.resolve()
    output_root.mkdir()
    trees = frozen_checkout(baseline_repository, baseline_revision, output_root / "reference")
    run = {
        "baseline_revision": baseline_revision,
        "frozen_trees": trees,
        "candidate_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "candidate_diff_sha256": (
            hashlib.sha256(subprocess.check_output(["git", "diff", "--binary", "HEAD"])).hexdigest()
        ),
        "python": sys.version,
        "package_versions": {
            name: importlib.metadata.version(name) for name in ("pyarrow", "pydantic", "tomlkit", "harbor-config")
        },
        "implementation_files": [
            file_identity(path)
            for path in sorted(
                {Path(__file__), Path(__file__).with_name("tasktrove_reference.py")}
                | set(Path(snapshots.__file__).parent.glob("*.py"))
                | set(Path(verifyit.__file__).parent.rglob("*.py"))
            )
        ],
        "grader_image": grader_image,
        "sources": names,
        "results": {},
    }
    (output_root / "run.json").write_text(json.dumps(run, indent=2) + "\n")
    for name in names:
        try:
            summary = compare_source(
                catalog[name],
                inputs[name],
                raw_root.resolve(),
                output_root / "reference",
                baseline_revision,
                output_root / name,
                grader_image,
                frozenset(Tolerance(value) for value in allowed),
            )
        except Exception as error:
            # A failed source stays explicit while the census continues through the remaining sources.
            run["results"][name] = {"status": "failed", "error": f"{type(error).__name__}: {error}"}
        else:
            run["results"][name] = {
                "status": "completed",
                "counts": summary["counts"],
                "elapsed_seconds": summary["elapsed_seconds"],
            }
        (output_root / "run.json").write_text(json.dumps(run, indent=2) + "\n")
        click.echo(json.dumps({"source": name, **run["results"][name]}))
    if any(result["status"] == "failed" for result in run["results"].values()):
        raise click.ClickException(f"Some sources failed; see {output_root / 'run.json'}")


if __name__ == "__main__":
    main()
