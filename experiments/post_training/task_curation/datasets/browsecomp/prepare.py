# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build complete TaskCompendium task directories from the pinned benchmark."""

import argparse
import hashlib
import json
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from finestore.rl.licenses import LicenseReceipt, require_license
from finestore.rl.status_taxonomy import DataRestriction
from fray.types import ResourceConfig
from marin.execution.artifact import Artifact
from marin.execution.lazy import ArtifactStep, StepContext
from taskcompendium.harbor.adapter import write_harbor_task
from taskcompendium.models import ExecutionRequirements
from taskcompendium.pipeline.audit_schema import NORMALIZED_SCHEMA
from taskcompendium.pipeline.conversion import normalization_columns
from taskcompendium.pipeline.inputs import SourceFormat
from taskcompendium.pipeline.models import IntendedUse, SourceStatus
from taskcompendium.pipeline.source_processing import SourceProcessingMode

from experiments.post_training.task_curation.campaign import CampaignArtifact, CampaignRuntime, PipelineResult
from experiments.post_training.task_curation.config import InputOverrides, PipelineOptions
from experiments.post_training.task_curation.license_binding import licensed_source
from experiments.post_training.task_curation.local import run_local_steps
from experiments.post_training.task_curation.pipeline import HfSource, download_step, input_file_overrides
from experiments.post_training.task_curation.source import RlDataSource, SourceInfo, SourceOrigin, TaskFormat

from .tasks import POPULATION, QUERY_REPO, QUERY_REVISION, STATE, VERSION, convert_query

SOURCE_ID = "Tevatron:browsecomp-plus@bm25"
INITIAL_REQUIREMENTS = ExecutionRequirements(
    cpus=1, memory_mb=2048, storage_mb=2048, agent_timeout_sec=300, verifier_timeout_sec=60
)


@dataclass(frozen=True)
class PrepareRun:
    input_root: str
    output_root: str
    actor_image: str
    license_receipt: LicenseReceipt
    input_sha256: dict[str, str]


@dataclass(frozen=True)
class BrowseCompDataset:
    actor_image: str
    source: HfSource

    @property
    def name(self) -> str:
        return "browsecomp-plus-bm25"

    @property
    def version(self) -> str:
        return VERSION

    @property
    def dataset(self) -> HfSource:
        return self.source

    @property
    def files(self) -> tuple[str, ...]:
        return self.source.files

    @property
    def intended_use(self) -> IntendedUse:
        return IntendedUse.EVAL


def query_file_hashes(root: Path) -> dict[str, str]:
    files = {path.relative_to(root).as_posix(): path for path in (root / "data").glob("test-*.parquet")}
    return {name: value.sha256 for name, value in input_file_overrides(files).items()}


def prepare(run: PrepareRun) -> CampaignArtifact:
    receipt = run.license_receipt
    require_license(receipt, SOURCE_ID, "eval", ban=None)
    input_sha256 = query_file_hashes(Path(run.input_root))
    if run.input_sha256 and input_sha256 != run.input_sha256:
        raise ValueError("BrowseComp query inputs changed after artifact planning")
    output = Path(run.output_root)
    if (output / "tasks.parquet").exists() or (output / "task-directories").exists():
        raise FileExistsError("BrowseComp conversion requires a new output version")
    output.mkdir(parents=True, exist_ok=True)
    records, normalized = [], []
    seen = set()
    for path in sorted((Path(run.input_root) / "data").glob("test-*.parquet")):
        # The source's relevance documents stay in the frozen upstream payload. They are not
        # solver context or duplicated into each contract; the full index serves retrieval.
        rows = pq.read_table(path, columns=["query_id", "query", "answer"]).to_pylist()
        for row in rows:
            task = convert_query(row, run.actor_image).model_copy(
                update={
                    "execution_requirements": INITIAL_REQUIREMENTS,
                    "verifier_execution_requirements": INITIAL_REQUIREMENTS,
                }
            )
            if task.id in seen:
                raise ValueError("Repeated BrowseComp query ID in the complete source")
            seen.add(task.id)
            directory = output / "task-directories" / task.id
            write_harbor_task(task, directory, INITIAL_REQUIREMENTS)
            encoded = task.model_dump_json()
            records.append(
                {
                    "id": task.id,
                    "task_spec": encoded,
                    "task_sha256": hashlib.sha256(encoded.encode()).hexdigest(),
                    "path": f"task-directories/{task.id}",
                }
            )
            normalized.append(
                {**normalization_columns(task.id, task.source, task, None, (), task.source.row), "task_json": encoded}
            )
    if len(records) != POPULATION:
        raise ValueError(f"Complete BrowseComp population has {len(records)} queries; expected {POPULATION}")
    if query_file_hashes(Path(run.input_root)) != input_sha256:
        raise ValueError("BrowseComp query inputs changed during conversion")
    pq.write_table(pa.Table.from_pylist(records), output / "tasks.parquet", compression="zstd")
    (output / "normalize").mkdir()
    pq.write_table(pa.Table.from_pylist(normalized, schema=NORMALIZED_SCHEMA), output / "normalize/part-00000.parquet")
    manifest = {
        "mode": "quick",
        "intended_use": "eval",
        "source_id": SOURCE_ID,
        "source_dataset": QUERY_REPO,
        "source_revision": QUERY_REVISION,
        "converter_version": VERSION,
        "input_rows": len(records),
        "converted_rows": len(records),
        "input_file_sha256": input_sha256,
        "conversion_rejections": {},
        "task_format": "taskcompendium",
        "actor_image": run.actor_image,
        "retrieval": STATE.model_dump(mode="json"),
        "oracle_solutions": False,
        "oracle_reason": "The source publishes reference answers but no executable oracle solutions.",
        "reviewed": False,
        "admitted": False,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (output / "license.json").write_text(receipt.model_dump_json(indent=2) + "\n")
    return CampaignArtifact(
        path=str(output),
        status=SourceStatus.COMPLETED,
        result=PipelineResult(
            SourceStatus.COMPLETED,
            {"normalize": str(output / "normalize"), "tasks": str(output / "tasks.parquet")},
            {"manifest": str(output / "manifest.json")},
            ("conversion", "task-directories"),
        ),
    )


def build_run(
    source: RlDataSource[BrowseCompDataset],
    options: PipelineOptions,
    downloaded: ArtifactStep[Artifact] | None,
    input_sha256: dict[str, str],
    ctx: StepContext,
) -> PrepareRun:
    config = source.config
    assert config is not None and source.info.license_receipt is not None
    root = ctx.artifact_path(downloaded) if downloaded is not None else options.inputs.root
    if root is None:
        raise ValueError("BrowseComp preparation requires its complete pinned query input")
    return PrepareRun(
        root,
        ctx.output_path,
        config.actor_image,
        source.info.license_receipt,
        input_sha256,
    )


def curate(source: RlDataSource[BrowseCompDataset], options: PipelineOptions) -> ArtifactStep[CampaignArtifact]:
    source = licensed_source(source)
    config = source.config
    assert config is not None
    if options.mode is not SourceProcessingMode.QUICK:
        raise ValueError("Prepare the complete benchmark in QUICK, then qualify the frozen private release")
    if options.inputs.files or options.inputs.auxiliary:
        raise ValueError("BrowseComp preparation accepts a complete query root, not partial file overrides")
    downloaded = download_step(config.source, options.runtime) if options.inputs.root is None else None
    hashes = query_file_hashes(Path(options.inputs.root)) if options.inputs.root is not None else {}
    return ArtifactStep(
        name="data/eval/browsecomp-plus-bm25",
        version=VERSION,
        artifact_type=CampaignArtifact,
        run=prepare,
        build_config=partial(build_run, source, options, downloaded, hashes),
        deps=(downloaded,) if downloaded is not None else (),
    )


def source(actor_image: str, receipt: LicenseReceipt) -> RlDataSource[BrowseCompDataset]:
    return RlDataSource(
        info=SourceInfo(
            data_restriction=DataRestriction.OPEN,
            origin=SourceOrigin.TEVATRON,
            upstream_dataset="browsecomp-plus",
            upstream_name="BrowseComp-Plus",
            variant="bm25",
            build_path="BrowseComp-Plus offline BM25",
            family="offline-search",
            tags=("benchmark", "license:mit"),
            task_format=TaskFormat.TASKSPEC,
            license_receipt=receipt.model_copy(update={"source_id": SOURCE_ID}),
        ),
        config=BrowseCompDataset(
            actor_image,
            HfSource(QUERY_REPO, QUERY_REVISION, ("data/test-*.parquet",), SourceFormat.PARQUET),
        ),
        pipeline=curate,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--actor-image", required=True)
    parser.add_argument("--license-receipt", type=Path, required=True)
    args = parser.parse_args()
    receipt = LicenseReceipt.model_validate_json(args.license_receipt.read_text()).model_copy(
        update={"source_id": SOURCE_ID}
    )
    selected = source(args.actor_image, receipt)
    runtime = CampaignRuntime()
    options = PipelineOptions(
        mode=SourceProcessingMode.QUICK,
        runtime=runtime,
        inputs=InputOverrides(root=str(args.input_root)),
    )
    results = run_local_steps(
        {selected.name: curate(selected, options)},
        args.output_root,
        runtime=runtime,
        max_workers=1,
        worker_resources=ResourceConfig(cpu=1, ram="4g"),
        cache_dir=args.output_root / "cache",
        source_ids={selected.name: selected.info.id},
    )
    print(json.dumps([{"source": SOURCE_ID, "output": result.path, "status": result.status} for result in results]))


if __name__ == "__main__":
    main()
