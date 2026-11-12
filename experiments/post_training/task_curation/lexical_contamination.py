# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Measure complete Atlas task populations with the existing Datakit lexical decontaminator."""

import argparse
import hashlib
import json
import logging
from collections import Counter
from collections.abc import Iterable
from contextlib import ExitStack
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import NamedTuple

import pyarrow as pa
import pyarrow.parquet as pq
from finestore.rl.source_identity import canonical_source_id, original_source_id
from fray.current_client import set_current_client
from fray.local_backend import LocalClient
from fray.types import ResourceConfig
from marin.execution.lazy import ArtifactStep, StepContext, run
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath, prefix_join
from taskcompendium.identity import canonical_sha256
from taskcompendium.models import CommandSemantics, EnvironmentRequirements, TaskSpec, format_conversation
from taskcompendium.pipeline.inputs import SourceFormat
from taskcompendium.pipeline.sources import conversion_context, source_shards, staged_raw_file_rows
from zephyr.context import ZephyrContext

from experiments.post_training.task_curation.campaign import CampaignRuntime
from experiments.post_training.task_curation.evaluation import EvalDataset
from experiments.post_training.task_curation.lexical_artifacts import SourceTextPopulation
from experiments.post_training.task_curation.local import artifact_prefix
from experiments.post_training.task_curation.pipeline import (
    CurationRecipe,
    HfSource,
    download_step,
    environment_requirements,
    planned_lock_url,
    source_recipe,
)
from experiments.post_training.task_curation.prebuilt import PrebuiltCohort
from experiments.post_training.task_curation.sources import all_sources

logger = logging.getLogger(__name__)
RENDERER_VERSION = "taskcompendium-format-conversation-v1"
PARTITION_BYTES = 64 * 1024 * 1024
PARTITION_ROWS = 1000
TEXT_SCHEMA = pa.schema([("id", pa.string()), ("text", pa.large_string()), ("partition_id", pa.int64())])
SPEC_SCHEMA = pa.schema([("id", pa.string()), ("task_spec", pa.large_string()), ("runtime", pa.large_string())])


@dataclass(frozen=True)
class SourceTextConfig:
    source_id: str
    source_input: str
    inputs: dict[str, str]
    output_path: str
    converter_revision: str


class TextPopulationFiles(NamedTuple):
    task_count: int
    text_sha256: str
    task_specs_sha256: str
    main_output_dir: str
    specs_output_dir: str


def source_documents(config: SourceTextConfig, rejected: Counter, admission: Counter) -> Iterable[tuple[dict, dict]]:
    source = all_sources()[canonical_source_id(config.source_id)]
    definition = source.config
    if isinstance(definition, PrebuiltCohort):
        with (StoragePath(config.source_input) / definition.files[0]).open("rb") as stream:
            parquet = pq.ParquetFile(stream)
            lowered = "lowered_task_spec" in parquet.schema_arrow.names
            column = "lowered_task_spec" if lowered else "task_spec"
            for batch in parquet.iter_batches(batch_size=32, columns=[column]):
                for encoded in batch.column(0).to_pylist():
                    document = json.loads(encoded)
                    yield (document["task"], document["runtime"]) if lowered else (document, {})
        return
    recipe_config = definition.recipe if isinstance(definition, EvalDataset) else definition
    if not isinstance(recipe_config, CurationRecipe):
        raise ValueError(f"Source has no registered TaskSpec converter: {config.source_id}")
    grader = None
    if recipe_config.grader is not None:
        grader = (
            environment_requirements(recipe_config.grader)
            if recipe_config.grader.image is not None
            else EnvironmentRequirements(
                command_semantics=CommandSemantics.LINUX_PROCESS,
                packages_lock=planned_lock_url(recipe_config.grader),
            )
        )
    recipe = source_recipe(recipe_config, config.inputs, grader)
    context = conversion_context(recipe)
    records = (
        row
        for shard in source_shards(config.source_input, recipe.source)
        for row in staged_raw_file_rows(config.source_input, shard, recipe.source, context)
    )
    # This shared converter observer retains TaskSpecs before common admission gates.
    # Harbor's optional config package is needed only when native allocation metadata is present.
    from experiments.post_training.task_curation.daytona_census import converted_records  # noqa: PLC0415

    yield from converted_records(records, recipe, rejected, admission)


def write_text_population(documents: Iterable[tuple[dict, dict]], output_path: str) -> TextPopulationFiles:
    """Retain every task, writing bounded text/spec partitions without deduplication."""
    text_output = prefix_join(output_path, "outputs/main")
    spec_output = prefix_join(output_path, "specs")
    text_digest, spec_digest = hashlib.sha256(), hashlib.sha256()
    seen = set()
    texts, specs = [], []
    size = partition = count = 0

    def flush() -> None:
        nonlocal partition, size
        name = f"part-{partition:06d}.parquet"
        for path, rows, schema in ((text_output, texts, TEXT_SCHEMA), (spec_output, specs, SPEC_SCHEMA)):
            with StoragePath(prefix_join(path, name)).open("wb", auto_mkdir=True) as stream:
                pq.write_table(
                    pa.Table.from_pylist(sorted(rows, key=lambda row: row["id"]), schema=schema),
                    stream,
                    compression="zstd",
                )
        texts.clear()
        specs.clear()
        size = 0
        partition += 1

    for document, runtime in documents:
        task = TaskSpec.model_validate(document)
        if task.id in seen:
            raise ValueError(f"Duplicate TaskSpec ID prevents exact per-task exclusions: {task.id}")
        seen.add(task.id)
        # Preserve case and paragraph boundaries: the shared eval bloom uses this policy.
        # Fingerprints use this same renderer, then normalize its text for similarity ranking.
        text = format_conversation(task.context.events)
        serialized = task.model_dump_json()
        text_digest.update(json.dumps([task.id, text], ensure_ascii=False).encode())
        spec_digest.update(serialized.encode())
        texts.append({"id": task.id, "text": text, "partition_id": partition})
        specs.append({"id": task.id, "task_spec": serialized, "runtime": json.dumps(runtime)})
        count += 1
        size += len(text.encode()) + len(serialized.encode())
        if size >= PARTITION_BYTES or len(texts) >= PARTITION_ROWS:
            flush()
    if texts:
        flush()
    return TextPopulationFiles(count, text_digest.hexdigest(), spec_digest.hexdigest(), text_output, spec_output)


def normalize_source(config: SourceTextConfig) -> SourceTextPopulation:
    source = all_sources()[canonical_source_id(config.source_id)]
    rejected, admission = Counter(), Counter()
    count, text_sha, spec_sha, texts, specs = write_text_population(
        source_documents(config, rejected, admission), config.output_path
    )
    if count == 0:
        raise ValueError(f"Registered converter emitted no TaskSpecs: {source.info.id}; {dict(rejected)}")
    dataset = source.dataset
    assert dataset is not None and dataset.revision is not None
    return SourceTextPopulation(
        path=config.output_path,
        source_id=source.info.id,
        dataset_id=dataset.name,
        dataset_revision=dataset.revision,
        source_version=source.version,
        converter_revision=config.converter_revision,
        text_renderer_version=RENDERER_VERSION,
        task_count=count,
        text_sha256=text_sha,
        task_specs_sha256=spec_sha,
        main_output_dir=texts,
        specs_output_dir=specs,
        converter_rejections=dict(rejected),
        admission_rejections=dict(admission),
    )


def text_config(
    source_id: str, primary: ArtifactStep, inputs: dict[str, ArtifactStep], converter_revision: str, ctx: StepContext
) -> SourceTextConfig:
    return SourceTextConfig(
        original_source_id(source_id),
        ctx.artifact_path(primary),
        {name: ctx.artifact_path(step) for name, step in inputs.items()},
        ctx.output_path,
        converter_revision,
    )


def text_steps(
    source_ids: list[str], runtime: CampaignRuntime, converter_revision: str, artifact_version: str
) -> list[ArtifactStep]:
    """Bind registered converters to their existing acquisition artifacts."""
    sources = all_sources()
    steps = []
    for source_id in source_ids:
        source_id = canonical_source_id(source_id)
        source = sources[source_id]
        config = source.config.recipe if isinstance(source.config, EvalDataset) else source.config
        if isinstance(config, PrebuiltCohort):
            primary = download_step(
                HfSource(config.repo, config.revision, config.files, format=SourceFormat.PARQUET), runtime
            )
            inputs = {}
        elif isinstance(config, CurationRecipe):
            primary = download_step(config.source, runtime)
            inputs = {name: download_step(reference, runtime) for name, reference in config.inputs.items()}
        else:
            raise ValueError(f"No registered converter for {source_id}")
        key = canonical_sha256(original_source_id(source_id))[:20]
        steps.append(
            ArtifactStep(
                name=f"rl-data/atlas/lexical/text/{key}",
                version=artifact_version,
                artifact_type=SourceTextPopulation,
                deps=[primary, *inputs.values()],
                run=normalize_source,
                build_config=partial(text_config, source_id, primary, inputs, converter_revision),
            )
        )
    return steps


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-ids", type=Path, required=True)
    parser.add_argument("--converter-revision", required=True)
    parser.add_argument("--artifact-version", required=True)
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-workers", type=int, required=True)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    configure_coreweave_s3()
    with artifact_prefix(args.artifact_root):
        source_ids = [canonical_source_id(value) for value in json.loads(args.source_ids.read_text())]
        runtime = CampaignRuntime()
        steps = text_steps(source_ids, runtime, args.converter_revision, args.artifact_version)
        plan = [
            {"source_id": source_id, "path": step.path(), "fingerprint": step.fingerprint()}
            for source_id, step in zip(source_ids, steps, strict=True)
        ]
        if not args.run:
            args.output.write_text(json.dumps(plan, indent=2) + "\n")
            return
        with ExitStack() as cleanup:
            client = LocalClient()
            cleanup.callback(client.shutdown)
            cleanup.enter_context(set_current_client(client))
            with ZephyrContext(
                client=client, max_workers=args.max_workers, resources=ResourceConfig(cpu=2, ram="4g")
            ) as context:
                with runtime.activate(context):
                    results = run(*steps, max_concurrent=2)
        args.output.write_text(json.dumps([result.model_dump(mode="json") for result in results], indent=2) + "\n")


if __name__ == "__main__":
    main()
