# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Measure PDBThink training against its complete native validation and test prompts."""

import argparse
import base64
import json
import logging
from collections import Counter
from dataclasses import dataclass
from functools import partial

import pyarrow.parquet as pq
from finestore.rl.licenses import require_license
from marin.datakit.decon import build_eval_bloom
from marin.execution.artifact import Artifact
from marin.execution.lazy import ArtifactStep, run
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath
from taskcompendium.evaluation.corpus import file_sha256
from taskcompendium.models import TextMessage, format_conversation
from taskcompendium.pipeline.inputs import ConversionContext
from taskcompendium.request_budget import tool_free_prompt_sha256
from zephyr.context import ZephyrContext

from experiments.post_training.task_curation.campaign import CampaignRuntime
from experiments.post_training.task_curation.datasets.pdbthink.tasks import DATASET, REVISION
from experiments.post_training.task_curation.datasets.tasktrove.conversion.archive import unpack_task_binary
from experiments.post_training.task_curation.ifbench_heldout import (
    BASE_BLOOM,
    BASE_FILTER_SHA256,
    BASE_INDEX_SHA256,
    MergeBloomConfig,
    merge_reference,
)
from experiments.post_training.task_curation.lexical_artifacts import (
    EvalReferenceBloom,
    LexicalDropSets,
    LexicalMeasurements,
    SourceLexicalMarks,
    SourceTextPopulation,
)
from experiments.post_training.task_curation.lexical_scan import (
    NGRAM,
    WORKER_RESOURCES,
    DropConfig,
    MarkConfig,
    ReportConfig,
    build_drop_sets,
    mark_population,
    measure_marks,
    population_key,
)
from experiments.post_training.task_curation.license_binding import LICENSE_RECEIPT_INDEX, read_license_receipts
from experiments.post_training.task_curation.local import artifact_prefix, local_client
from infra.marina.apps.rl_data_catalog.server.licensing import require_unbanned

SOURCE = "OpenAthena:pdbthink-coordinate-tasks"
ROOT = "rl-data/atlas/pdbthink-heldout"
EXPECTED_SPLITS = (("validation", 4411), ("test", 4435))
REFERENCE_VERSION = "pdbthink-native-heldout-3734406c-v1"


@dataclass(frozen=True)
class ReferenceConfig:
    manifest_path: str
    manifest_sha256: str
    expected_splits: tuple[tuple[str, int], ...]
    output_path: str


@dataclass(frozen=True)
class BloomConfig:
    reference_path: str
    expected_records: int
    max_workers: int
    output_path: str


def prepare_reference(config: ReferenceConfig) -> Artifact:
    """Render all held-out public prompts with the same renderer as the TRAIN population."""
    manifest = StoragePath(config.manifest_path)
    if file_sha256(manifest) != config.manifest_sha256:
        raise ValueError("Pinned PDBThink raw inventory changed")
    (snapshot,) = json.loads(manifest.read_text())
    if (snapshot["source_id"], snapshot["dataset_id"], snapshot["revision"]) != (SOURCE, DATASET, REVISION):
        raise ValueError("Held-out reference belongs to another source release")
    output = StoragePath(config.output_path)
    counts: Counter[str] = Counter()
    identifiers = set()
    files = []
    context = ConversionContext(inputs={}, grader_environment=None)
    with (output / "prompts.jsonl").open("w", auto_mkdir=True) as target:
        for file in sorted(snapshot["files"], key=lambda item: item["path"]):
            if not any(file["path"].startswith(f"data/{split}-") for split, _ in config.expected_splits):
                continue
            source = StoragePath(file["uri"])
            if file_sha256(source) != file["sha256"]:
                raise ValueError(f"Pinned held-out shard changed: {file['path']}")
            files.append(file)
            with source.open("rb") as stream:
                for batch in pq.ParquetFile(stream).iter_batches(
                    batch_size=4, columns=["path", "split", "task_binary", "prompt_sha256"]
                ):
                    for row in batch.to_pylist():
                        split = row["split"]
                        if split not in dict(config.expected_splits) or not file["path"].startswith(f"data/{split}-"):
                            raise ValueError("Held-out shard includes another split")
                        native = unpack_task_binary(row, context)
                        if native["archive_links"]:
                            raise ValueError("PDBThink held-out package contains archive links")
                        prompt = json.loads(base64.b64decode(native["files"]["prompt.json"], validate=True))
                        if set(prompt) != {"system_prompt", "user_prompt"}:
                            raise ValueError("Held-out package has unexpected prompt fields")
                        messages = tuple(
                            {"role": role, "content": prompt[field]}
                            for role, field in (("system", "system_prompt"), ("user", "user_prompt"))
                        )
                        if tool_free_prompt_sha256(messages) != row["prompt_sha256"]:
                            raise ValueError("Held-out public prompt differs from its released hash")
                        if native["instruction"] != prompt["system_prompt"] + "\n\n" + prompt["user_prompt"]:
                            raise ValueError("Held-out instruction differs from prompt.json")
                        identifier = f"pdbthink-heldout:{split}:{row['path']}"
                        if identifier in identifiers:
                            raise ValueError("Held-out reference repeats a task identity")
                        identifiers.add(identifier)
                        counts[split] += 1
                        target.write(
                            json.dumps(
                                {
                                    "id": identifier,
                                    "text": format_conversation(tuple(TextMessage(**message) for message in messages)),
                                    "split": split,
                                    "prompt_sha256": row["prompt_sha256"],
                                    "archive_sha256": native["archive_sha256"],
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
    if dict(counts) != dict(config.expected_splits):
        raise ValueError(f"Held-out reference is incomplete: {dict(counts)}")
    receipt = {
        "source_id": SOURCE,
        "dataset_id": DATASET,
        "revision": REVISION,
        "raw_inventory": {"uri": config.manifest_path, "sha256": config.manifest_sha256},
        "files": files,
        "split_counts": dict(counts),
        "records": sum(counts.values()),
        "extraction": "Complete released system/user prompts; TRAIN format_conversation renderer; no answers",
        "prompts_sha256": file_sha256(output / "prompts.jsonl"),
    }
    (output / "reference-manifest.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return Artifact(path=config.output_path)


def prepare_bloom(config: BloomConfig, *, campaign: CampaignRuntime) -> EvalReferenceBloom:
    bloom = build_eval_bloom(
        eval_data_sources=[str(StoragePath(config.reference_path) / "prompts.jsonl")],
        output_path=config.output_path,
        ngram=NGRAM,
        estimated_doc_count=50_000_000,
        false_positive_rate=1e-9,
        worker_resources=WORKER_RESOURCES,
        max_workers=config.max_workers,
        zephyr_context=campaign.context,
    )
    if bloom.n_eval_records != config.expected_records:
        raise ValueError("Bloom does not cover the complete native held-out reference")
    receipt = {
        "reference_manifest": str(StoragePath(config.reference_path) / "reference-manifest.json"),
        "reference_manifest_sha256": file_sha256(StoragePath(config.reference_path) / "reference-manifest.json"),
        "bloom": bloom.model_dump(mode="json"),
        "filter_sha256": file_sha256(StoragePath(str(bloom.bloom_path))),
        "index_sha256": file_sha256(StoragePath(str(bloom.eval_hash_index_path))),
    }
    (StoragePath(config.output_path) / "bloom-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return EvalReferenceBloom(path=config.output_path, **bloom.model_dump())


def reference_steps(
    manifest: str, digest: str, version: str, runtime: CampaignRuntime, workers: int
) -> ArtifactStep[EvalReferenceBloom]:
    raw = ArtifactStep.adopt(ROOT + "/raw-inventory", version, manifest)
    reference = ArtifactStep(
        name=ROOT + "/reference",
        version=version,
        artifact_type=Artifact,
        deps=[raw],
        run=prepare_reference,
        build_config=lambda ctx: ReferenceConfig(ctx.artifact_path(raw), digest, EXPECTED_SPLITS, ctx.output_path),
    )
    return ArtifactStep(
        name=ROOT + "/bloom",
        version=version,
        artifact_type=EvalReferenceBloom,
        deps=[reference],
        runtime_args={"max_workers": workers},
        run=partial(prepare_bloom, campaign=runtime),
        build_config=lambda ctx: BloomConfig(
            ctx.artifact_path(reference),
            sum(dict(EXPECTED_SPLITS).values()),
            ctx.runtime_arg("max_workers"),
            ctx.output_path,
        ),
    )


def scan_steps(
    populations: list[SourceTextPopulation],
    bloom: ArtifactStep[EvalReferenceBloom],
    version: str,
    runtime: CampaignRuntime,
    workers: int,
) -> list[ArtifactStep[LexicalMeasurements]]:
    inputs = [
        ArtifactStep.adopt(ROOT + "/input/" + population_key(p.source_id), version, p.path, kind=SourceTextPopulation)
        for p in populations
    ]
    (train,) = [handle for p, handle in zip(populations, inputs, strict=True) if p.source_id == SOURCE]
    base = ArtifactStep.adopt(ROOT + "/base-bloom", version, BASE_BLOOM)
    merged = ArtifactStep(
        name=ROOT + "/merged-bloom",
        version=version,
        artifact_type=EvalReferenceBloom,
        deps=[base, bloom],
        run=merge_reference,
        build_config=lambda ctx: MergeBloomConfig(
            ctx.artifact_path(base), ctx.artifact_path(bloom), BASE_FILTER_SHA256, BASE_INDEX_SHA256, ctx.output_path
        ),
    )
    drops = ArtifactStep(
        name=ROOT + "/df",
        version=version,
        artifact_type=LexicalDropSets,
        deps=[*inputs, merged],
        runtime_args={"max_workers": workers},
        run=partial(build_drop_sets, campaign=runtime),
        build_config=lambda ctx: DropConfig(
            [ctx.artifact_path(handle) for handle in inputs],
            ctx.artifact_path(merged),
            ctx.output_path,
            ctx.runtime_arg("max_workers"),
        ),
    )
    return [
        report_step(train, drops, reference, name, corpus_version, version, runtime, workers)
        for reference, name, corpus_version in (
            (bloom, "native", REFERENCE_VERSION),
            (merged, "combined", "atlas-eval-coverage-v2+" + REFERENCE_VERSION),
        )
    ]


def report_step(
    train: ArtifactStep[SourceTextPopulation],
    drops: ArtifactStep[LexicalDropSets],
    bloom: ArtifactStep[EvalReferenceBloom],
    name: str,
    corpus_version: str,
    version: str,
    runtime: CampaignRuntime,
    workers: int,
) -> ArtifactStep[LexicalMeasurements]:
    marks = ArtifactStep(
        name=ROOT + "/" + name + "-marks",
        version=version,
        artifact_type=SourceLexicalMarks,
        deps=[train, drops, bloom],
        runtime_args={"max_workers": workers},
        run=partial(mark_population, campaign=runtime),
        build_config=lambda ctx: MarkConfig(
            ctx.artifact_path(train),
            ctx.artifact_path(drops),
            ctx.artifact_path(bloom),
            ctx.output_path,
            ctx.runtime_arg("max_workers"),
        ),
    )
    return ArtifactStep(
        name=ROOT + "/" + name + "-report",
        version=version,
        artifact_type=LexicalMeasurements,
        deps=[marks, drops, bloom],
        run=measure_marks,
        build_config=lambda ctx: ReportConfig(
            [ctx.artifact_path(marks)],
            ctx.artifact_path(drops),
            ctx.artifact_path(bloom),
            corpus_version,
            "retained",
            ctx.output_path,
        ),
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("reference", "scan"), required=True)
    parser.add_argument("--raw-inventory", required=True)
    parser.add_argument("--raw-inventory-sha256", required=True)
    parser.add_argument("--populations")
    parser.add_argument("--populations-sha256")
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--artifact-version", required=True)
    parser.add_argument("--max-workers", type=int, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    configure_coreweave_s3()
    license_receipt = read_license_receipts(LICENSE_RECEIPT_INDEX)[SOURCE]
    require_license(license_receipt, SOURCE, DATASET, REVISION, "eval", ban=None)
    require_unbanned(SOURCE, DATASET, license_receipt, "eval")
    runtime = CampaignRuntime()
    with artifact_prefix(args.artifact_root):
        step = reference_steps(
            args.raw_inventory, args.raw_inventory_sha256, args.artifact_version, runtime, args.max_workers
        )
        steps: list[ArtifactStep] = [step]
        if args.stage == "scan":
            if not args.populations or not args.populations_sha256:
                parser.error("scan requires a frozen population inventory and SHA-256")
            path = StoragePath(args.populations)
            if file_sha256(path) != args.populations_sha256:
                raise ValueError("Frozen full shared population inventory changed")
            populations = [SourceTextPopulation.model_validate(row) for row in json.loads(path.read_text())]
            if len({p.source_id for p in populations}) != len(populations):
                raise ValueError("Frozen population inventory repeats a source")
            steps = scan_steps(populations, step, args.artifact_version, runtime, args.max_workers)
        if not args.run:
            print(json.dumps([{"path": item.path(), "fingerprint": item.fingerprint()} for item in steps]))
            return
        with (
            local_client() as client,
            ZephyrContext(client=client, max_workers=args.max_workers, resources=WORKER_RESOURCES) as context,
            runtime.activate(context),
        ):
            results = run(*steps)
    output = StoragePath(args.output)
    if output.exists():
        raise FileExistsError(str(output))
    output.write_text(
        json.dumps([result.model_dump(mode="json") for result in results], indent=2) + "\n", auto_mkdir=True
    )


if __name__ == "__main__":
    main()
