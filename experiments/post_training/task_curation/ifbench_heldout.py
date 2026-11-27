# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Measure IFBench training tasks against the complete pinned IFBench test prompts."""

import argparse
import json
import logging
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import pyarrow.parquet as pq
from finestore.rl.licenses import require_license
from marin.datakit.decon import build_eval_bloom, merge_eval_blooms
from marin.execution.artifact import Artifact
from marin.execution.lazy import ArtifactStep, run
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath
from taskcompendium.evaluation.corpus import file_sha256
from taskcompendium.pipeline.inputs import SourceFormat
from zephyr.context import ZephyrContext

from experiments.evaluation.benchmarks.instruction_following import IFBENCH_REVISION
from experiments.post_training.task_curation.campaign import CampaignRuntime
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
from experiments.post_training.task_curation.pipeline import HfSource, download_step
from infra.marina.apps.rl_data_catalog.server.licensing import require_unbanned

EVAL_SOURCE_ID = "Evalchemy:ifbench"
TRAIN_SOURCE_ID = "AllenAI:IF_multi_constraints_upto5"
EVAL_DATASET = "allenai/IFBench_test"
EVAL_FILE = "data/train-00000-of-00001.parquet"
EXPECTED_TEST_TASKS = 300
REFERENCE_VERSION = "atlas-eval-coverage-v2+ifbench-test-2e8a48de-v1"
BASE_BLOOM = "s3://marin-us-east-02a/marin/datakit/bloom/_combined_fixed_3a94b52b"
BASE_FILTER_SHA256 = "70fbe0bfd4b7ab6018461799605cb0609ae22e9043882f1fad7cf51d5730d817"
BASE_INDEX_SHA256 = "7a97553a278d1af774b0a724102414299d41bbd084f402b73190090dee6bab4b"
BASE_DROP = (
    "s3://marin-us-east-02a/marin/rl-data/atlas/lexical/scan/33c2822241770463e486/"
    "report/2026.11.15.203/525249f36c340614a89d/drop_manifest.parquet"
)
BASE_DROP_SHA256 = "7aa74b4714bcf047a02ecf3573d854bfebceb34215a18c21abd10f11323ccfd7"


@dataclass(frozen=True)
class ReferenceConfig:
    input_path: str
    output_path: str


@dataclass(frozen=True)
class BloomConfig:
    input_path: str
    output_path: str
    estimated_doc_count: int
    false_positive_rate: float


@dataclass(frozen=True)
class MergeBloomConfig:
    base_path: str
    supplementary_path: str
    base_filter_sha256: str
    base_index_sha256: str
    output_path: str


def prepare_reference(config: ReferenceConfig) -> Artifact:
    """Preserve every benchmark key and its exact evaluated user prompt."""
    source = StoragePath(config.input_path) / EVAL_FILE
    output = StoragePath(config.output_path)
    keys = set()
    with source.open("rb") as stream, (output / "prompts.jsonl").open("w", auto_mkdir=True) as target:
        for batch in pq.ParquetFile(stream).iter_batches(columns=["key", "prompt"]):
            for row in batch.to_pylist():
                key = str(row["key"])
                prompt = row["prompt"]
                if key in keys or not isinstance(prompt, str) or not prompt.strip():
                    raise ValueError("Held-out reference has duplicate keys or empty prompts")
                keys.add(key)
                target.write(json.dumps({"id": "ifbench-test:" + key, "text": prompt}) + "\n")
    if len(keys) != EXPECTED_TEST_TASKS:
        raise ValueError(f"Expected the complete {EXPECTED_TEST_TASKS}-task test split, got {len(keys)}")
    manifest = {
        "source_id": EVAL_SOURCE_ID,
        "dataset_id": EVAL_DATASET,
        "revision": IFBENCH_REVISION,
        "input_sha256": file_sha256(source),
        "records": len(keys),
        "text_field": "prompt",
        "extraction": "exact user prompt used by convert_instructions; no grader data or outputs",
        "prompts_sha256": file_sha256(output / "prompts.jsonl"),
    }
    (output / "reference-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return Artifact(path=config.output_path)


def prepare_bloom(config: BloomConfig, *, campaign: CampaignRuntime) -> EvalReferenceBloom:
    bloom = build_eval_bloom(
        eval_data_sources=[str(StoragePath(config.input_path) / "prompts.jsonl")],
        output_path=config.output_path,
        ngram=NGRAM,
        estimated_doc_count=config.estimated_doc_count,
        false_positive_rate=config.false_positive_rate,
        worker_resources=WORKER_RESOURCES,
        max_workers=4,
        zephyr_context=campaign.context,
    )
    if bloom.n_eval_records != EXPECTED_TEST_TASKS:
        raise ValueError("The standard feature policy did not cover every held-out prompt")
    receipt = {
        "reference_manifest": str(StoragePath(config.input_path) / "reference-manifest.json"),
        "reference_manifest_sha256": file_sha256(StoragePath(config.input_path) / "reference-manifest.json"),
        "bloom": bloom.model_dump(mode="json"),
        "filter_sha256": file_sha256(StoragePath(str(bloom.bloom_path))),
        "index_sha256": file_sha256(StoragePath(str(bloom.eval_hash_index_path))),
    }
    (StoragePath(config.output_path) / "bloom-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return EvalReferenceBloom(path=config.output_path, **bloom.model_dump())


def merge_reference(config: MergeBloomConfig) -> EvalReferenceBloom:
    for name, expected in (
        ("filter.bin", config.base_filter_sha256),
        ("eval_hash_index.parquet", config.base_index_sha256),
    ):
        if file_sha256(StoragePath(config.base_path) / "_bloom" / name) != expected:
            raise ValueError("Frozen original evaluation Bloom changed: " + name)
    merged = merge_eval_blooms(
        per_eval_bloom_dirs=[config.base_path, config.supplementary_path], output_path=config.output_path
    )
    receipt = {
        "base_bloom": config.base_path,
        "base_filter_sha256": config.base_filter_sha256,
        "base_index_sha256": config.base_index_sha256,
        "supplementary_bloom": config.supplementary_path,
        "filter_sha256": file_sha256(StoragePath(str(merged.bloom_path))),
        "index_sha256": file_sha256(StoragePath(str(merged.eval_hash_index_path))),
        "bloom": merged.model_dump(mode="json"),
    }
    (StoragePath(config.output_path) / "merge-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return EvalReferenceBloom(path=config.output_path, **merged.model_dump())


def heldout_steps(
    populations: list[SourceTextPopulation], version: str, runtime: CampaignRuntime, max_workers: int
) -> ArtifactStep[LexicalMeasurements]:
    """Use the complete frozen DF cohort and report only the IFBench training population."""
    root = "rl-data/atlas/ifbench-heldout"
    downloaded = download_step(HfSource(EVAL_DATASET, IFBENCH_REVISION, (EVAL_FILE,), SourceFormat.PARQUET), runtime)
    reference = ArtifactStep(
        name=root + "/reference",
        version=version,
        artifact_type=Artifact,
        deps=[downloaded],
        run=prepare_reference,
        build_config=lambda ctx: ReferenceConfig(ctx.artifact_path(downloaded), ctx.output_path),
    )
    supplementary = ArtifactStep(
        name=root + "/bloom",
        version=version,
        artifact_type=EvalReferenceBloom,
        deps=[reference],
        run=partial(prepare_bloom, campaign=runtime),
        build_config=lambda ctx: BloomConfig(ctx.artifact_path(reference), ctx.output_path, 50_000_000, 1e-9),
    )
    base = ArtifactStep.adopt(root + "/base-bloom", version, BASE_BLOOM)
    bloom = ArtifactStep(
        name=root + "/merged-bloom",
        version=version,
        artifact_type=EvalReferenceBloom,
        deps=[base, supplementary],
        run=merge_reference,
        build_config=lambda ctx: MergeBloomConfig(
            ctx.artifact_path(base),
            ctx.artifact_path(supplementary),
            BASE_FILTER_SHA256,
            BASE_INDEX_SHA256,
            ctx.output_path,
        ),
    )
    inputs = [
        ArtifactStep.adopt(root + "/input/" + population_key(p.source_id), version, p.path, kind=SourceTextPopulation)
        for p in populations
    ]
    selected = [handle for p, handle in zip(populations, inputs, strict=True) if p.source_id == TRAIN_SOURCE_ID]
    if len(selected) != 1:
        raise ValueError("Frozen DF cohort must contain IFBench training exactly once")
    (train,) = selected
    drops = ArtifactStep(
        name=root + "/df",
        version=version,
        artifact_type=LexicalDropSets,
        deps=[*inputs, bloom],
        runtime_args={"max_workers": max_workers},
        run=partial(build_drop_sets, campaign=runtime),
        build_config=lambda ctx: DropConfig(
            [ctx.artifact_path(handle) for handle in inputs],
            ctx.artifact_path(bloom),
            ctx.output_path,
            ctx.runtime_arg("max_workers"),
        ),
    )
    marks = ArtifactStep(
        name=root + "/marks",
        version=version,
        artifact_type=SourceLexicalMarks,
        deps=[train, drops, bloom],
        runtime_args={"max_workers": max_workers},
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
        name=root + "/report",
        version=version,
        artifact_type=LexicalMeasurements,
        deps=[marks, drops, bloom],
        run=measure_combined_marks,
        build_config=lambda ctx: ReportConfig(
            [ctx.artifact_path(marks)],
            ctx.artifact_path(drops),
            ctx.artifact_path(bloom),
            REFERENCE_VERSION,
            "retained",
            ctx.output_path,
        ),
    )


def compare_original_drops(result: LexicalMeasurements) -> None:
    base = StoragePath(BASE_DROP)
    if file_sha256(base) != BASE_DROP_SHA256:
        raise ValueError("Original shared-scan IFBench drops changed")
    with base.open("rb") as stream:
        original = {row["id"]: row for row in pq.read_table(stream).to_pylist()}
    with StoragePath(result.drop_manifests[TRAIN_SOURCE_ID]).open("rb") as stream:
        combined = {row["id"]: row for row in pq.read_table(stream).to_pylist()}
    if len(original) != 14 or not original.keys() <= combined.keys():
        raise ValueError("Combined reference lost an original IFBench exclusion")
    for key, row in original.items():
        if not set(row["matched_eval_ids"]) <= set(combined[key]["matched_eval_ids"]):
            raise ValueError("Combined reference lost original causal evaluation IDs")
    comparison = {
        "original_drop_uri": BASE_DROP,
        "original_drop_sha256": BASE_DROP_SHA256,
        "original_exclusions": len(original),
        "combined_exclusions": len(combined),
        "newly_flagged_rows": [combined[key] for key in sorted(combined.keys() - original.keys())],
        "all_original_drops_and_eval_ids_preserved": True,
    }
    (StoragePath(result.path) / "original-drop-comparison.json").write_text(json.dumps(comparison, indent=2) + "\n")


def measure_combined_marks(config: ReportConfig) -> LexicalMeasurements:
    result = measure_marks(config)
    compare_original_drops(result)
    return result


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--populations", type=Path, required=True)
    parser.add_argument("--populations-sha256", required=True)
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--artifact-version", required=True)
    parser.add_argument("--max-workers", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    receipt = read_license_receipts(LICENSE_RECEIPT_INDEX)[EVAL_SOURCE_ID]
    require_license(receipt, EVAL_SOURCE_ID, "eval", ban=None)
    require_unbanned(EVAL_SOURCE_ID, EVAL_DATASET, receipt)
    if file_sha256(StoragePath(str(args.populations))) != args.populations_sha256:
        raise ValueError("Frozen shared DF population inventory changed")
    populations = [SourceTextPopulation.model_validate(row) for row in json.loads(args.populations.read_text())]
    if len({p.source_id for p in populations}) != len(populations):
        raise ValueError("Frozen shared DF population inventory has duplicate sources")
    (train,) = [p for p in populations if p.source_id == TRAIN_SOURCE_ID]
    require_license(read_license_receipts(LICENSE_RECEIPT_INDEX)[TRAIN_SOURCE_ID], TRAIN_SOURCE_ID, "train", ban=None)
    require_unbanned(TRAIN_SOURCE_ID, train.dataset_id)
    configure_coreweave_s3()
    runtime = CampaignRuntime()
    with artifact_prefix(args.artifact_root):
        report = heldout_steps(populations, args.artifact_version, runtime, args.max_workers)
        if not args.run:
            args.output.write_text(
                json.dumps(
                    {"report": report.path(), "fingerprint": report.fingerprint(), "df_populations": len(populations)},
                    indent=2,
                )
                + "\n"
            )
            return
        with (
            local_client() as client,
            ZephyrContext(client=client, max_workers=args.max_workers, resources=WORKER_RESOURCES) as context,
            runtime.activate(context),
        ):
            result = run(report, max_concurrent=2)[0]
        args.output.write_text(result.model_dump_json(indent=2) + "\n")


if __name__ == "__main__":
    main()
