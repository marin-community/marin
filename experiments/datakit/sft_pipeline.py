# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Deduplicate and decontaminate rendered SFT sources with the Datakit reference policy.

Run ``python -m experiments.datakit.sft_pipeline --sources superior-reasoning``
from an environment configured for the data's region. Omit --sources to process
all chat sources. The versioned eval corpus must already be prepared in-region.

The graph compares user, assistant, and tool message bodies, excluding system and
developer instructions and template-injected tool definitions. Exact dedup uses
full structured-chat identity. Fuzzy dedup and eval decontamination use message
bodies separated by blank lines. Filtered outputs retain full rendered conversations.
Deduplication spans only the selected SFT sources. Outputs are NormalizedData
artifacts suitable for downstream tokenization.
"""

import argparse
import logging
from dataclasses import dataclass, replace

import pyarrow as pa
from marin.datakit.chat_normalize import message_text
from marin.datakit.chat_render import CHAT_RENDER_VERSION, render_chat_record
from marin.datakit.chat_template import MARIN_CHAT_TEMPLATE
from marin.datakit.decon import DeconAttributes, build_eval_bloom_step, decon_step
from marin.datakit.normalize import NormalizedData
from marin.datakit.sft_sources import all_sft_sources
from marin.datakit.source_key import datakit_source_key
from marin.execution.artifact import read_artifact
from marin.execution.step_runner import StepRunner
from marin.execution.step_spec import StepSpec
from marin.processing.classification.consolidate import FilterConfig, FilterType, consolidate
from marin.processing.classification.deduplication.fuzzy_dups import FuzzyDupsAttrData
from marin.processing.classification.deduplication.fuzzy_minhash import MinHashAttrData
from marin.processing.classification.deduplication.fuzzy_verification import FuzzyVerificationParams
from marin.processing.classification.deduplication.verify_fuzzy_dups import (
    REFERENCE_LOCAL_REPRESENTATIVE_PARAMS,
    VERIFIED_FUZZY_DUPS_ATTR_DATA_VERSION,
    FuzzyVerificationStoreConfig,
    VerifiedFuzzyDupsAttrData,
    verify_fuzzy_dups,
)
from openai_harmony import Message, Role
from rigging.filesystem.storage_path import prefix_join
from rigging.log_setup import configure_logging
from zephyr.context import ZephyrContext
from zephyr.dataset import Dataset
from zephyr.readers import load_parquet
from zephyr.runners import SubprocessRunner

from experiments.datakit.global_exact_dedup import GlobalExactDedupData
from experiments.datakit.reference_pipeline import (
    AA_BENCHMARK_NAMES,
    AA_MANIFEST_PATH,
    DECON_EXCLUDED_EVAL_TASKS,
    DEFAULT_SCALE,
    ESTIMATED_DOC_COUNT,
    EVAL_CORPUS_VERSION,
    EVAL_ROOT,
    FALSE_POSITIVE_RATE,
    FLAGGED_SAMPLE_SIZE,
    LMH_MANIFEST_PATH,
    NGRAM_LENGTH,
    OVERLAP_THRESHOLD,
    PipelineScale,
    zephyr_datakit_steps,
)
from experiments.datakit.reports.decontam import decontam_report
from experiments.datakit.reports.dedup import dedup_report


@dataclass(frozen=True)
class SftSteps:
    """Comparison inputs, filtered training text, and runnable stages for SFT."""

    sources: dict[str, StepSpec]
    filtered: dict[str, StepSpec]
    all_steps: list[StepSpec]


def select_sft_sources(names: list[str] | None = None) -> dict[str, StepSpec]:
    """Select normalized structured chat sources; None selects the full registry."""
    registry = all_sft_sources()
    selected = list(registry) if names is None else names
    unknown = sorted(set(selected) - registry.keys())
    if unknown:
        raise KeyError(f"unknown SFT sources {unknown}; known: {sorted(registry)}")
    if not selected:
        raise ValueError("Select at least one SFT source")
    return {name: registry[name].chat_normalized for name in sorted(selected)}


_COMPARISON_SCHEMA = pa.schema(
    [("id", pa.string()), ("source_id", pa.string()), ("text", pa.string()), ("rendered_text", pa.string())]
)


def chat_comparison_record(record: dict) -> dict:
    """Separate conversation bodies for matching from the full training rendering."""
    messages = [Message.from_dict(message) for message in record["messages"]]
    text = "\n\n".join(
        message_text(message) for message in messages if message.author.role in (Role.USER, Role.ASSISTANT, Role.TOOL)
    )
    return {
        "id": record["id"],
        "source_id": record["id"],
        "text": text,
        "rendered_text": render_chat_record(record)["text"],
    }


def _comparison_source(output_path: str, chat: NormalizedData, scale: PipelineScale) -> NormalizedData:
    main_output_dir = prefix_join(output_path, "outputs/main")
    result = ZephyrContext(name="sft-comparison", resources=scale.pool.worker, max_workers=scale.pool.n_workers).execute(
        Dataset.from_files(prefix_join(chat.main_output_dir, "*.parquet"))
        .flat_map(load_parquet)
        .map(chat_comparison_record)
        .write_parquet(
            prefix_join(main_output_dir, "part-{shard:05d}-of-{total:05d}.parquet"), schema=_COMPARISON_SCHEMA
        )
    )
    return NormalizedData(main_output_dir=main_output_dir, dup_output_dir=chat.dup_output_dir, counters=result.counters)


def _training_record(record: dict) -> dict:
    return {"id": record["id"], "source_id": record["source_id"], "text": record["rendered_text"]}


def filter_sft_source(
    *,
    output_path: str,
    normalized: NormalizedData,
    exact: GlobalExactDedupData,
    verified: VerifiedFuzzyDupsAttrData,
    decontam: DeconAttributes,
    scale: PipelineScale = DEFAULT_SCALE,
) -> NormalizedData:
    """Remove marked conversations and restore their full training text."""
    main_output_dir = prefix_join(output_path, "outputs/main")
    retained_dir = prefix_join(output_path, "retained-comparison")
    result = consolidate(
        input_path=normalized.main_output_dir,
        output_path=retained_dir,
        filetype="parquet",
        filters=[
            FilterConfig(
                type=FilterType.REMOVE_DOC,
                attribute_path=decontam.main_output_dir,
                name="contaminated",
                attribute_filetype="parquet",
            ),
            FilterConfig(
                type=FilterType.REMOVE_DOC,
                attribute_path=exact.sources[datakit_source_key(normalized.main_output_dir)].attr_dir,
                name="dup_doc",
                attribute_filetype="parquet",
                keep_if_missing=True,
            ),
            FilterConfig(
                type=FilterType.REMOVE_DOC,
                attribute_path=verified.attr_dir_for_source(normalized.main_output_dir),
                name="dup_doc",
                attribute_filetype="parquet",
                keep_if_missing=True,
            ),
        ],
        worker_resources=scale.pool.worker,
        max_workers=scale.pool.n_workers,
    )
    rendered = ZephyrContext(
        name="sft-filtered-render", resources=scale.pool.worker, max_workers=scale.pool.n_workers
    ).execute(
        Dataset.from_files(prefix_join(retained_dir, "*.parquet"))
        .flat_map(load_parquet)
        .map(_training_record)
        .write_parquet(
            prefix_join(main_output_dir, "part-{shard:05d}-of-{total:05d}.parquet"),
            schema=pa.schema([("id", pa.string()), ("source_id", pa.string()), ("text", pa.string())]),
        )
    )
    return NormalizedData(
        main_output_dir=main_output_dir,
        # Consolidation does not materialize rejected rows. Retain the original
        # normalizer's duplicate side output; global drop markers are separate artifacts.
        dup_output_dir=normalized.dup_output_dir,
        counters={**result.counters, **{f"render/{key}": value for key, value in rendered.counters.items()}},
    )


def sft_datakit_steps(
    sources: dict[str, StepSpec],
    *,
    scale: PipelineScale = DEFAULT_SCALE,
    zephyr_context: ZephyrContext | None = None,
) -> SftSteps:
    """Build filtering from structured-chat NormalizedData steps, using message bodies for matching."""
    if not sources:
        raise ValueError("Select at least one SFT source")
    sources = {
        name: StepSpec(
            name=f"datakit/sft/comparison/{name}",
            deps=[chat],
            hash_attrs={"v": 1, "render_version": CHAT_RENDER_VERSION, "chat_template": MARIN_CHAT_TEMPLATE},
            fn=lambda output_path, chat=chat: _comparison_source(
                output_path, read_artifact(chat.output_path, NormalizedData), scale
            ),
        )
        for name, chat in sorted(sources.items())
    }
    dedup = zephyr_datakit_steps(sources, scale, zephyr_context)
    bloom = build_eval_bloom_step(
        name="datakit/bloom/_combined_fixed",
        eval_data_sources=[EVAL_ROOT],
        ngram_length=NGRAM_LENGTH,
        overlap_threshold=OVERLAP_THRESHOLD,
        estimated_doc_count=ESTIMATED_DOC_COUNT,
        false_positive_rate=FALSE_POSITIVE_RATE,
        exclude_eval_dirs=DECON_EXCLUDED_EVAL_TASKS,
        required_eval_manifest_path=AA_MANIFEST_PATH,
        required_eval_corpus_version=EVAL_CORPUS_VERSION,
        required_eval_names=AA_BENCHMARK_NAMES,
        best_effort_eval_manifest_path=LMH_MANIFEST_PATH,
        best_effort_eval_corpus_version=EVAL_CORPUS_VERSION,
    )
    # Repeated eval prompts remain contamination even when many SFT rollouts
    # share them. Template boilerplate is removed by the comparison projection.
    decontam = {
        name: decon_step(
            name=f"datakit/decontam/{name}",
            normalized=source,
            prebuilt_bloom=bloom,
            ngram_length=NGRAM_LENGTH,
            overlap_threshold=OVERLAP_THRESHOLD,
            estimated_doc_count=ESTIMATED_DOC_COUNT,
            false_positive_rate=FALSE_POSITIVE_RATE,
            flagged_sample_size=FLAGGED_SAMPLE_SIZE,
            worker_resources=scale.pool.worker,
            zephyr_context=zephyr_context,
        )
        for name, source in sources.items()
    }
    verification_params = FuzzyVerificationParams()
    verification_store_config = FuzzyVerificationStoreConfig(
        recovery_timeout=1_800,
        ready_timeout=1_800,
        lookup_batch_size=128,
    )
    verified = StepSpec(
        name="datakit/verify_fuzzy_dups",
        deps=[*sources.values(), *dedup.minhash.values(), dedup.fuzzy_dedup],
        hash_attrs={
            "artifact_version": VERIFIED_FUZZY_DUPS_ATTR_DATA_VERSION,
            "verification": verification_params.model_dump(mode="json"),
            "local_representatives": REFERENCE_LOCAL_REPRESENTATIVE_PARAMS.model_dump(mode="json"),
        },
        fn=lambda op: verify_fuzzy_dups(
            normalized_sources={name: read_artifact(step.output_path, NormalizedData) for name, step in sources.items()},
            minhash_sources={
                name: read_artifact(step.output_path, MinHashAttrData) for name, step in dedup.minhash.items()
            },
            candidates=read_artifact(dedup.fuzzy_dedup.output_path, FuzzyDupsAttrData),
            output_path=op,
            verification_params=verification_params,
            local_representative_params=REFERENCE_LOCAL_REPRESENTATIVE_PARAMS,
            store_config=verification_store_config,
            max_workers=scale.pool.n_workers,
            worker_resources=scale.pool.worker,
        ),
    )
    filtered = {
        name: StepSpec(
            name=f"datakit/sft/filtered/{name}",
            deps=[normalized, dedup.exact_dedup, verified, decontam[name]],
            hash_attrs={"v": 2},
            fn=lambda output_path, source=normalized, marking=decontam[name]: filter_sft_source(
                output_path=output_path,
                normalized=read_artifact(source.output_path, NormalizedData),
                exact=read_artifact(dedup.exact_dedup.output_path, GlobalExactDedupData),
                verified=read_artifact(verified.output_path, VerifiedFuzzyDupsAttrData),
                decontam=read_artifact(marking.output_path, DeconAttributes),
                scale=scale,
            ),
        )
        for name, normalized in sources.items()
    }
    reports = [
        StepSpec(
            name="datakit/sft/report/decontam",
            deps=list(decontam.values()),
            hash_attrs={"v": 1},
            fn=lambda output_path: decontam_report(
                output_path,
                {name: read_artifact(step.output_path, DeconAttributes) for name, step in decontam.items()},
            ),
        ),
        StepSpec(
            name="datakit/sft/report/dedup",
            deps=[dedup.fuzzy_dedup, verified],
            hash_attrs={"v": 2},
            fn=lambda output_path: dedup_report(
                output_path,
                read_artifact(dedup.fuzzy_dedup.output_path, FuzzyDupsAttrData),
                read_artifact(verified.output_path, VerifiedFuzzyDupsAttrData),
            ),
        ),
    ]
    return SftSteps(
        sources=sources,
        filtered=filtered,
        all_steps=[
            dedup.exact_dedup,
            *dedup.minhash.values(),
            dedup.fuzzy_dedup,
            verified,
            bloom,
            *decontam.values(),
            *filtered.values(),
            *reports,
        ],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", default="all", help="comma-separated SFT source names, or all")
    parser.add_argument("--pool-workers", type=int, default=DEFAULT_SCALE.pool.n_workers)
    parser.add_argument("--pool-cpu", type=float, default=DEFAULT_SCALE.pool.worker.cpu)
    parser.add_argument("--pool-ram", default=DEFAULT_SCALE.pool.worker.ram)
    parser.add_argument("--pool-disk", default=DEFAULT_SCALE.pool.worker.disk)
    parser.add_argument("--max-concurrent", type=int, default=8)
    args = parser.parse_args()
    configure_logging(logging.INFO)
    names = None if args.sources == "all" else [name.strip() for name in args.sources.split(",") if name.strip()]
    sources = select_sft_sources(names)
    worker = replace(DEFAULT_SCALE.pool.worker, cpu=args.pool_cpu, ram=args.pool_ram, disk=args.pool_disk)
    scale = replace(DEFAULT_SCALE, pool=replace(DEFAULT_SCALE.pool, n_workers=args.pool_workers, worker=worker))
    with ZephyrContext(
        name="datakit-sft",
        resources=worker,
        max_workers=scale.pool.n_workers,
        stage_runner_factory=SubprocessRunner,
    ) as context:
        pipeline = sft_datakit_steps(sources, scale=scale, zephyr_context=context)
        StepRunner().run(pipeline.all_steps, max_concurrent=args.max_concurrent)
    for name, step in pipeline.filtered.items():
        logging.info("Filtered SFT source %s: %s", name, step.output_path)


if __name__ == "__main__":
    main()
