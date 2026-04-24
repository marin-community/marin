# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Datakit Testbed fuzzy-dedup variant — non-trivial dedup arm of the ranking protocol.

Shares the sample stage (``build_testbed_steps(run_id, ...)``) with the
baseline and with other fuzzy-dedup variants so one set of sampled
parquet serves every hyperparam sweep. Each variant then MinHash→fuzzy-dups
→consolidates the sampled data with its own fuzzy-dedup parameters,
tokenizes the deduped output, and trains.

Pipeline mirrors ``experiments/ferries/datakit_ferry.py`` but fanned across
every sampled source and with sample (not normalize) as the input.
"""

from __future__ import annotations

import logging
import os

from fray import ResourceConfig
from rigging.filesystem import marin_prefix
from rigging.log_setup import configure_logging

from marin.datakit.normalize import NormalizedData
from marin.execution.artifact import Artifact
from marin.execution.executor import Executor, ExecutorMainConfig, ExecutorStep, executor_main
from marin.execution.step_runner import StepRunner
from marin.execution.step_spec import StepSpec
from marin.processing.classification.consolidate import FilterConfig, FilterType, consolidate
from marin.processing.classification.deduplication.fuzzy_dups import FuzzyDupsAttrData, compute_fuzzy_dups_attrs
from marin.processing.classification.deduplication.fuzzy_minhash import MinHashAttrData, compute_minhash_attrs

from experiments.datakit_testbed.mixture import weights_from_tokenized_bucket_stats
from experiments.datakit_testbed.sampler import build_testbed_steps
from experiments.datakit_testbed.settings import TESTBED_TOKENIZER
from experiments.datakit_testbed.train import run_testbed_config, testbed_tokenize

logger = logging.getLogger(__name__)

STAGING_PREFIX = "gs://marin-us-central1"
TARGET_TOTAL_TOKENS_B = 1000.0

_SAMPLE_STEP_PREFIX = "data/datakit/"
_FUZZY_DUPS_MAX_PARALLELISM = 128
_MINHASH_WORKER_RESOURCES = ResourceConfig(cpu=2, ram="5g")
_FUZZY_DUPS_WORKER_RESOURCES = ResourceConfig(cpu=2, ram="5g")
_CONSOLIDATE_WORKER_RESOURCES = ResourceConfig(cpu=2, ram="5g")


def _minhash_step(src_name: str, sampled: StepSpec, **params: int) -> StepSpec:
    """MinHash bucket attrs for one sampled source."""
    return StepSpec(
        name=f"data/datakit/minhash/{src_name}",
        deps=[sampled],
        hash_attrs={
            "num_perms": params["num_perms"],
            "num_bands": params["num_bands"],
            "ngram_size": params["ngram_size"],
            "seed": params["seed"],
        },
        fn=lambda output_path, sampled=sampled: compute_minhash_attrs(
            source=Artifact.load(sampled, NormalizedData),
            output_path=output_path,
            num_perms=params["num_perms"],
            num_bands=params["num_bands"],
            ngram_size=params["ngram_size"],
            seed=params["seed"],
            worker_resources=_MINHASH_WORKER_RESOURCES,
        ),
    )


def _fuzzy_dups_step(minhash_steps: list[StepSpec], cc_max_iterations: int) -> StepSpec:
    """Global fuzzy-dup cluster attrs across every source's MinHash."""
    return StepSpec(
        name="data/datakit/fuzzy_dups",
        deps=list(minhash_steps),
        hash_attrs={"cc_max_iterations": cc_max_iterations},
        fn=lambda output_path: compute_fuzzy_dups_attrs(
            inputs=[Artifact.load(mh, MinHashAttrData) for mh in minhash_steps],
            output_path=output_path,
            cc_max_iterations=cc_max_iterations,
            max_parallelism=_FUZZY_DUPS_MAX_PARALLELISM,
            worker_resources=_FUZZY_DUPS_WORKER_RESOURCES,
        ),
    )


def _deduped_step(src_name: str, sampled: StepSpec, fuzzy_dups: StepSpec) -> StepSpec:
    """Per-source consolidate: keep the canonical cluster member, drop the rest.

    Writes to ``{output_path}/outputs/main/part-*.parquet`` so the downstream
    tokenize's ``outputs/main/*.parquet`` glob picks it up unchanged.
    """
    return StepSpec(
        name=f"data/datakit/deduped/{src_name}",
        deps=[sampled, fuzzy_dups],
        fn=lambda output_path, sampled=sampled: consolidate(
            input_path=Artifact.load(sampled, NormalizedData).main_output_dir,
            output_path=os.path.join(output_path, "outputs/main"),
            filetype="parquet",
            filters=[
                FilterConfig(
                    type=FilterType.KEEP_DOC,
                    attribute_path=Artifact.load(fuzzy_dups, FuzzyDupsAttrData)
                    .sources[Artifact.load(sampled, NormalizedData).main_output_dir]
                    .attr_dir,
                    name="is_cluster_canonical",
                    attribute_filetype="parquet",
                    keep_if_missing=True,
                ),
            ],
            worker_resources=_CONSOLIDATE_WORKER_RESOURCES,
        ),
    )


def dedup(
    steps: list[StepSpec],
    *,
    name: str,
    tokenizer: str,
    fuzzy_dedup_num_perms: int = 286,
    fuzzy_dedup_num_bands: int = 26,
    fuzzy_dedup_ngram_size: int = 5,
    fuzzy_dedup_seed: int = 42,
    fuzzy_dedup_cc_max_iterations: int = 10,
) -> ExecutorStep:
    """Assemble the fuzzy-dedup training step off a testbed DAG.

    Defaults for ``fuzzy_dedup_*`` match
    :func:`marin.processing.classification.deduplication.fuzzy_minhash.compute_minhash_attrs`
    and :func:`marin.processing.classification.deduplication.fuzzy_dups.compute_fuzzy_dups_attrs`.
    """
    sampled_by_source = {
        s.name.removeprefix(_SAMPLE_STEP_PREFIX): s for s in steps if s.name.startswith(_SAMPLE_STEP_PREFIX)
    }
    if not sampled_by_source:
        raise ValueError("no sample steps found in the DAG (expected names under 'data/datakit/...')")

    minhash_params = {
        "num_perms": fuzzy_dedup_num_perms,
        "num_bands": fuzzy_dedup_num_bands,
        "ngram_size": fuzzy_dedup_ngram_size,
        "seed": fuzzy_dedup_seed,
    }
    minhash_by_source = {
        src_name: _minhash_step(src_name, sampled, **minhash_params) for src_name, sampled in sampled_by_source.items()
    }
    fuzzy_dups = _fuzzy_dups_step(list(minhash_by_source.values()), fuzzy_dedup_cc_max_iterations)
    deduped_by_source = {
        src_name: _deduped_step(src_name, sampled, fuzzy_dups) for src_name, sampled in sampled_by_source.items()
    }

    logger.info(
        "fuzzy-dedup variant %s: %d sources → minhash → fuzzy_dups → consolidate. params=%s, cc_max=%d",
        name,
        len(sampled_by_source),
        minhash_params,
        fuzzy_dedup_cc_max_iterations,
    )

    # Materialize the dedup chain (StepRunner walks transitive deps, so we only
    # need to hand it the terminal consolidates).
    StepRunner().run(list(deduped_by_source.values()))

    # Now the deduped parquet lives at each consolidate step's output_path under
    # outputs/main/ — bridge those into tokenize ExecutorSteps exactly like
    # baseline wires sample outputs.
    tokenized_buckets = {
        src_name: testbed_tokenize(src_name, deduped, tokenizer) for src_name, deduped in deduped_by_source.items()
    }
    prefix = marin_prefix()
    tokenize_executor = Executor(
        prefix=prefix,
        executor_info_base_path=os.path.join(prefix, "experiments"),
    )
    tokenize_executor.run(list(tokenized_buckets.values()))

    resolved_output_paths = {
        bucket_name: tokenize_executor.output_paths[step] for bucket_name, step in tokenized_buckets.items()
    }
    weights = weights_from_tokenized_bucket_stats(resolved_output_paths)
    return run_testbed_config(
        name=name,
        tokenized_buckets=tokenized_buckets,
        weights=weights,
        tokenizer=tokenizer,
    )


def main() -> None:
    """Entry-point: ferry → minhash → fuzzy_dups → consolidate → tokenize → train."""
    os.environ["MARIN_PREFIX"] = STAGING_PREFIX

    tokenizer = TESTBED_TOKENIZER
    run_id = "fuzzy_dedup"

    testbed_steps = build_testbed_steps(run_id, target_total_tokens_b=TARGET_TOTAL_TOKENS_B)
    logger.info("Materializing %d ferry StepSpecs under %s", len(testbed_steps), STAGING_PREFIX)
    StepRunner().run(testbed_steps)

    training_step = dedup(testbed_steps, name=run_id, tokenizer=tokenizer)
    executor_main(ExecutorMainConfig(), [training_step])


if __name__ == "__main__":
    configure_logging()
    main()
