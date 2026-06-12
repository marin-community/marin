# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run David's PPL bundles on every finished swarm candidate's checkpoint.

Mirrors ``swarm_mmlu_eval.py`` but for ``eval_perplexity.run_grug_perplexity_eval``.
For each (candidate, bundle) pair we emit one ``ExecutorStep`` whose output is a
``results.json`` of per-dataset bpb scores. Resources match the swarm itself
(v4-8 preemptible, us-central2-b) so the checkpoint reads stay in-region.
"""

import re
from concurrent.futures import ThreadPoolExecutor

import fsspec
from fray.cluster import ResourceConfig
from marin.execution.executor import ExecutorStep, InputName, executor_main
from marin.execution.remote import remote
from marin.execution.types import this_output_path

from experiments.evals.perplexity_gap_registry import registered_perplexity_gap_bundles
from experiments.grug.moe.eval_perplexity import GrugPerplexityEvalConfig, run_grug_perplexity_eval
from experiments.grug.moe.heuristic import build_from_heuristic
from experiments.grug.moe.swarm_fisher_dsp import _BUDGET as _EVAL_BUDGET, _HIDDEN_DIM, _TARGET_STEPS

_OUTPUT_PREFIX = "gs://marin-us-central2/grug/"
_CANDIDATE_RE = re.compile(r"swarm_fisher_dsp_d512_(\d{6})-[a-f0-9]+")
# Skip base_raw (Paloma + uncheatable already covered by the lm-eval suite).
# Note: multilingual_raw still includes the base_raw datasets as a superset.
_BUNDLES = tuple(b for b in registered_perplexity_gap_bundles() if b.key != "base_raw")
# `refseq_viral_gff` is dead upstream — NCBI removed
# https://ftp.ncbi.nlm.nih.gov/refseq/release/viral/viral.1.gff.gz, so the
# slice's zephyr download can never succeed. `model_perplexity_gap_suite.py`
# already lists it in SKIPPED_DATASETS_FOR_THIS_RUN. The accompanying fasta
# slice has all 1977 parquet shards staged on GCS, so it scores fine after we
# clear the upstream's FAILED marker (data is there; the marker was set only
# because the sibling gff slice failed).
_DATASETS_SKIP = ("bio_chem/refseq/refseq_viral_gff",)


def _find_finished_checkpoints() -> dict[int, str]:
    """Returns ``{idx: checkpoints_subpath}`` for every SUCCESS candidate."""
    fs = fsspec.filesystem("gs")
    cand_dirs = [d for d in fs.ls(_OUTPUT_PREFIX, detail=False) if "swarm_fisher_dsp_d512_" in d]

    def probe(d):
        try:
            with fs.open(f"gs://{d}/.executor_status", "rt") as f:
                return d, f.read().strip() == "SUCCESS"
        except FileNotFoundError:
            return d, False

    with ThreadPoolExecutor(max_workers=64) as ex:
        results = list(ex.map(probe, cand_dirs))

    by_idx: dict[int, str] = {}
    for d, ok in results:
        if not ok:
            continue
        m = _CANDIDATE_RE.search(d)
        if not m:
            continue
        idx = int(m.group(1))
        existing = by_idx.get(idx)
        if existing is None or d > existing:
            by_idx[idx] = d
    return {idx: f"{d.removeprefix('marin-us-central2/')}/checkpoints" for idx, d in by_idx.items()}


_MODEL, _, _, _ = build_from_heuristic(
    budget=_EVAL_BUDGET,
    hidden_dim=_HIDDEN_DIM,
    target_steps=_TARGET_STEPS,
)


def _build_step(idx: int, ckpt_subpath: str, bundle) -> ExecutorStep:
    slug = f"swarm_fisher_dsp_d{_HIDDEN_DIM}_{idx:06d}"
    datasets = {k: v for k, v in bundle.datasets().items() if k not in _DATASETS_SKIP}
    return ExecutorStep(
        name=f"evaluation/grug_ppl/{slug}/{bundle.key}",
        fn=remote(
            run_grug_perplexity_eval,
            resources=ResourceConfig.with_tpu("v4-8", zone="us-central2-b", preemptible=True),
            pip_dependency_groups=["eval", "tpu"],
        ),
        config=GrugPerplexityEvalConfig(
            grug_model_config=_MODEL,
            checkpoint_path=InputName.hardcoded(ckpt_subpath),
            output_path=this_output_path(),
            bundle_key=bundle.key,
            datasets=datasets,
            max_eval_length=bundle.max_eval_length,
            max_docs_per_dataset=bundle.max_docs_per_dataset,
            max_doc_bytes=bundle.max_doc_bytes,
        ),
    )


_FINISHED = _find_finished_checkpoints()
print(
    f"swarm_ppl_eval: {len(_FINISHED)} finished candidates × {len(_BUNDLES)} bundles "
    f"= {len(_FINISHED) * len(_BUNDLES)} ExecutorSteps"
)

swarm_ppl_steps: list[ExecutorStep] = [
    _build_step(idx, ckpt, bundle)
    for idx, ckpt in sorted(_FINISHED.items())
    for bundle in _BUNDLES
]


if __name__ == "__main__":
    executor_main(
        steps=swarm_ppl_steps,
        description=(
            f"David's PPL bundles ({', '.join(b.key for b in _BUNDLES)}) on "
            f"{len(_FINISHED)} finished swarm candidates (D512). v4-8 preemptible, us-central2-b."
        ),
        max_concurrent=120,
    )
