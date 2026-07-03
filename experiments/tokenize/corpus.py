# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Raw text corpus for training our own tokenizers (issue #6796 Track C).

``train_tokenizers.py`` needs raw, untokenized text to learn merges from — unlike the rest of
``experiments/datasets/``, which builds already-tokenized ``TokenizedCache`` handles via
:func:`marin.experiment.data.tokenized`. This module follows the same lazy-artifact convention
(:func:`marin.experiment.data.raw_download`, an :class:`~marin.execution.lazy.ArtifactStep`,
build-opt-in under ``--run``) but produces a plain :class:`~marin.execution.artifact.Artifact`:
sharded ``<domain>.jsonl.gz`` text files plus a ``manifest.json`` of what was actually written.

Domains mirror ``fertility_report.EVAL_DOMAINS`` (English web / code / math — the mix the
trained tokenizers are judged on) but pull from a different split or a skipped prefix so the
~1.5 GB training sample never contains the ~4 MB held-out fertility-eval sample.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass

import click
import datasets
from marin.execution.artifact import Artifact
from marin.execution.lazy import ArtifactStep, lower, run
from marin.experiment.data import raw_download
from marin.utils import fsspec_mkdirs
from rigging.filesystem import atomic_rename, open_url

logger = logging.getLogger(__name__)

# (hf_dataset_id, config, split, text_field).
DomainSpec = tuple[str, str | None, str, str]

# fertility_report.EVAL_DOMAINS reads english_web from the "validation" split of the same
# dataset (zero overlap by construction) and math from the same "train" split we use here, so
# math skips a leading prefix of the stream (_EVAL_OVERLAP_SKIP_BYTES) before collecting,
# comfortably past the ~4 MB/domain the fertility harness reads. ``codeparrot/github-code-clean``
# (fertility_report's code domain) ships as a legacy HF "dataset script," which the installed
# `datasets` version refuses to run (`RuntimeError: Dataset scripts are no longer supported`);
# fertility_report tolerates this per-domain and just skips it. codeparrot-clean-valid is a
# maintained, script-free Python-code parquet dataset, so it is a different source entirely
# (no eval-overlap skip needed).
TRAIN_DOMAINS: dict[str, DomainSpec] = {
    "english_web": ("DKYoon/SlimPajama-6B", None, "train", "text"),
    "code": ("codeparrot/codeparrot-clean-valid", None, "train", "content"),
    "math": ("HuggingFaceTB/finemath", "finemath-3plus", "train", "text"),
}

# Deployment-mix weights: English-dominant general web (the bake-off's SlimPajama-6B training
# set) with a code and math slice so the trained tokenizers learn merges for the full mix they
# are scored on, not just prose.
DOMAIN_WEIGHTS: dict[str, float] = {"english_web": 0.70, "code": 0.20, "math": 0.10}

TOTAL_BYTES = 1_500_000_000  # ~1.5 GB raw text, split across TRAIN_DOMAINS by DOMAIN_WEIGHTS

_EVAL_OVERLAP_SKIP_BYTES = 20_000_000


@dataclass
class CorpusBuildConfig:
    output_path: str = ""
    total_bytes: int = TOTAL_BYTES
    domain_weights: dict[str, float] | None = None


def _stream_domain_shard(spec: DomainSpec, *, max_bytes: int, skip_bytes: int, output_file: str) -> dict:
    """Stream ``spec`` past ``skip_bytes``, then write up to ``max_bytes`` as one jsonl.gz shard."""
    hf_id, config, split, field = spec
    dataset = datasets.load_dataset(hf_id, config, split=split, streaming=True)

    written_bytes = 0
    skipped_bytes = 0
    docs = 0
    with atomic_rename(output_file) as temp_path, open_url(temp_path, "wt", encoding="utf-8", compression="gzip") as out:
        for row in dataset:
            text = row.get(field) or ""
            if not text:
                continue
            encoded = text.encode("utf-8")
            if skipped_bytes < skip_bytes:
                skipped_bytes += len(encoded)
                continue
            out.write(json.dumps({"text": text}, ensure_ascii=False))
            out.write("\n")
            written_bytes += len(encoded)
            docs += 1
            if written_bytes >= max_bytes:
                break

    logger.info("domain %s: wrote %.1f MB / %d docs to %s", hf_id, written_bytes / 1e6, docs, output_file)
    return {"source": hf_id, "bytes": written_bytes, "docs": docs}


def build_tokenizer_training_corpus(cfg: CorpusBuildConfig) -> dict:
    """Stream ``cfg.total_bytes`` across :data:`TRAIN_DOMAINS` into ``cfg.output_path``.

    Writes one ``<domain>.jsonl.gz`` shard per domain plus a ``manifest.json`` recording the
    bytes/docs actually written (streaming HF sources drift over time, so budgets are nominal,
    not exact).
    """
    weights = cfg.domain_weights or DOMAIN_WEIGHTS
    fsspec_mkdirs(cfg.output_path, exist_ok=True)

    manifest: dict[str, dict] = {}
    for domain, spec in TRAIN_DOMAINS.items():
        budget = int(cfg.total_bytes * weights[domain])
        skip = _EVAL_OVERLAP_SKIP_BYTES if domain == "math" else 0
        output_file = f"{cfg.output_path}/{domain}.jsonl.gz"
        logger.info("domain %s: budget=%.1f MB skip=%.1f MB", domain, budget / 1e6, skip / 1e6)
        manifest[domain] = _stream_domain_shard(spec, max_bytes=budget, skip_bytes=skip, output_file=output_file)

    with atomic_rename(f"{cfg.output_path}/manifest.json") as temp_path, open_url(temp_path, "w") as f:
        json.dump({"domain_weights": weights, "total_bytes": cfg.total_bytes, "domains": manifest}, f, indent=2)
    return manifest


def tokenizer_training_corpus_raw() -> ArtifactStep[Artifact]:
    """The raw (untokenized) text corpus :mod:`train_tokenizers` learns merges from."""
    return raw_download(
        "raw/tokenizer_bakeoff_training_corpus",
        fn=build_tokenizer_training_corpus,
        build_config=lambda ctx: CorpusBuildConfig(output_path=ctx.output_path),
        version="2026.07.03",
    )


@click.command()
@click.option("--run", "build", is_flag=True, help="Build the corpus (default: only print the plan).")
def main(build: bool) -> None:
    handle = tokenizer_training_corpus_raw()
    if not build:
        click.echo(lower(handle))
        return
    run(handle)


if __name__ == "__main__":
    main()
