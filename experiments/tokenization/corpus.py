# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Raw text corpus for training our own tokenizers (Track C).

``train_tokenizer.py`` needs raw, untokenized text to learn merges from — unlike the rest of
``experiments/datasets/``, which builds already-tokenized ``TokenizedCache`` handles via
:func:`marin.experiment.data.tokenized`. This module follows the same lazy-artifact convention
(:func:`marin.experiment.data.raw_download`, an :class:`~marin.execution.lazy.ArtifactStep`,
build-opt-in under ``--run``) but produces a plain :class:`~marin.execution.artifact.Artifact`:
sharded ``<domain>.jsonl.gz`` text files plus a ``manifest.json`` of what was actually written.

Domains span English web, code, several languages, and math — a representative pretraining mix
— and pull from a different split or a skipped prefix than ``fertility.EVAL_DOMAINS`` so
the ~4 GB training sample never contains the ~4 MB held-out fertility-eval sample.
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

# Each domain draws from one or more script-free, ungated HF parquet sources; a domain's byte
# budget is split evenly across its sub-sources. This mirrors the representative composition of a
# general 10T pretraining mixture (web-heavy, plus code, multiple languages, and math) so the
# trained tokenizers learn merges for the full range of content they are deployed on — not just
# English prose. fertility.EVAL_DOMAINS reads english_web from the "validation" split of
# the same dataset (zero overlap by construction) and math from the same "train" split used here,
# so math skips a leading prefix of the stream (_EVAL_OVERLAP_SKIP_BYTES) before collecting,
# comfortably past the ~4 MB/domain the fertility harness reads. Multilingual spans three scripts
# (Latin/Cyrillic/CJK + Arabic + Devanagari) via Wikipedia language editions; code is Python
# (codeparrot-clean-valid, the maintained script-free parquet — github-code-clean and the-stack
# are a legacy dataset-script and gated respectively, so neither loads here).
TRAIN_DOMAINS: dict[str, tuple[DomainSpec, ...]] = {
    "english_web": (("DKYoon/SlimPajama-6B", None, "train", "text"),),
    "code": (("codeparrot/codeparrot-clean-valid", None, "train", "content"),),
    "multilingual": (
        ("wikimedia/wikipedia", "20231101.de", "train", "text"),
        ("wikimedia/wikipedia", "20231101.fr", "train", "text"),
        ("wikimedia/wikipedia", "20231101.es", "train", "text"),
        ("wikimedia/wikipedia", "20231101.ru", "train", "text"),
        ("wikimedia/wikipedia", "20231101.zh", "train", "text"),
        ("wikimedia/wikipedia", "20231101.ja", "train", "text"),
        ("wikimedia/wikipedia", "20231101.ar", "train", "text"),
        ("wikimedia/wikipedia", "20231101.hi", "train", "text"),
    ),
    "math": (("HuggingFaceTB/finemath", "finemath-3plus", "train", "text"),),
}

# Representative deployment-mix weights: English-dominant general web, with code, multiple
# languages, and math slices. Mirrors the domain composition of the grug-moe datakit pretrain
# mixture (which is itself only available pre-tokenized under one tokenizer, so cannot be used
# directly to train alternative tokenizers — this raw sample stands in for it).
DOMAIN_WEIGHTS: dict[str, float] = {"english_web": 0.50, "code": 0.20, "multilingual": 0.20, "math": 0.10}

TOTAL_BYTES = 4_000_000_000  # ~4 GB raw text, split across TRAIN_DOMAINS by DOMAIN_WEIGHTS

_EVAL_OVERLAP_SKIP_BYTES = 20_000_000


@dataclass
class CorpusBuildConfig:
    output_path: str = ""
    total_bytes: int = TOTAL_BYTES
    domain_weights: dict[str, float] | None = None


def _stream_one_source(spec: DomainSpec, *, max_bytes: int, skip_bytes: int, out) -> dict:
    """Stream ``spec`` past ``skip_bytes`` into ``out``, up to ``max_bytes``; return its stats."""
    hf_id, config, split, field = spec
    dataset = datasets.load_dataset(hf_id, config, split=split, streaming=True)
    written_bytes = 0
    skipped_bytes = 0
    docs = 0
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
    return {"source": hf_id, "config": config, "bytes": written_bytes, "docs": docs}


def _stream_domain_shard(specs: tuple[DomainSpec, ...], *, max_bytes: int, skip_bytes: int, output_file: str) -> dict:
    """Stream ``specs`` (a domain's sub-sources) into one jsonl.gz shard, splitting the budget.

    The domain's ``max_bytes`` is divided evenly across its sub-sources so a multi-language or
    multi-source domain is balanced across them. A sub-source that fails to stream (gated, moved,
    or a transient Hub error) is logged and skipped rather than aborting the whole corpus build —
    the same tolerance ``fertility`` applies per-domain.
    """
    per_source_budget = max(1, max_bytes // len(specs))
    sources: list[dict] = []
    total_bytes = 0
    total_docs = 0
    with atomic_rename(output_file) as temp_path, open_url(temp_path, "wt", encoding="utf-8", compression="gzip") as out:
        for spec in specs:
            try:
                stats = _stream_one_source(spec, max_bytes=per_source_budget, skip_bytes=skip_bytes, out=out)
            except Exception as e:
                logger.warning("sub-source %s (config=%s) failed to stream: %s", spec[0], spec[1], e)
                sources.append({"source": spec[0], "config": spec[1], "bytes": 0, "docs": 0, "error": str(e)})
                continue
            sources.append(stats)
            total_bytes += stats["bytes"]
            total_docs += stats["docs"]

    logger.info(
        "domain shard %s: wrote %.1f MB / %d docs from %d sources",
        output_file,
        total_bytes / 1e6,
        total_docs,
        len(specs),
    )
    return {"bytes": total_bytes, "docs": total_docs, "sources": sources}


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
    """The raw (untokenized) text corpus :mod:`train_tokenizer` learns merges from."""
    return raw_download(
        "raw/tokenizer_bakeoff_training_corpus",
        fn=build_tokenizer_training_corpus,
        build_config=lambda ctx: CorpusBuildConfig(output_path=ctx.output_path),
        version="2026.07.04",
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
