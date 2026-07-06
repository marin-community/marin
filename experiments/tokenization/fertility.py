# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Measure per-arm, per-domain fertility (tokens/byte) — the "evaluate tokenizer fertility" stage.

Fertility is an intrinsic property of a tokenizer + corpus (independent of model scale), and it
is what turns into the deployment serving cost via
:mod:`experiments.tokenization.cost_model`. Two measurement corpora are supported:

* **Held-out HF eval domains** (:data:`EVAL_DOMAINS`) — a cheap pre-filter streamed from the Hub,
  used to rank arms before any GPU time. See :func:`stream_eval_domain_samples`.
* **The soak training corpus** — a bounded per-domain sample of the same corpus the soak arms
  train on (:mod:`experiments.tokenization.corpus`), so fertility is measured on exactly the
  mixture the run is scored against. See :func:`corpus_domain_samples`.

Either way, raw per-domain token/byte counts (not ratios) are written in the shape
:mod:`experiments.tokenization.analysis` reads via ``--fertility``, so a different domain
weighting or serving-cost model can be replayed offline without re-tokenizing.
:func:`fertility_over_corpus_step` is the stage as an :class:`ArtifactStep`.
"""

from __future__ import annotations

import json
import logging
import os

import click
import datasets
from levanter.tokenizers import MarinTokenizer, load_tokenizer
from marin.execution.artifact import Artifact
from marin.execution.lazy import ArtifactStep, apply
from marin.execution.remote import remote
from pydantic import BaseModel, Field
from rigging.filesystem import open_url

from experiments.tokenization.arms import ALL_ARMS, arm_by_name
from experiments.tokenization.cost_model import DEFAULT_SERVING, arm_cost, fertility_of

logger = logging.getLogger(__name__)


class DomainCount(BaseModel):
    """Raw token/byte counts (and their ratio) for one tokenizer over one domain's sample."""

    tokens: int
    bytes: int
    fertility: float  # tokens per byte


class ArmFertility(BaseModel):
    """One arm's per-domain fertility signature — raw counts, so it re-scores under any cost model."""

    name: str
    ref: str
    axis: str
    vocab_size: int
    fertility_overall: float
    total_tokens: int
    total_bytes: int
    by_domain: dict[str, DomainCount]


class FertilityReport(Artifact):
    """The typed output of the fertility stage: per-arm, per-domain tokens/byte over a fixed corpus."""

    domains: list[str] = Field(default_factory=list)
    arms: list[ArmFertility] = Field(default_factory=list)


# The soak training corpus domains (see experiments.tokenization.corpus.TRAIN_DOMAINS).
CORPUS_DOMAINS: tuple[str, ...] = ("english_web", "code", "multilingual", "math")

# Per-domain fertility sample size for the soak-corpus mode: enough for a stable tokens/byte,
# small enough to read quickly without pulling a full (up to ~2 GB) domain shard into memory.
CORPUS_SAMPLE_BYTES = 20_000_000

# (dataset, config, split, text_field). Held-out HF eval domains for the pre-filter mode: kept
# small and diverse so the fertility profile spans the regimes where tokenizers differ most
# (prose, code, numerics, non-Latin script).
DomainSpec = tuple[str, str | None, str, str]
EVAL_DOMAINS: dict[str, DomainSpec] = {
    "english_web": ("DKYoon/SlimPajama-6B", None, "validation", "text"),
    "code": ("codeparrot/github-code-clean", "all-all", "train", "code"),
    "math": ("HuggingFaceTB/finemath", "finemath-3plus", "train", "text"),
    "multilingual_zh": ("wikimedia/wikipedia", "20231101.zh", "train", "text"),
}


# --- sample sources ---------------------------------------------------------------


def _stream_text(spec: DomainSpec, max_bytes: int) -> list[str]:
    """Stream up to ``max_bytes`` of UTF-8 text from a HF dataset as a list of documents."""
    name, config, split, field = spec
    ds = datasets.load_dataset(name, config, split=split, streaming=True)
    chunks: list[str] = []
    total = 0
    for row in ds:
        text = row.get(field) or ""
        if not text:
            continue
        chunks.append(text)
        total += len(text.encode("utf-8"))
        if total >= max_bytes:
            break
    return chunks


def stream_eval_domain_samples(max_bytes: int) -> dict[str, list[str]]:
    """Fetch each held-out :data:`EVAL_DOMAINS` sample, skipping (with a warning) any that fail."""
    samples: dict[str, list[str]] = {}
    for domain, spec in EVAL_DOMAINS.items():
        try:
            texts = _stream_text(spec, max_bytes)
        except Exception as e:  # a flaky/gated source shouldn't sink the whole report
            logger.warning("skipping domain %s (%s): %s", domain, spec[0], str(e)[:160])
            continue
        if texts:
            samples[domain] = texts
            logger.info("domain %s: %.2f MB", domain, sum(len(t.encode("utf-8")) for t in texts) / 1e6)
    if not samples:
        raise RuntimeError("no eval domains loaded; cannot produce a fertility report")
    return samples


def read_domain_sample(corpus_dir: str, domain: str, max_bytes: int) -> list[str]:
    """Read documents from the front of ``domain``'s corpus shard up to ``max_bytes`` of text.

    Stops as soon as the running total reaches the budget, so a small per-domain sample never
    requires reading a full (up to ~2 GB) ``<domain>.jsonl.gz`` shard into memory.
    """
    path = f"{corpus_dir}/{domain}.jsonl.gz"
    texts: list[str] = []
    total_bytes = 0
    with open_url(path, "rt", encoding="utf-8", compression="gzip") as f:
        for line in f:
            text = json.loads(line).get("text")
            if not text:
                continue
            texts.append(text)
            total_bytes += len(text.encode("utf-8"))
            if total_bytes >= max_bytes:
                break
    return texts


def corpus_domain_samples(
    corpus_dir: str,
    domains: tuple[str, ...] = CORPUS_DOMAINS,
    max_bytes_per_domain: int = CORPUS_SAMPLE_BYTES,
) -> dict[str, list[str]]:
    """A bounded, fixed per-domain sample of the soak corpus to score every tokenizer against."""
    samples = {domain: read_domain_sample(corpus_dir, domain, max_bytes_per_domain) for domain in domains}
    for domain, texts in samples.items():
        sample_bytes = sum(len(t.encode("utf-8")) for t in texts)
        logger.info("domain %s: %d docs, %.1f MB sample", domain, len(texts), sample_bytes / 1e6)
    return samples


# --- measurement ------------------------------------------------------------------


def arm_fertility_by_domain(tokenizer: MarinTokenizer, domain_samples: dict[str, list[str]]) -> dict[str, DomainCount]:
    """Raw token/byte counts for one tokenizer over each domain's sample.

    Encodes with ``add_special_tokens=False``. Loads through levanter (``Tokenizer.from_file``),
    the exact path marin's tokenize pipeline uses — this matters for SuperBPE, where
    ``AutoTokenizer.from_pretrained`` would honor the repo's GPT2Tokenizer class and overwrite the
    superword pretokenizer, silently measuring a worse-than-baseline fertility.
    """

    def encode(text: str) -> list[int]:
        return tokenizer.encode(text, add_special_tokens=False)

    out: dict[str, DomainCount] = {}
    for domain, texts in domain_samples.items():
        m = fertility_of(encode, texts)
        out[domain] = DomainCount(tokens=m.total_tokens, bytes=m.total_bytes, fertility=m.fertility)
    return out


def measure_arms(arm_names: tuple[str, ...], domain_samples: dict[str, list[str]]) -> list[ArmFertility]:
    """Per-arm fertility, one :class:`ArmFertility` per arm."""
    rows = []
    for name in arm_names:
        arm = arm_by_name(name)
        tokenizer = load_tokenizer(arm.ref)
        real_vocab = len(tokenizer.get_vocab())
        if real_vocab != arm.vocab_size:
            logger.warning("%s: registered vocab %d != loaded %d; using loaded", arm.name, arm.vocab_size, real_vocab)
        by_domain = arm_fertility_by_domain(tokenizer, domain_samples)
        total_tokens = sum(d.tokens for d in by_domain.values())
        total_bytes = sum(d.bytes for d in by_domain.values())
        rows.append(
            ArmFertility(
                name=arm.name,
                ref=arm.ref,
                axis=str(arm.axis),
                vocab_size=real_vocab,
                fertility_overall=total_tokens / total_bytes,
                total_tokens=total_tokens,
                total_bytes=total_bytes,
                by_domain=by_domain,
            )
        )
        logger.info("measured %s: vocab=%d, ref=%s", arm.name, real_vocab, arm.ref)
    return rows


# --- step ---------------------------------------------------------------------------


def build_fertility_report(
    corpus_dir: str,
    arm_names: list[str],
    max_bytes_per_domain: int = CORPUS_SAMPLE_BYTES,
) -> FertilityReport:
    """Measure ``arm_names`` fertility over the soak corpus at ``corpus_dir`` as a :class:`FertilityReport`."""
    domain_samples = corpus_domain_samples(corpus_dir, max_bytes_per_domain=max_bytes_per_domain)
    return FertilityReport(domains=list(CORPUS_DOMAINS), arms=measure_arms(tuple(arm_names), domain_samples))


def fertility_over_corpus_step(
    arm_names: tuple[str, ...],
    corpus: ArtifactStep[Artifact],
    *,
    max_bytes_per_domain: int = CORPUS_SAMPLE_BYTES,
    resources=None,
    version: str = "dev",
) -> ArtifactStep[FertilityReport]:
    """The "evaluate tokenizer fertility" stage: ``arm_names`` measured over the soak ``corpus``.

    Deps on the corpus artifact; each arm's tokenizer resolves through its ``trained/<name>`` (or
    off-the-shelf) ref, so the trained arms must already be pushed (see
    :func:`experiments.tokenization.train_tokenizer.trained_tokenizer`). Returns a typed
    :class:`FertilityReport` for :func:`experiments.tokenization.analysis.bakeoff_report_step`.
    """
    fn = build_fertility_report
    return apply(
        "tokenizer-soak/fertility",
        remote(fn, resources=resources) if resources is not None else fn,
        version=version,
        artifact_type=FertilityReport,
        corpus_dir=corpus,
        arm_names=list(arm_names),
        max_bytes_per_domain=max_bytes_per_domain,
    )


def _print_cost_table(rows: list[ArmFertility], domains: list[str]) -> None:
    """Price rows at the default serving model and print a fertility + serving-cost table."""
    costs = {r.name: arm_cost(r.name, r.vocab_size, r.fertility_overall, DEFAULT_SERVING) for r in rows}
    ref = costs.get("marin-128k") or next(iter(costs.values()))
    attn_share = DEFAULT_SERVING.attention_flop_fraction(ref.vocab_size) * 100
    print(f"\n=== fertility + serving cost @ {DEFAULT_SERVING.context_len} ctx ===")
    print(f"(attention = {attn_share:.1f}% of forward FLOPs at this context; 5:1 local:global)")
    header = f"{'arm':16s} {'vocab':>7s} {'B/tok':>6s} " + " ".join(f"{d[:8]:>8s}" for d in domains)
    header += f" {'infFLOP/B':>10s} {'rel_serve':>9s} {'head%':>5s}"
    print(header)
    for r in sorted(rows, key=lambda x: costs[x.name].infer_flops_per_byte):
        c = costs[r.name]
        per = " ".join(f"{r.by_domain[d].bytes / r.by_domain[d].tokens:8.2f}" for d in domains)
        print(
            f"{r.name:16s} {r.vocab_size:7d} {1.0 / r.fertility_overall:6.2f} {per} "
            f"{c.infer_flops_per_byte:10.3e} {c.infer_flops_per_byte / ref.infer_flops_per_byte:9.3f} "
            f"{c.lm_head_flop_fraction * 100:4.1f}%"
        )


@click.command()
@click.option("--corpus-dir", default=None, help="soak corpus dir; if unset, streams the HF EVAL_DOMAINS")
@click.option("--arms", default=None, help="comma-separated arm names (default: all registered)")
@click.option("--max-mb", type=float, default=CORPUS_SAMPLE_BYTES / 1e6, help="MB of text per domain")
@click.option("--out", default="fertility.json", help="write the FertilityReport JSON here")
def main(corpus_dir: str | None, arms: str | None, max_mb: float, out: str) -> None:
    """Measure per-arm fertility over the soak corpus (or streamed HF eval domains) and write it."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    arm_names = tuple(arms.split(",")) if arms else tuple(a.name for a in ALL_ARMS)
    max_bytes = int(max_mb * 1e6)
    if corpus_dir:
        domain_samples = corpus_domain_samples(corpus_dir, max_bytes_per_domain=max_bytes)
        domains = list(CORPUS_DOMAINS)
    else:
        domain_samples = stream_eval_domain_samples(max_bytes)
        domains = list(domain_samples.keys())

    rows = measure_arms(arm_names, domain_samples)
    out_dir = os.path.dirname(out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(out, "w") as f:
        json.dump(FertilityReport(domains=domains, arms=rows).model_dump(exclude={"path"}), f, indent=2)
    _print_cost_table(rows, domains)
    print(f"\nRaw data: {out}")


if __name__ == "__main__":
    main()
