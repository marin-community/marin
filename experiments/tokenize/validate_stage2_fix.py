# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Quantify the stage-2 sampling bug's effect on per-domain tokenizer compression.

Commit 11bd2f4e9c fixed a bug in ``train_tokenizers.py``: SuperBPE stage 2 learns superword
merges from a ``STAGE2_SAMPLE_BYTES`` (300 MB) sample of the training corpus, but
``read_corpus`` concatenates domains in a fixed order (``english_web`` first, ~half of the
~4 GB corpus), so the un-shuffled leading-byte sample was 100% English web text — stage 2
never saw code, multilingual, or math. ``_sample_stage2_corpus`` now shuffles the corpus
before sampling, so the currently-deployed ``soak-superbpe-64k``/``soak-superbpe-128k`` arms
(trained/pushed before the fix) are "buggy", while calling ``train_one`` from this working
tree produces the "fixed" tokenizers.

This script measures the fix's effect without a GPU re-run: it re-trains the two ``soak-*``
SuperBPE arms on the existing soak corpus (CPU-only, from the already-fixed working tree),
loads the buggy tokenizers already deployed under ``trained/soak-superbpe-*`` plus the
Llama-3 baseline (``marin-128k``), and compares fertility (tokens/byte, lower = better
compression) per domain across all five.

Run on a CW CPU box (SuperBPE stage 2 is single-threaded numpy, so 1-2 CPUs is enough; the
corpus is ~4 GB in memory during retraining, so give it generous memory headroom):

    uv run iris --cluster=cw-rno2a job run --cpu 4 --memory 64GB --extra cpu \\
      --enable-extra-resources --job-name validate-stage2-fix \\
      -- python -m experiments.tokenize.validate_stage2_fix
"""

from __future__ import annotations

import argparse
import json
import logging
import os

from levanter.tokenizers import MarinTokenizer, load_tokenizer
from rigging.filesystem import open_url

from experiments.tokenize.bakeoff_tokenizers import arm_by_name
from experiments.tokenize.flop_equivalent import FertilityMeasurement, fertility_of
from experiments.tokenize.train_tokenizers import CORPUS_DOMAINS, TRAIN_SPECS, read_corpus, train_one

logger = logging.getLogger(__name__)

# Must match build_soak_tokenizers.py's `_CORPUS_VERSION` — the corpus the currently-deployed
# (buggy) soak tokenizers were actually trained from.
CORPUS_VERSION = "2026.07.04"

# Per-domain fertility test sample size: enough for a stable tokens/byte, small enough to read
# quickly without pulling a full (up to ~2 GB) domain shard into memory.
DOMAIN_SAMPLE_BYTES = 20_000_000

# The two soak SuperBPE vocab sizes under test.
VOCAB_LABELS: tuple[str, ...] = ("64k", "128k")

BASELINE_ARM_NAME = "marin-128k"
OVERALL_DOMAIN = "overall"

DEFAULT_OUT_PATH = "experiments/tokenize/results/stage2_fix_validation.json"
DEFAULT_TRAIN_OUT_DIR = "/tmp/validate_stage2_fix/trained"


def _soak_arm_name(vocab_label: str) -> str:
    return f"soak-superbpe-{vocab_label}"


def _fixed_tokenizer_dir(train_out_dir: str, vocab_label: str) -> str:
    return f"{train_out_dir}/{_soak_arm_name(vocab_label)}"


def read_domain_sample(corpus_dir: str, domain: str, max_bytes: int) -> list[str]:
    """Read documents from the front of ``domain``'s shard up to ``max_bytes`` of UTF-8 text.

    Stops as soon as the running total reaches the budget, so a small per-domain fertility
    sample never requires reading a full (up to ~2 GB) domain shard into memory.
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


def domain_fertility_samples(
    corpus_dir: str,
    domains: tuple[str, ...] = CORPUS_DOMAINS,
    max_bytes_per_domain: int = DOMAIN_SAMPLE_BYTES,
) -> dict[str, list[str]]:
    """A bounded, fixed per-domain text sample to score every tokenizer against."""
    samples = {domain: read_domain_sample(corpus_dir, domain, max_bytes_per_domain) for domain in domains}
    for domain, texts in samples.items():
        sample_bytes = sum(len(t.encode("utf-8")) for t in texts)
        logger.info("domain %s: %d docs, %.1f MB test sample", domain, len(texts), sample_bytes / 1e6)
    return samples


def tokenizer_fertility_by_domain(
    tokenizer: MarinTokenizer, domain_samples: dict[str, list[str]]
) -> dict[str, FertilityMeasurement]:
    """Tokens/byte for ``tokenizer`` over each domain's test sample.

    Encodes with ``add_special_tokens=False``, the same call
    ``fertility_report.measure_arm`` uses, so numbers are directly comparable.
    """

    def encode(text: str) -> list[int]:
        return tokenizer.encode(text, add_special_tokens=False)

    return {domain: fertility_of(encode, texts) for domain, texts in domain_samples.items()}


def retrain_fixed_tokenizer(vocab_label: str, corpus_texts: list[str], train_out_dir: str) -> MarinTokenizer:
    """Retrain ``soak-superbpe-<vocab_label>`` on the working tree's fixed stage-2 sampling.

    The working tree already has the ``_sample_stage2_corpus`` shuffle fix, so calling
    ``train_one`` now — unlike the currently-deployed arm of the same name — produces the
    FIXED tokenizer.
    """
    name = _soak_arm_name(vocab_label)
    spec = next(s for s in TRAIN_SPECS if s.name == name)
    out_dir = _fixed_tokenizer_dir(train_out_dir, vocab_label)
    row = train_one(spec, corpus_texts, out_dir)
    logger.info("retrained FIXED %s: vocab=%d in %.1fs -> %s", name, row["vocab_size"], row["train_seconds"], out_dir)
    return load_tokenizer(out_dir)


def load_buggy_tokenizer(vocab_label: str) -> MarinTokenizer:
    """The currently-deployed (pre-fix) ``soak-superbpe-<vocab_label>`` arm."""
    return load_tokenizer(arm_by_name(_soak_arm_name(vocab_label)).ref)


def _pct_change(before: float, after: float) -> float:
    return 100.0 * (after - before) / before


def format_fertility_table(fertility: dict[str, dict[str, FertilityMeasurement]], tokenizer_labels: list[str]) -> str:
    """A domain x tokenizer table of tokens/byte (lower = better compression)."""
    lines = [f"{'domain':16s}" + "".join(f"{label:>14s}" for label in tokenizer_labels)]
    for domain, by_tokenizer in fertility.items():
        row = f"{domain:16s}" + "".join(f"{by_tokenizer[label].fertility:14.4f}" for label in tokenizer_labels)
        lines.append(row)
    return "\n".join(lines)


def format_deltas(fertility: dict[str, dict[str, FertilityMeasurement]]) -> str:
    """Buggy->fixed and fixed-vs-``marin-128k`` %% deltas, negative = fixed compresses better."""
    lines = ["\nbuggy -> fixed (%, negative = fixed compresses better)"]
    lines.append(f"{'domain':16s}" + "".join(f"{v:>14s}" for v in VOCAB_LABELS))
    for domain, by_tokenizer in fertility.items():
        row = f"{domain:16s}"
        for v in VOCAB_LABELS:
            before = by_tokenizer[f"buggy-{v}"].fertility
            after = by_tokenizer[f"fixed-{v}"].fertility
            row += f"{_pct_change(before, after):13.2f}%"
        lines.append(row)

    lines.append(f"\nfixed vs {BASELINE_ARM_NAME} (%, negative = fixed compresses better)")
    lines.append(f"{'domain':16s}" + "".join(f"{v:>14s}" for v in VOCAB_LABELS))
    for domain, by_tokenizer in fertility.items():
        row = f"{domain:16s}"
        for v in VOCAB_LABELS:
            fixed = by_tokenizer[f"fixed-{v}"].fertility
            baseline = by_tokenizer[BASELINE_ARM_NAME].fertility
            row += f"{_pct_change(baseline, fixed):13.2f}%"
        lines.append(row)
    return "\n".join(lines)


def _measurement_to_json(m: FertilityMeasurement) -> dict:
    return {"tokens": m.total_tokens, "bytes": m.total_bytes, "fertility": m.fertility}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--corpus-dir",
        default=None,
        help="override the corpus dir (default: $MARIN_PREFIX/raw/soak_tokenizer_corpus/<version>)",
    )
    ap.add_argument(
        "--domain-sample-mb", type=float, default=DOMAIN_SAMPLE_BYTES / 1e6, help="MB of per-domain fertility test text"
    )
    ap.add_argument(
        "--train-out-dir", default=DEFAULT_TRAIN_OUT_DIR, help="where to save the retrained FIXED tokenizers"
    )
    ap.add_argument(
        "--out", default=DEFAULT_OUT_PATH, help="write the raw per-domain/tokenizer measurements as JSON here"
    )
    args = ap.parse_args()

    corpus_dir = (
        args.corpus_dir or f"{os.environ['MARIN_PREFIX'].rstrip('/')}/raw/soak_tokenizer_corpus/{CORPUS_VERSION}"
    )
    logger.info("corpus: %s", corpus_dir)

    domain_samples = domain_fertility_samples(corpus_dir, max_bytes_per_domain=int(args.domain_sample_mb * 1e6))
    domain_samples[OVERALL_DOMAIN] = [text for texts in domain_samples.values() for text in texts]

    corpus_texts = read_corpus(corpus_dir)
    total_mb = sum(len(t.encode("utf-8")) for t in corpus_texts) / 1e6
    logger.info("loaded full corpus for retraining: %d docs, %.1f MB", len(corpus_texts), total_mb)

    tokenizer_labels: list[str] = []
    tokenizers: dict[str, MarinTokenizer] = {}
    refs: dict[str, str] = {}

    for v in VOCAB_LABELS:
        buggy_label = f"buggy-{v}"
        tokenizer_labels.append(buggy_label)
        refs[buggy_label] = arm_by_name(_soak_arm_name(v)).ref
        tokenizers[buggy_label] = load_buggy_tokenizer(v)

        fixed_label = f"fixed-{v}"
        tokenizer_labels.append(fixed_label)
        refs[fixed_label] = _fixed_tokenizer_dir(args.train_out_dir, v)
        tokenizers[fixed_label] = retrain_fixed_tokenizer(v, corpus_texts, args.train_out_dir)

    tokenizer_labels.append(BASELINE_ARM_NAME)
    refs[BASELINE_ARM_NAME] = arm_by_name(BASELINE_ARM_NAME).ref
    tokenizers[BASELINE_ARM_NAME] = load_tokenizer(refs[BASELINE_ARM_NAME])

    fertility: dict[str, dict[str, FertilityMeasurement]] = {domain: {} for domain in domain_samples}
    for label in tokenizer_labels:
        for domain, measurement in tokenizer_fertility_by_domain(tokenizers[label], domain_samples).items():
            fertility[domain][label] = measurement

    print("\nFertility (tokens/byte, lower = better compression)")
    print(format_fertility_table(fertility, tokenizer_labels))
    print(format_deltas(fertility))

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    report = {
        "corpus_dir": corpus_dir,
        "domain_sample_bytes": {d: sum(len(t.encode("utf-8")) for t in texts) for d, texts in domain_samples.items()},
        "tokenizer_refs": refs,
        "fertility": {
            domain: {label: _measurement_to_json(m) for label, m in by_tokenizer.items()}
            for domain, by_tokenizer in fertility.items()
        },
    }
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    logger.info("wrote %s", args.out)


if __name__ == "__main__":
    main()
