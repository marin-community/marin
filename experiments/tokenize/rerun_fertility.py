# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Measure per-arm per-domain fertility for the corrected soak tokenizer re-run.

Commit 11bd2f4e9c fixed a stage-2 sampling bug in ``train_tokenizers.py`` (see
``validate_stage2_fix.py`` for the full story); the corrected soak re-run trains six arms —
the four stage-2-fixed SuperBPE soak variants plus the off-the-shelf ``superbpe-128k`` and the
incumbent ``marin-128k`` baseline — and this script measures each arm's fertility (tokens/byte)
per domain over the same soak corpus, so the training run can be scored for feBPB once it
finishes.

Reuses ``validate_stage2_fix``'s corpus loading: reads a bounded ~20 MB/domain sample from the
soak corpus (``$MARIN_PREFIX/raw/soak_tokenizer_corpus/<version>``) and tokenizes it with each
arm's tokenizer via ``levanter.tokenizers.load_tokenizer``. Writes raw per-domain token/byte
counts (not ratios) in the exact shape ``experiments.tokenize.bakeoff_analysis`` reads via
``--fertility``, so a different domain weighting can be replayed later without re-tokenizing.

Run on a CW CPU box (loads ~4GB of corpus samples across 6 tokenizers; CPU-only):

    uv run iris --cluster=cw-rno2a job run --cpu 4 --memory 64GB --extra cpu \\
      --enable-extra-resources --job-name rerun-fertility \\
      -e MARIN_PREFIX s3://marin-us-east-02a/marin \\
      -- python -m experiments.tokenize.rerun_fertility
"""

from __future__ import annotations

import argparse
import json
import logging
import os

from levanter.tokenizers import MarinTokenizer, load_tokenizer

from experiments.tokenize.bakeoff_tokenizers import arm_by_name
from experiments.tokenize.flop_equivalent import fertility_of
from experiments.tokenize.train_tokenizers import CORPUS_DOMAINS
from experiments.tokenize.validate_stage2_fix import DOMAIN_SAMPLE_BYTES, domain_fertility_samples

logger = logging.getLogger(__name__)

# Must match build_soak_tokenizers.py's/build_fixed_soak_tokenizers.py's `_CORPUS_VERSION` --
# the corpus the re-run arms are trained/scored against.
CORPUS_VERSION = "2026.07.04"

# The 6 arms under test in the corrected soak re-run: the 4 stage-2-fixed SuperBPE soak
# variants (base + individual-digit pretok, at 64k/128k vocab) plus the two reference arms
# (off-the-shelf SuperBPE-128k and the incumbent marin-128k baseline).
RERUN_ARM_NAMES: tuple[str, ...] = (
    "soak-superbpe-64k-fixed",
    "soak-superbpe-128k-fixed",
    "soak-superbpe-64k-digits-fixed",
    "soak-superbpe-128k-digits-fixed",
    "superbpe-128k",
    "marin-128k",
)

DEFAULT_OUT_PATH = "experiments/tokenize/results/rerun_fertility.json"


def arm_fertility_by_domain(
    tokenizer: MarinTokenizer, domain_samples: dict[str, list[str]]
) -> dict[str, dict[str, int]]:
    """Real token/byte counts for one tokenizer over each domain's test sample.

    Encodes with ``add_special_tokens=False``, matching ``fertility_report.measure_arm`` and
    ``validate_stage2_fix.tokenizer_fertility_by_domain``. Returns raw counts (not fertility
    ratios) in the ``{"tokens": int, "bytes": int}`` shape
    ``bakeoff_analysis._weighted_fertility`` reads.
    """

    def encode(text: str) -> list[int]:
        return tokenizer.encode(text, add_special_tokens=False)

    by_domain: dict[str, dict[str, int]] = {}
    for domain, texts in domain_samples.items():
        measurement = fertility_of(encode, texts)
        by_domain[domain] = {"tokens": measurement.total_tokens, "bytes": measurement.total_bytes}
    return by_domain


def measure_rerun_arms(arm_names: tuple[str, ...], domain_samples: dict[str, list[str]]) -> list[dict]:
    """Per-arm fertility rows in the exact shape ``bakeoff_analysis --fertility`` expects."""
    rows = []
    for name in arm_names:
        arm = arm_by_name(name)
        tokenizer = load_tokenizer(arm.ref)
        by_domain = arm_fertility_by_domain(tokenizer, domain_samples)
        rows.append({"name": arm.name, "vocab_size": arm.vocab_size, "by_domain": by_domain})
        logger.info("measured %s: vocab=%d, ref=%s", arm.name, arm.vocab_size, arm.ref)
    return rows


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
    ap.add_argument("--out", default=DEFAULT_OUT_PATH, help="write the fertility JSON (bakeoff_analysis input) here")
    args = ap.parse_args()

    corpus_dir = (
        args.corpus_dir or f"{os.environ['MARIN_PREFIX'].rstrip('/')}/raw/soak_tokenizer_corpus/{CORPUS_VERSION}"
    )
    logger.info("corpus: %s", corpus_dir)

    domain_samples = domain_fertility_samples(corpus_dir, max_bytes_per_domain=int(args.domain_sample_mb * 1e6))
    arms = measure_rerun_arms(RERUN_ARM_NAMES, domain_samples)

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    report = {"domains": list(CORPUS_DOMAINS), "arms": arms}
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    logger.info("wrote %s", args.out)
    # Also emit to stdout between markers so the JSON can be recovered from the job logs
    # (the --out file lives on the ephemeral job filesystem).
    print("RERUN_FERTILITY_JSON_BEGIN")
    print(json.dumps(report))
    print("RERUN_FERTILITY_JSON_END")


if __name__ == "__main__":
    main()
