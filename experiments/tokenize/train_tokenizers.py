# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Train our own tokenizers on the grug-moe data mix (Track C).

Off-the-shelf arms (``bakeoff_tokenizers.BASELINE_ARMS``/``SUPERBPE_ARMS``) sample whatever
vocabulary and segmentation other teams optimized for other data. This harness instead trains
plain BPE and SuperBPE (two-stage superword BPE; see ``superbpe_trainer.py`` for the method and
why it is a reimplementation rather than the paper's own code) directly on
``experiments.tokenize.corpus``'s English/code/math sample, and exports each as an HF
``tokenizer.json`` + ``tokenizer_config.json`` pair ready to register as a
:class:`~experiments.tokenize.bakeoff_tokenizers.TokenizerArm`.

Run:
  uv run python -m experiments.tokenize.corpus --run          # build the training corpus once
  uv run python -m experiments.tokenize.train_tokenizers \\
      --corpus-dir <printed corpus output_path> --out-dir experiments/tokenize/results/trained_tokenizers
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, replace
from enum import StrEnum

from rigging.filesystem import open_url

from experiments.tokenize.superbpe_trainer import Pretok, extend_with_superwords, pretok_config, train_plain_bpe

logger = logging.getLogger(__name__)

EOS_TOKEN = "<|endoftext|>"
CORPUS_DOMAINS: tuple[str, ...] = ("english_web", "code", "multilingual", "math")

# Stage 1 (plain BPE, stock Rust trainer) uses the full corpus — it is cheap regardless of
# corpus size. Stage 2's from-scratch merge learner (superbpe_trainer._learn_merges) costs
# roughly a fixed amount per merge, dominated by one global vectorized pair-recount per batch
# over its flattened segment array (see that module's docstring); recount cost scales with
# array size, i.e. with corpus size. Benchmarked at ~2-5ms/merge on a 60 MB sample (~12-15M
# flattened tokens); this cap keeps the largest sweep config (~96k merges) to well under an
# hour while still giving stage 2 a multiple of the eval fertility sample size to converge on.
STAGE2_SAMPLE_BYTES = 300_000_000


def _bounded_byte_prefix(texts: list[str], max_bytes: int) -> list[str]:
    """The longest prefix of ``texts`` whose cumulative UTF-8 size does not exceed ``max_bytes``."""
    out: list[str] = []
    total = 0
    for text in texts:
        size = len(text.encode("utf-8"))
        if total + size > max_bytes and out:
            break
        out.append(text)
        total += size
    return out


# Fixed seed so the stage-2 sample is reproducible across tokenizer builds.
_STAGE2_SAMPLE_SEED = 0


def _sample_stage2_corpus(texts: list[str], max_bytes: int) -> list[str]:
    """A representative ~``max_bytes`` sample of the corpus for stage-2 superword learning.

    ``read_corpus`` concatenates domains in a fixed order (english_web first, ~half the
    corpus), so a leading byte prefix would draw the sample from that one domain and the
    superword layer would never see code, multilingual, or math. Shuffling first makes the
    sample track the corpus's domain proportions — i.e. the target training mixture.
    """
    shuffled = list(texts)
    random.Random(_STAGE2_SAMPLE_SEED).shuffle(shuffled)
    return _bounded_byte_prefix(shuffled, max_bytes)


class TokenizerKind(StrEnum):
    PLAIN_BPE = "plain_bpe"
    SUPERBPE = "superbpe"


@dataclass(frozen=True)
class TrainSpec:
    """One tokenizer to train. ``transition_vocab_size`` is SuperBPE's stage-1 cutoff ``t``.

    ``pretok`` selects the pretokenizer variant (word-split regex + digit handling); see
    :class:`~experiments.tokenize.superbpe_trainer.Pretok`.
    """

    name: str
    kind: TokenizerKind
    vocab_size: int
    transition_vocab_size: int | None = None
    pretok: Pretok = Pretok.SUPERBPE


# The sweep: a plain-BPE vocab control (does our own data beat the off-the-shelf BPE arms at
# matched vocab?), the main SuperBPE (vocab x transition-point) sweep, and a small-vocab
# SuperBPE pair (gpt-neox's 50k vocab co-led feBPB off the shelf, so a superword tokenizer at
# that scale is worth a direct test). Transition points are chosen at t/T ~= 0.4 (the paper's
# "efficiency-best" ratio, e.g. t=80k at T=200k) and ~= 0.8 (close to its "quality-best" ratio,
# t=180k at T=200k); the small-vocab pair uses a single t/T = 0.5 point.
TRAIN_SPECS: tuple[TrainSpec, ...] = (
    TrainSpec("trained-bpe-64k", TokenizerKind.PLAIN_BPE, 64_000),
    TrainSpec("trained-bpe-96k", TokenizerKind.PLAIN_BPE, 96_000),
    TrainSpec("trained-bpe-128k", TokenizerKind.PLAIN_BPE, 128_000),
    TrainSpec("trained-superbpe-96k-t38k", TokenizerKind.SUPERBPE, 96_000, 38_000),
    TrainSpec("trained-superbpe-96k-t77k", TokenizerKind.SUPERBPE, 96_000, 77_000),
    TrainSpec("trained-superbpe-128k-t51k", TokenizerKind.SUPERBPE, 128_000, 51_000),
    TrainSpec("trained-superbpe-128k-t102k", TokenizerKind.SUPERBPE, 128_000, 102_000),
    TrainSpec("trained-superbpe-160k-t64k", TokenizerKind.SUPERBPE, 160_000, 64_000),
    TrainSpec("trained-superbpe-160k-t128k", TokenizerKind.SUPERBPE, 160_000, 128_000),
    TrainSpec("trained-superbpe-64k-t32k", TokenizerKind.SUPERBPE, 64_000, 32_000),
    TrainSpec("trained-superbpe-48k-t24k", TokenizerKind.SUPERBPE, 48_000, 24_000),
    TrainSpec("trained-superbpe-40k-t20k", TokenizerKind.SUPERBPE, 40_000, 20_000),
    TrainSpec("trained-superbpe-32k-t16k", TokenizerKind.SUPERBPE, 32_000, 16_000),
    TrainSpec("trained-superbpe-80k-t40k", TokenizerKind.SUPERBPE, 80_000, 40_000),
    # Soak-run arms: 64k & 128k SuperBPE (t/T = 0.5) trained on the representative grug mixture,
    # with the two pretokenizer variants under test — individual-digit encoding (math) and the
    # Llama-3 production word regex. Scored at 10B/500M active over a 24h run (see EXPERIMENT_LOG).
    TrainSpec("soak-superbpe-64k", TokenizerKind.SUPERBPE, 64_000, 32_000, Pretok.SUPERBPE),
    TrainSpec("soak-superbpe-128k", TokenizerKind.SUPERBPE, 128_000, 64_000, Pretok.SUPERBPE),
    TrainSpec("soak-superbpe-64k-digits", TokenizerKind.SUPERBPE, 64_000, 32_000, Pretok.DIGITS),
    TrainSpec("soak-superbpe-128k-digits", TokenizerKind.SUPERBPE, 128_000, 64_000, Pretok.DIGITS),
    TrainSpec("soak-superbpe-64k-llama", TokenizerKind.SUPERBPE, 64_000, 32_000, Pretok.LLAMA3),
    TrainSpec("soak-superbpe-128k-llama", TokenizerKind.SUPERBPE, 128_000, 64_000, Pretok.LLAMA3),
)

# Fixed soak arms: identical config to the corresponding soak-* spec above (only ``name``
# differs), retrained after commit 11bd2f4e9c fixed `_sample_stage2_corpus` to shuffle before
# sampling. The already-deployed soak-* tokenizers above were trained on the pre-fix (English-
# only stage-2 sample) code, so they stay registered as-is; these get a new name so a re-run can
# select the fixed tokenizer via BAKEOFF_ARM without touching the confounded soak results.
_FIXED_SOAK_BASE_NAMES: tuple[str, ...] = (
    "soak-superbpe-64k",
    "soak-superbpe-128k",
    "soak-superbpe-64k-digits",
    "soak-superbpe-128k-digits",
)

FIXED_SOAK_SPECS: tuple[TrainSpec, ...] = tuple(
    replace(next(s for s in TRAIN_SPECS if s.name == base_name), name=f"{base_name}-fixed")
    for base_name in _FIXED_SOAK_BASE_NAMES
)


def read_corpus(corpus_dir: str, domains: tuple[str, ...] = CORPUS_DOMAINS) -> list[str]:
    """Load the ``<domain>.jsonl.gz`` shards :mod:`experiments.tokenize.corpus` wrote."""
    texts: list[str] = []
    for domain in domains:
        path = f"{corpus_dir}/{domain}.jsonl.gz"
        with open_url(path, "rt", encoding="utf-8", compression="gzip") as f:
            for line in f:
                text = json.loads(line).get("text")
                if text:
                    texts.append(text)
    return texts


def _save_tokenizer(tokenizer, out_dir: str) -> int:
    """Add the EOS special token, write ``tokenizer.json`` + ``tokenizer_config.json``.

    Mirrors the off-the-shelf ``superbpe-128k`` arm's shape exactly (single `<|endoftext|>`
    special token used as both BOS and UNK, empty eos/pad strings, `GPT2Tokenizer` class) so a
    trained arm loads and behaves identically through the same `levanter.load_tokenizer` path.
    Returns the final vocab size (base vocab + the added EOS token).
    """
    os.makedirs(out_dir, exist_ok=True)
    tokenizer.add_special_tokens([EOS_TOKEN])
    vocab_size = tokenizer.get_vocab_size()
    tokenizer.save(f"{out_dir}/tokenizer.json")

    config = {
        "add_prefix_space": False,
        "added_tokens_decoder": {
            str(vocab_size - 1): {
                "content": EOS_TOKEN,
                "lstrip": False,
                "normalized": False,
                "rstrip": False,
                "single_word": False,
                "special": True,
            }
        },
        "bos_token": EOS_TOKEN,
        "clean_up_tokenization_spaces": False,
        "eos_token": "",
        "model_max_length": 1_000_000_000_000_000_019_884_624_838_656,
        "pad_token": "",
        "tokenizer_class": "GPT2Tokenizer",
        "unk_token": EOS_TOKEN,
    }
    with open(f"{out_dir}/tokenizer_config.json", "w") as f:
        json.dump(config, f, indent=2)
    return vocab_size


def train_one(spec: TrainSpec, texts: list[str], out_dir: str) -> dict:
    """Train ``spec``, save it under ``out_dir``, and return its manifest row."""
    start = time.time()
    cfg = pretok_config(spec.pretok)
    if spec.kind == TokenizerKind.PLAIN_BPE:
        tokenizer = train_plain_bpe(texts, spec.vocab_size, regex_string=cfg.stage1_regex, split_digits=cfg.split_digits)
        achieved_vocab = tokenizer.get_vocab_size()
    else:
        if spec.transition_vocab_size is None:
            raise ValueError(f"{spec.name}: SuperBPE requires transition_vocab_size")
        stage1 = train_plain_bpe(
            texts, spec.transition_vocab_size, regex_string=cfg.stage1_regex, split_digits=cfg.split_digits
        )
        stage2_texts = _sample_stage2_corpus(texts, STAGE2_SAMPLE_BYTES)
        result = extend_with_superwords(
            stage1, stage2_texts, spec.vocab_size, stage2_regex=cfg.stage2_regex, split_digits=cfg.split_digits
        )
        tokenizer = result.tokenizer
        achieved_vocab = result.final_vocab_size

    final_vocab = _save_tokenizer(tokenizer, out_dir)
    elapsed = time.time() - start
    logger.info(
        "%s: vocab %d (requested %d, +eos %d) in %.1fs -> %s",
        spec.name,
        achieved_vocab,
        spec.vocab_size,
        final_vocab,
        elapsed,
        out_dir,
    )
    return {
        "name": spec.name,
        "kind": str(spec.kind),
        "pretok": str(spec.pretok),
        "requested_vocab": spec.vocab_size,
        "transition_vocab": spec.transition_vocab_size,
        "vocab_size": final_vocab,
        "train_seconds": round(elapsed, 1),
        "tokenizer_dir": out_dir,
    }


def _train_one_into(spec: TrainSpec, texts: list[str], out_dir: str) -> dict:
    # Top-level (not a closure) so it is picklable for ProcessPoolExecutor's fork workers.
    return train_one(spec, texts, out_dir)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus-dir", required=True, help="output_path from experiments.tokenize.corpus")
    ap.add_argument("--out-dir", default="experiments/tokenize/results/trained_tokenizers")
    ap.add_argument("--arms", default=None, help="comma-separated TrainSpec names (default: all)")
    ap.add_argument("--jobs", type=int, default=1, help="train this many arms concurrently (separate processes)")
    args = ap.parse_args()

    specs = TRAIN_SPECS
    if args.arms:
        wanted = set(args.arms.split(","))
        unknown = wanted - {s.name for s in TRAIN_SPECS}
        if unknown:
            raise SystemExit(f"unknown --arms {sorted(unknown)}; known: {[s.name for s in TRAIN_SPECS]}")
        specs = tuple(s for s in TRAIN_SPECS if s.name in wanted)

    texts = read_corpus(args.corpus_dir)
    total_bytes = sum(len(t.encode("utf-8")) for t in texts)
    logger.info("loaded corpus: %d docs, %.1f MB from %s", len(texts), total_bytes / 1e6, args.corpus_dir)

    if args.jobs <= 1:
        manifest = [train_one(spec, texts, f"{args.out_dir}/{spec.name}") for spec in specs]
    else:
        # Fork workers share the parent's already-loaded `texts` via copy-on-write (Linux's
        # default multiprocessing start method), so this does not duplicate the corpus in
        # memory per worker.
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            futures = {
                pool.submit(_train_one_into, spec, texts, f"{args.out_dir}/{spec.name}"): spec.name for spec in specs
            }
            manifest = []
            for future in as_completed(futures):
                manifest.append(future.result())

    manifest_path = f"{args.out_dir}/manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info("wrote %s", manifest_path)


if __name__ == "__main__":
    main()
