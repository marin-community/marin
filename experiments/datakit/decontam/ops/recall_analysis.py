# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Recall analysis for the all-sources decon bloom — synthetic injection.

The plan:

1. Sample ``N`` records from the parquet eval corpus, balanced across
   the AA and LMH subtrees.
2. For each sample, generate four planted variants of the eval text:
     * verbatim
     * verbatim-with-prefix (1-2 sentences of unrelated prose before the eval)
     * verbatim-with-suffix
     * paraphrased (via Claude); this stresses the bloom's invariance to
       minor wording changes
3. Run the combined bloom inline (no iris) over the synthetic shard and
   record which variants get flagged at the production
   ``OVERLAP_THRESHOLD``.
4. Report recall per variant; classify which variants pass / fail.

Verbatim recall should be 100% by construction (the bloom was built from
the same eval text). The interesting numbers are the with-prefix /
with-suffix / paraphrase recall — these tell us how robust the bloom is
to the contamination patterns that show up in real training corpora.

Usage:

    ANTHROPIC_API_KEY=... uv run python \\
        experiments/datakit/decontam/ops/recall_analysis.py \\
        --bloom-dir gs://.../datakit/bloom/_combined \\
        --sample-size 50

Writes ``recall_report.tsv`` next to the script.
"""

import argparse
import logging
import os
import random
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import dupekit
import pyarrow.parquet as pq
import yaml
from marin.datakit.decon import NGramConfig, _paragraph_overlap_and_matches, bloom_paths
from pyarrow import fs as pa_fs
from rigging.filesystem import StoragePath
from rigging.log_setup import configure_logging

from experiments.datakit.decontam.all_sources_decon import (
    EVAL_ROOT,
    NGRAM_LENGTH,
    OVERLAP_THRESHOLD,
)

logger = logging.getLogger(__name__)

REPORT_PATH = Path(__file__).with_name("recall_report.tsv")

PARAPHRASE_MODEL = "claude-sonnet-4-6"
PARAPHRASE_PROMPT = """\
Paraphrase the text below. Keep all factual content -- numbers, names,
proper nouns, code, equations -- exactly as-is. Vary the wording and
sentence structure of the surrounding prose. Do NOT add commentary;
output only the paraphrase.

TEXT:
{text}
"""


@dataclass
class Variant:
    name: str
    text: str


def _read_anthropic_key() -> str:
    key = os.environ.get("ANTHROPIC_API_KEY")
    if key:
        return key
    here = os.path.dirname(os.path.abspath(__file__))
    cand = os.path.join(here, "..", "..", "..", ".marin.yaml")
    if os.path.exists(cand):
        with open(cand) as f:
            data = yaml.safe_load(f) or {}
        env = data.get("env", {})
        key = env.get("ANTHROPIC_API_KEY")
        if key:
            return key
    raise RuntimeError("ANTHROPIC_API_KEY not set and not found in .marin.yaml")


def _list_parquet_recursive(root: str) -> list[str]:
    gcs = pa_fs.GcsFileSystem()
    bare = root.removeprefix("gs://").rstrip("/")
    entries = gcs.get_file_info(pa_fs.FileSelector(bare, recursive=True))
    return sorted(f"gs://{e.path}" for e in entries if e.path.endswith(".parquet"))


def _sample_eval_records(eval_root: str, n: int, seed: int) -> Iterator[tuple[str, str, str]]:
    """Yield (eval_id, eval_text, subtree_key) for ``n`` records.

    Two-stage sampler that doesn't iterate the whole 1.8M-record corpus:

    1. Uniformly pick ``n`` parquet files from the eval tree (with
       replacement). Files have varying record counts (~80-58k), so this
       biases toward small evals -- acceptable for a smoke-test recall
       check across the eval landscape.
    2. For each picked file, open it and select one random non-empty
       record by absolute index via parquet metadata.
    """
    files = _list_parquet_recursive(eval_root)
    rng = random.Random(seed)
    gcs = pa_fs.GcsFileSystem()
    picked_files = [rng.choice(files) for _ in range(n)]
    for f in picked_files:
        bare = f.removeprefix("gs://")
        rel = f.removeprefix(eval_root.rstrip("/") + "/")
        subtree_key = "/".join(rel.split("/")[:2])
        pf = pq.ParquetFile(bare, filesystem=gcs)
        total = pf.metadata.num_rows
        if total == 0:
            continue
        target_idx = rng.randint(0, total - 1)
        # Walk batches until we cross ``target_idx``; cheap because each
        # eval file is small (a few thousand rows max).
        seen = 0
        for batch in pf.iter_batches(columns=["id", "text"], batch_size=2048):
            ids = batch.column("id").to_pylist()
            texts = batch.column("text").to_pylist()
            if seen + len(ids) <= target_idx:
                seen += len(ids)
                continue
            local = target_idx - seen
            eid, t = ids[local], texts[local]
            if t:
                yield (str(eid), str(t), subtree_key)
            break


def _paraphrase(client, text: str) -> str | None:
    try:
        resp = client.messages.create(
            model=PARAPHRASE_MODEL,
            max_tokens=2048,
            messages=[{"role": "user", "content": PARAPHRASE_PROMPT.format(text=text[:8000])}],
        )
    except Exception as exc:
        logger.warning("paraphrase api failed: %s", exc)
        return None
    return resp.content[0].text if resp.content else None


# Short unrelated prose stitched around the eval text. Deliberately
# benign so it doesn't itself trigger the bloom.
_FILLER_PREFIX = (
    "The following passage was reconstructed from notes taken during a "
    "lecture last spring. It has been edited lightly for clarity. The "
    "instructor's broader theme was perception and inference, but the "
    "concrete example we are about to read stands on its own."
)
_FILLER_SUFFIX = (
    "We pause here. There will be exercises related to the above in a "
    "later chapter, but for now consider how the structure of the "
    "argument differs from the ones we have seen in earlier units."
)


def _variants(eval_text: str, paraphrase: str | None) -> list[Variant]:
    out = [
        Variant("verbatim", eval_text),
        Variant("with_prefix", _FILLER_PREFIX + "\n\n" + eval_text),
        Variant("with_suffix", eval_text + "\n\n" + _FILLER_SUFFIX),
    ]
    if paraphrase:
        out.append(Variant("paraphrase", paraphrase))
    return out


def _bloom_flags(bf: dupekit.Bloom, ngram: NGramConfig, text: str) -> tuple[bool, float]:
    """Same scoring as decon.decon_to_parquet's mark_shard: max paragraph overlap."""
    max_score = 0.0
    for para in text.split("\n"):
        if not para:
            continue
        score, _ = _paragraph_overlap_and_matches(para, bf, ngram)
        if score > max_score:
            max_score = score
    return (max_score >= ngram.overlap_threshold and max_score > 0, max_score)


def _load_bloom(bloom_dir: str) -> dupekit.Bloom:
    bloom_path, _ = bloom_paths(bloom_dir)
    return dupekit.Bloom.load_bytes(StoragePath(bloom_path).read_bytes())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bloom-dir", required=True, help="GCS dir with _bloom/filter.bin")
    parser.add_argument("--sample-size", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-paraphrase", action="store_true", help="Skip the Claude paraphrase variant")
    args = parser.parse_args()

    configure_logging(logging.INFO)
    bf = _load_bloom(args.bloom_dir)
    ngram = NGramConfig(ngram_length=NGRAM_LENGTH, overlap_threshold=OVERLAP_THRESHOLD)

    samples = list(_sample_eval_records(EVAL_ROOT, args.sample_size, args.seed))
    logger.info("sampled %d eval records", len(samples))

    if args.skip_paraphrase:
        client = None
    else:
        from anthropic import Anthropic  # noqa: PLC0415  # optional dep: anthropic

        client = Anthropic(api_key=_read_anthropic_key())

    counts: dict[str, dict[str, int]] = {}
    with REPORT_PATH.open("w", encoding="utf-8") as f:
        f.write("eval_id\tsubtree\tvariant\tflagged\tmax_overlap\teval_chars\n")
        for i, (eid, text, subtree) in enumerate(samples, 1):
            paraphrase = _paraphrase(client, text) if client is not None else None
            for v in _variants(text, paraphrase):
                flagged, score = _bloom_flags(bf, ngram, v.text)
                counts.setdefault(v.name, {"flagged": 0, "total": 0})
                counts[v.name]["total"] += 1
                if flagged:
                    counts[v.name]["flagged"] += 1
                f.write(f"{eid}\t{subtree}\t{v.name}\t{int(flagged)}\t{score:.3f}\t{len(text)}\n")
            if i % 10 == 0:
                logger.info("processed %d/%d", i, len(samples))

    print()
    print(f"recall per variant (sample={args.sample_size}):")
    for v_name, c in sorted(counts.items()):
        r = c["flagged"] / c["total"] if c["total"] else 0.0
        print(f"  {v_name:>12s}: {c['flagged']:>4d}/{c['total']:>4d} = {r:.3f}")
    print(f"\nreport: {REPORT_PATH}")


if __name__ == "__main__":
    main()
