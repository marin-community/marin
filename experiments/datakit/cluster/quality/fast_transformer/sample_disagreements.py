# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Surface the documents two quality scorers disagree about most, for reading.

Aggregate metrics can move the wrong way for the right-looking reasons. A candidate
that improved every summary statistic — wider bucket spread, better cross-domain
parity, higher within-domain variance — turned out to promote gym timetables and
demote worked Fourier derivations, and no aggregate showed it. Reading twenty
documents did, in minutes.

So this is deliberately not a metric. It joins two scorers on the same documents,
picks the pairs where they most disagree, and prints them with enough text to judge.
The reader decides which scorer is closer to right, and the tally of those decisions
is the finding.

Both directions are sampled, because they fail differently: a scorer that promotes
junk wastes training budget, while one that demotes good documents discards data
that cannot be recovered later. Sampling is stratified by source so one prolific
source cannot fill the page.
"""

import argparse
import collections
import logging
import random
from dataclasses import dataclass

import pyarrow.parquet as pq
from rigging.filesystem import StoragePath
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.log_setup import configure_logging

logger = logging.getLogger(__name__)

DEFAULT_PER_DIRECTION = 12
# Enough of a document to judge its type and substance without flooding the page.
EXCERPT_CHARS = 700


def _read_dir(path: str, columns: list[str]) -> dict[str, list]:
    out: dict[str, list] = {c: [] for c in columns}
    shards = sorted(str(m) for m in StoragePath(f"{path.rstrip('/')}/*.parquet").glob())
    if not shards:
        raise ValueError(f"no parquet shards under {path}")
    for shard in shards:
        with StoragePath(shard).open("rb") as handle:
            table = pq.ParquetFile(handle).read(columns=columns)
        for c in columns:
            out[c].extend(table.column(c).to_pylist())
    return out


@dataclass(frozen=True)
class Disagreement:
    """One document and what each scorer said about it."""

    doc_id: str
    source: str
    text: str
    lhs_score: float
    lhs_bucket: int
    rhs_score: float
    rhs_bucket: int

    @property
    def gap(self) -> int:
        """Buckets the right-hand scorer places it above the left-hand one."""
        return self.rhs_bucket - self.lhs_bucket


def _stratified(pool: list[Disagreement], want: int, rng: random.Random) -> list[Disagreement]:
    """Take ``want`` rows, spreading them over sources round-robin."""
    by_source: dict[str, list[Disagreement]] = collections.defaultdict(list)
    for row in pool:
        by_source[row.source].append(row)
    for rows in by_source.values():
        rng.shuffle(rows)
    picked: list[Disagreement] = []
    while len(picked) < want and any(by_source.values()):
        for rows in list(by_source.values()):
            if rows and len(picked) < want:
                picked.append(rows.pop())
    return picked


def find_disagreements(
    *, docs_dir: str, lhs: tuple[str, str], rhs: tuple[str, str], per_direction: int, seed: int
) -> dict[str, list[Disagreement]]:
    """The sharpest disagreements in each direction, stratified by source."""
    meta = _read_dir(docs_dir, ["id", "text", "source"])
    by_id = {doc_id: i for i, doc_id in enumerate(meta["id"])}

    scores: dict[str, dict[str, tuple[float, int]]] = {}
    for name, root in (lhs, rhs):
        cols = _read_dir(f"{root.rstrip('/')}/outputs/main", ["id", "score", "quality_bucket"])
        scores[name] = {
            i: (s, b) for i, s, b in zip(cols["id"], cols["score"], cols["quality_bucket"], strict=True) if i in by_id
        }

    lhs_name, rhs_name = lhs[0], rhs[0]
    shared = set(scores[lhs_name]) & set(scores[rhs_name])
    logger.info("sample_disagreements: %d documents scored by both", len(shared))

    rows = []
    for doc_id in shared:
        ls, lb = scores[lhs_name][doc_id]
        rs, rb = scores[rhs_name][doc_id]
        i = by_id[doc_id]
        rows.append(
            Disagreement(
                doc_id=doc_id,
                source=meta["source"][i],
                text=meta["text"][i],
                lhs_score=ls,
                lhs_bucket=lb,
                rhs_score=rs,
                rhs_bucket=rb,
            )
        )

    rng = random.Random(seed)
    up = [r for r in rows if r.gap >= 2]  # rhs rates it far higher
    down = [r for r in rows if r.gap <= -2]  # rhs rates it far lower
    logger.info("sample_disagreements: %d up (%s higher), %d down", len(up), rhs_name, len(down))
    return {
        f"{rhs_name} much HIGHER": _stratified(up, per_direction, rng),
        f"{rhs_name} much LOWER": _stratified(down, per_direction, rng),
    }


def _print(groups: dict[str, list[Disagreement]], lhs_name: str, rhs_name: str) -> None:
    for title, rows in groups.items():
        print(f"\n{'=' * 78}\n{title}  ({len(rows)} documents)\n{'=' * 78}")
        for n, row in enumerate(rows, 1):
            left = f"{lhs_name} b{row.lhs_bucket} {row.lhs_score:.3f}"
            right = f"{rhs_name} b{row.rhs_bucket} {row.rhs_score:.3f}"
            print(f"\n--- {n}. {row.source}   {left}  |  {right}")
            text = " ".join((row.text or "").split())
            print(f"    {text[:EXCERPT_CHARS]}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--docs-dir", required=True, help="evaluation documents (id, text, source)")
    parser.add_argument("--lhs", required=True, metavar="NAME=PATH")
    parser.add_argument("--rhs", required=True, metavar="NAME=PATH")
    parser.add_argument("--per-direction", type=int, default=DEFAULT_PER_DIRECTION)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    configure_logging(logging.INFO)
    configure_coreweave_s3()
    lhs = tuple(args.lhs.split("=", 1))
    rhs = tuple(args.rhs.split("=", 1))
    groups = find_disagreements(
        docs_dir=args.docs_dir, lhs=lhs, rhs=rhs, per_direction=args.per_direction, seed=args.seed
    )
    _print(groups, lhs[0], rhs[0])


if __name__ == "__main__":
    main()
