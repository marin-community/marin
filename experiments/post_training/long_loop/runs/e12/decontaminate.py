# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Exact task IDs, DataKit 13-grams and normalized eval-side 8-gram coverage."""

import argparse
import collections
import hashlib
import json
import re
import time
import unicodedata
from array import array
from pathlib import Path

import pyarrow.parquet as pq
from marin.datakit import decon

GRAM = decon.NGramConfig(ngram_length=13, overlap_threshold=0.5, min_matched_features=2)
NORMAL = re.compile(r"[^a-z0-9]+")
COMMON = 20


def normalized(text: str) -> str:
    return NORMAL.sub(" ", unicodedata.normalize("NFKC", text).casefold()).strip()


def grams8(text: str) -> set[int]:
    tokens = normalized(text).split()
    return {decon._bloom_hash(" ".join(tokens[i : i + 8])) for i in range(len(tokens) - 7)}


def text_strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for key in ("content", "text", "question", "prompt", "answer", "reference", "correct_answer", "solution"):
            if key in value:
                yield from text_strings(value[key])
    elif isinstance(value, list):
        for part in value:
            yield from text_strings(part)


def task_text(spec: dict) -> str:
    context = list(text_strings(spec.get("context", {}).get("events", [])))
    # Include exposed grader answer fields only. Script-grader answers hidden in packages are unavailable;
    # no sandbox package or grader execution is performed.
    grader = list(text_strings(spec.get("grader", {})))
    return "\n\n".join(context + grader)


def aliases(identity: str) -> set[str]:
    return {identity, identity.rsplit("/", 1)[-1], identity.rsplit(":", 1)[-1]}


def corpus_rows(paths: list[Path]):
    for path in paths:
        for batch in pq.ParquetFile(path).iter_batches(batch_size=64):
            for row in batch.to_pylist():
                raw = row["task_spec"]
                spec = json.loads(raw) if isinstance(raw, str) else raw
                identity = spec["id"]
                other_ids = [str(row[k]) for k in ("task_id", "source_task_id") if row.get(k)]
                yield path, identity, other_ids, task_text(spec)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--rendered", action="store_true")
    parser.add_argument("--corpus", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = time.time()
    args.output.mkdir(parents=True, exist_ok=True)
    source = [json.loads(line) for line in args.candidates.open()]
    texts = []
    identities = []
    owners = []
    for row in source:
        sha = row["source_code_sha256"]
        fields = (
            ("rendered_text", "parent_instruction", "before_text", "after_text")
            if args.rendered
            else ("code", "parent_instruction")
        )
        for field in fields:
            texts.append(row[field])
            identities.append(row["task_id"])
            owners.append((sha, field))
    actual_candidates = len(texts)
    control_specs = []
    for corpus in args.corpus:
        for _, _identity, _, text in corpus_rows([corpus]):
            if len(text.split()) >= 30:
                control_specs.extend(
                    [(str(corpus), "exact", text, True), (str(corpus), "normalized", normalized(text), True)]
                )
                break
    control_specs.append(
        (
            "negative",
            "random",
            " ".join("e12nonoverlap" + hashlib.sha256(str(i).encode()).hexdigest() for i in range(40)),
            False,
        )
    )
    for corpus, kind, text, _expected in control_specs:
        texts.append(text)
        identities.append("e12-control-never-a-task-id")
        owners.append(("control:" + corpus + kind, "control"))
    inverted8 = collections.defaultdict(list)
    needed13 = set()
    exact = collections.defaultdict(set)
    for index, text in enumerate(texts):
        needed13.update(decon._bloom_hash(feature) for feature in decon._extract_features(text, GRAM))
        for h in grams8(text):
            inverted8[h].append(index)
        norm = normalized(text)
        if len(norm) >= 40 and len(norm.split()) >= 8:
            exact[hashlib.sha256(norm.encode()).hexdigest()].add(index)
    hits13 = set()
    sample13 = {}
    hits8 = {}
    corpus_docs = []
    corpus_identities = set()
    exact_hits = collections.defaultdict(list)
    corpus_count = collections.Counter()
    print(json.dumps({"candidate_texts": len(texts), "needed13": len(needed13), "needed8": len(inverted8)}), flush=True)
    for index, (path, identity, other_ids, text) in enumerate(corpus_rows(args.corpus)):
        corpus_count[str(path)] += 1
        for value in [identity, *other_ids]:
            corpus_identities.update(aliases(value))
        for feature in decon._extract_features(text, GRAM):
            h = decon._bloom_hash(feature)
            if h in needed13:
                hits13.add(h)
                sample13.setdefault(h, identity)
        g8 = grams8(text)
        corpus_docs.append({"id": identity, "gram8_count": len(g8)})
        for h in g8:
            if h not in inverted8:
                continue
            if h not in hits8:
                hits8[h] = array("I")
            references = hits8[h]
            if references is not None:
                references.append(index)
                if len(references) > COMMON:
                    hits8[h] = None
        norm = normalized(text)
        if len(norm) >= 40 and len(norm.split()) >= 8:
            for candidate in exact.get(hashlib.sha256(norm.encode()).hexdigest(), ()):
                exact_hits[candidate].append(identity)
        if (index + 1) % 5000 == 0:
            print(json.dumps({"corpus_rows": index + 1, "elapsed": time.time() - started}), flush=True)
    per_candidate8 = [collections.Counter() for _ in texts]
    for h, references in hits8.items():
        if references is None:
            continue
        for candidate in inverted8[h]:
            per_candidate8[candidate].update(references)
    excluded = set()
    decisions = []
    for index, text in enumerate(texts):
        sha, field = owners[index]
        score, matched = decon._document_overlap_and_matches(text, hits13, GRAM)
        eight = []
        for corpus_index, matched_count in per_candidate8[index].items():
            total = corpus_docs[corpus_index]["gram8_count"]
            if matched_count >= 2 and total and matched_count / total >= 0.5:
                eight.append(
                    {
                        "id": corpus_docs[corpus_index]["id"],
                        "matched": matched_count,
                        "total": total,
                        "coverage": matched_count / total,
                    }
                )
        id_hit = bool(aliases(identities[index]) & corpus_identities)
        flagged = bool(matched or eight or exact_hits.get(index) or id_hit)
        if flagged:
            excluded.add(sha)
        decisions.append(
            {
                "source_code_sha256": sha,
                "field": field,
                "flagged": flagged,
                "exact_task_id": id_hit,
                "full_normalized_exact": exact_hits.get(index, []),
                "datakit13_max_overlap": score,
                "datakit13_flagged_features": len(set(matched)),
                "datakit13_example_corpus_ids": sorted({sample13[h] for h in matched})[:10],
                "normalized8_hits": sorted(eight, key=lambda x: x["id"])[:20],
            }
        )
    # These sentinels passed through the same corpus stream and aggregate indexes.
    controls = []
    for decision, (corpus, kind, _text, expected) in zip(decisions[actual_candidates:], control_specs, strict=True):
        if decision["flagged"] != expected:
            raise ValueError(f"Decontamination control failed: {corpus} {kind}")
        controls.append({"corpus": corpus, "kind": kind, "expected_flag": expected, "actual_flag": decision["flagged"]})
    decisions = decisions[:actual_candidates]
    all_hashes = {row["source_code_sha256"] for row in source}
    excluded.intersection_update(all_hashes)
    (args.output / "decisions.jsonl").write_text("".join(json.dumps(x) + "\n" for x in decisions))
    (args.output / "allowed-code-hashes.json").write_text(json.dumps(sorted(all_hashes - excluded)) + "\n")
    manifest = {
        "method": (
            "Exact task IDs/full normalized equality; DataKit13 per-paragraph >=0.5 and >=2 f"
            "eatures; NFKC/casefold/alnum normalized8 eval-side coverage>=0.5 and >=2 distinc"
            "t features, grams found in >20 corpus rows ignored in normalized8 only. No seman"
            "tic/paraphrase guarantee."
        ),
        "candidate_source_hashes": len(all_hashes),
        "excluded_source_hashes": len(excluded),
        "allowed_source_hashes": len(all_hashes - excluded),
        "candidate_texts": actual_candidates,
        "corpus_rows": len(corpus_docs),
        "corpus_counts": dict(corpus_count),
        "common8_features_ignored": sum(v is None for v in hits8.values()),
        "elapsed_seconds": time.time() - started,
        "controls": controls,
        "datakit_source_sha256": hashlib.sha256(Path(decon.__file__).read_bytes()).hexdigest(),
        "inputs": [
            {
                "path": str(p),
                "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
                "rows": pq.ParquetFile(p).metadata.num_rows,
            }
            for p in args.corpus
        ],
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
