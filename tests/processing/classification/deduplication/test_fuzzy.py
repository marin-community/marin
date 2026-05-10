# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import os
import random
import string
from pathlib import Path

import dupekit
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from fray import LocalClient, set_current_client
from marin.datakit.normalize import NormalizedData, generate_id, normalize_to_parquet
from marin.processing.classification.deduplication.fuzzy_dups import compute_fuzzy_dups_attrs
from marin.processing.classification.deduplication.fuzzy_minhash import (
    MinHashAttrData,
    MinHashParams,
    compute_minhash_attrs,
)
from zephyr import write_jsonl_file, write_parquet_file

TEST_MINHASH_PARAMS = MinHashParams(num_perms=286, num_bands=26, ngram_size=5, seed=42)


@pytest.fixture(autouse=True)
def flow_backend_ctx():
    with set_current_client(LocalClient()):
        yield


def _normalize(input_dir: str, output_dir: str) -> NormalizedData:
    """Normalize a fox-corpus shard directory into a NormalizedData dataset."""
    return normalize_to_parquet(input_path=input_dir, output_path=output_dir)


def _read_main_records(source: NormalizedData) -> dict[str, dict]:
    """Return a mapping from generated ``id`` to the full main-output record."""
    out: dict[str, dict] = {}
    for pf in sorted(Path(source.main_output_dir).glob("*.parquet")):
        for record in pq.read_table(str(pf)).to_pylist():
            out[record["id"]] = record
    return out


def _read_cluster_attrs(attr_dir: str) -> list[dict]:
    """Return every cluster-member row (flat list) under *attr_dir*."""
    rows: list[dict] = []
    for pf in sorted(Path(attr_dir).glob("*.parquet")):
        rows.extend(pq.read_table(str(pf)).to_pylist())
    return rows


def _write_minhash_attr_dataset(
    *,
    output_dir: str,
    source_main_dir: str,
    rows: list[dict],
) -> MinHashAttrData:
    """Write a one-shard MinHash attr dataset for focused fuzzy-dup tests."""
    attr_dir = os.path.join(output_dir, "outputs")
    Path(attr_dir).mkdir(parents=True, exist_ok=True)
    write_parquet_file(rows, os.path.join(attr_dir, "part-00000.parquet"))
    return MinHashAttrData(
        params=TEST_MINHASH_PARAMS,
        source_main_dir=source_main_dir,
        attr_dir=attr_dir,
        counters={},
    )


def test_minhash_attrs_co_partitioned_with_source(fox_corpus):
    """Each source shard produces a same-named MinHash attr parquet with {id, buckets}."""
    norm_dir = os.path.join(fox_corpus["output_dir"], "normalized")
    minhash_dir = os.path.join(fox_corpus["output_dir"], "minhash")

    source = _normalize(fox_corpus["test_dir"], norm_dir)
    minhash = compute_minhash_attrs(source=source, output_path=minhash_dir)

    assert minhash.source_main_dir == source.main_output_dir
    assert minhash.params.num_perms == 286
    assert minhash.params.num_bands == 26

    source_basenames = {p.name for p in Path(source.main_output_dir).glob("*.parquet")}
    attr_basenames = {p.name for p in Path(minhash.attr_dir).glob("*.parquet")}
    assert source_basenames == attr_basenames
    assert source_basenames  # non-empty

    # At least one non-empty shard exists with the expected {id, buckets} schema.
    # Empty source shards produce empty attr parquets with no schema, which we skip.
    seen_non_empty = False
    for pf in Path(minhash.attr_dir).glob("*.parquet"):
        rows = pq.read_table(str(pf)).to_pylist()
        if not rows:
            continue
        seen_non_empty = True
        rec = rows[0]
        assert isinstance(rec["id"], str)
        assert isinstance(rec["buckets"], list)
        assert all(isinstance(b, str) for b in rec["buckets"])
    assert seen_non_empty, "expected at least one non-empty MinHash attr shard"
    assert minhash.counters["minhash/documents"] >= 1


def test_fuzzy_dups_single_source_schema_and_pair(fox_corpus):
    """Attr rows cover every cluster member; singletons get no row.

    Uses the test corpus's fuzzy pair (``test_contaminated_1`` ~
    ``test_high_overlap``, one-word diff) to exercise a real cluster of 2
    and verifies:
    * both members get an attr row,
    * rows carry ``dup_cluster_id`` and ``is_cluster_canonical``,
    * all rows for one cluster share the same ``dup_cluster_id``,
    * exactly one member per cluster is canonical,
    * unique docs have no attr row.
    """
    norm_dir = os.path.join(fox_corpus["output_dir"], "normalized")
    minhash_dir = os.path.join(fox_corpus["output_dir"], "minhash")
    dups_dir = os.path.join(fox_corpus["output_dir"], "fuzzy_dups")

    source = _normalize(fox_corpus["test_dir"], norm_dir)
    minhash = compute_minhash_attrs(source=source, output_path=minhash_dir)
    dups = compute_fuzzy_dups_attrs(inputs=[minhash], output_path=dups_dir, max_parallelism=4)

    assert dups.params == minhash.params
    per_source = dups.sources[source.main_output_dir]

    by_id = _read_main_records(source)
    rows = _read_cluster_attrs(per_source.attr_dir)
    by_source_id = {by_id[r["id"]]["source_id"]: r for r in rows if r["id"] in by_id}

    # The fuzzy pair is a cluster of 2: both members must have an attr row,
    # sharing one dup_cluster_id, with exactly one canonical.
    pair = {"test_contaminated_1", "test_high_overlap"}
    assert pair <= by_source_id.keys(), f"missing attr rows for pair: {pair - by_source_id.keys()}"
    cluster_ids = {by_source_id[s]["attributes"]["dup_cluster_id"] for s in pair}
    assert len(cluster_ids) == 1, f"pair should share a dup_cluster_id; got {cluster_ids}"
    canonicals = [s for s in pair if by_source_id[s]["attributes"]["is_cluster_canonical"]]
    assert len(canonicals) == 1, f"exactly one canonical expected; got {canonicals}"

    # Unique docs never have attr rows (no cluster → no annotation).
    assert "test_unique_1" not in by_source_id
    assert "test_unique_2" not in by_source_id


def test_fuzzy_dups_multi_source_per_source_attr_trees(fox_corpus):
    """Two MinHashAttrData inputs produce two per-source attr trees.

    Cross-source exact-text duplicates (e.g. ``train_arctic_1`` ==
    ``test_contaminated_1`` byte-identical → same normalized id in both
    datasets) must be detected as a 2-member cluster rather than collapsing
    into a single node. Each side independently carries its own attr row for
    the shared content hash, with a shared ``dup_cluster_id`` and exactly one
    canonical across the pair.

    This test targets multi-source fuzzy dedup behavior directly. Normalization
    and MinHash generation already have separate coverage above.
    """
    train_main_dir = os.path.join(fox_corpus["output_dir"], "train_main")
    test_main_dir = os.path.join(fox_corpus["output_dir"], "test_main")
    train_mh = _write_minhash_attr_dataset(
        output_dir=os.path.join(fox_corpus["output_dir"], "mh_train"),
        source_main_dir=train_main_dir,
        rows=[
            {
                "id": generate_id("Arctic predators have superior auditory capabilities for hunting beneath snow."),
                "buckets": ["shared-arctic"],
            },
            {
                "id": generate_id("Red canids inhabit northern territories worldwide."),
                "buckets": ["shared-red"],
            },
            {
                "id": generate_id("Newborn kits emerge sightless and vulnerable."),
                "buckets": ["train-unique"],
            },
        ],
    )
    test_mh = _write_minhash_attr_dataset(
        output_dir=os.path.join(fox_corpus["output_dir"], "mh_test"),
        source_main_dir=test_main_dir,
        rows=[
            {
                "id": generate_id("Arctic predators have superior auditory capabilities for hunting beneath snow."),
                "buckets": ["shared-arctic"],
            },
            {
                "id": generate_id("Red canids inhabit northern territories worldwide."),
                "buckets": ["shared-red"],
            },
            {
                "id": generate_id("Rapid runners represent the most diminutive wild dogs."),
                "buckets": ["test-unique"],
            },
        ],
    )

    dups = compute_fuzzy_dups_attrs(
        inputs=[train_mh, test_mh],
        output_path=os.path.join(fox_corpus["output_dir"], "fuzzy_dups"),
        max_parallelism=1,
    )

    assert set(dups.sources.keys()) == {train_main_dir, test_main_dir}
    for per_source in dups.sources.values():
        assert per_source.attr_dir.rsplit("/", 1)[-1].startswith("source_"), per_source.attr_dir
        assert Path(per_source.attr_dir).exists()

    def rows_by_id(main_dir: str) -> dict[str, dict]:
        return {r["id"]: r for r in _read_cluster_attrs(dups.sources[main_dir].attr_dir)}

    train_rows = rows_by_id(train_main_dir)
    test_rows = rows_by_id(test_main_dir)

    # Each cross-source byte-identical text must appear as an attr row on both
    # sides (keyed by the same content hash), share a dup_cluster_id, and have
    # exactly one canonical across the pair.
    for shared_text in (
        "Arctic predators have superior auditory capabilities for hunting beneath snow.",
        "Red canids inhabit northern territories worldwide.",
    ):
        content_id = generate_id(shared_text)
        assert content_id in train_rows, f"missing train attr row for {shared_text!r}"
        assert content_id in test_rows, f"missing test attr row for {shared_text!r}"
        a, b = train_rows[content_id]["attributes"], test_rows[content_id]["attributes"]
        assert a["dup_cluster_id"] == b["dup_cluster_id"], f"{shared_text!r}: dup_cluster_id mismatch"
        assert (
            a["is_cluster_canonical"] != b["is_cluster_canonical"]
        ), f"{shared_text!r}: exactly one canonical expected across pair"


def test_fuzzy_dups_rejects_param_mismatch(fox_corpus):
    """Inputs with mismatched MinHash params must be rejected up front."""
    source = _normalize(fox_corpus["test_dir"], os.path.join(fox_corpus["output_dir"], "norm"))
    a = compute_minhash_attrs(source=source, output_path=os.path.join(fox_corpus["output_dir"], "mh_a"))
    # Same num_perms, different num_bands → still divisible, but params differ.
    b = compute_minhash_attrs(
        source=source,
        output_path=os.path.join(fox_corpus["output_dir"], "mh_b"),
        num_bands=22,  # 286 % 22 == 0
    )

    with pytest.raises(ValueError, match=r"identical MinHash params"):
        compute_fuzzy_dups_attrs(
            inputs=[a, b],
            output_path=os.path.join(fox_corpus["output_dir"], "fuzzy_dups"),
            max_parallelism=4,
        )


def test_fuzzy_dups_rejects_duplicate_source(fox_corpus):
    """Two inputs pointing to the same ``source_main_dir`` must be rejected to avoid output clobbering."""
    source = _normalize(fox_corpus["test_dir"], os.path.join(fox_corpus["output_dir"], "norm"))
    mh = compute_minhash_attrs(source=source, output_path=os.path.join(fox_corpus["output_dir"], "mh"))

    with pytest.raises(ValueError, match=r"Duplicate source_main_dir"):
        compute_fuzzy_dups_attrs(
            inputs=[mh, mh],
            output_path=os.path.join(fox_corpus["output_dir"], "fuzzy_dups"),
            max_parallelism=4,
        )


# ---------------------------------------------------------------------------
# Char-5-gram Jaccard recall / precision tests.
#
# The dupekit MinHash pipeline shingles by character (rust/dupekit/src/
# minhash_ops.rs:69-76: text.chars().windows(ngram_size)), so char-Jaccard
# directly governs LSH collision probability. We construct text from a
# lowercase-only alphabet so dupekit's CleanText (lowercase + strip punct +
# collapse whitespace) is the identity, and the Jaccard we measure on the
# raw string equals what the system sees internally.
# ---------------------------------------------------------------------------

_CHAR_VOCAB = string.ascii_lowercase


def _char_5grams(text: str) -> set[str]:
    return {text[i : i + 5] for i in range(len(text) - 4)}


def _char_5gram_jaccard(a: str, b: str) -> float:
    ga, gb = _char_5grams(a), _char_5grams(b)
    return len(ga & gb) / len(ga | gb) if (ga | gb) else 1.0


def _make_pair_with_char_5gram_jaccard(seed: int, target_j: float, n_chars: int = 1000) -> tuple[str, str]:
    """Build (a, b) with char-5-gram-Jaccard(a, b) ≈ ``target_j``.

    A is ``n_chars`` random lowercase letters. B differs from A at ``k``
    well-spaced positions (each ≥5 apart, and bounded away from the edges)
    so each substitution kills exactly 5 char-5-grams from the intersection
    and adds 5 novel ones to the union, giving::

        J = (M - 5k) / (M + 5k),   M = n_chars - 4

    Solve for k: ``k = round(M*(1-J) / (5*(1+J)))``. Each substituted char
    is replaced with a different alphabet letter; with a 1000-char random
    backbone, accidental collisions of new 5-grams with existing ones occur
    at ~8e-5 per gram (996 unique grams over 26^5 possibilities), small
    enough to ignore at the construction tolerances asserted below.
    """
    M = n_chars - 4
    k = round(M * (1.0 - target_j) / (5.0 * (1.0 + target_j)))

    rng = random.Random(seed)
    a_chars = [rng.choice(_CHAR_VOCAB) for _ in range(n_chars)]

    # Restrict to non-edge positions so each substitution kills exactly 5
    # 5-grams. Greedy pick with mutual spacing ≥5.
    chosen: list[int] = []
    candidates = list(range(4, n_chars - 4))
    while len(chosen) < k:
        if not candidates:
            raise RuntimeError(f"could not place {k} substitutions ≥5 apart in {n_chars} chars")
        p = rng.choice(candidates)
        chosen.append(p)
        candidates = [c for c in candidates if abs(c - p) >= 5]

    b_chars = list(a_chars)
    for pos in chosen:
        b_chars[pos] = rng.choice([c for c in _CHAR_VOCAB if c != a_chars[pos]])

    return "".join(a_chars), "".join(b_chars)


def _dupekit_pipeline(params: MinHashParams) -> list:
    return [
        dupekit.Transformation.CleanText(input_col="text", output_col="clean_text"),
        dupekit.Transformation.MinHash(
            input_col="clean_text",
            output_col="signature",
            num_perms=params.num_perms,
            ngram_size=params.ngram_size,
            seed=params.seed,
        ),
        dupekit.Transformation.MinHashLSH(input_col="signature", output_col="buckets", num_bands=params.num_bands),
        dupekit.Transformation.SelectColumns(columns=["id", "buckets"]),
    ]


def _shared_lsh_bucket(text_a: str, text_b: str) -> bool:
    """Return True iff (a, b) share at least one MinHash-LSH bucket."""
    batch = pa.RecordBatch.from_pylist([{"id": "a", "text": text_a}, {"id": "b", "text": text_b}])
    out = dupekit.transform(batch, _dupekit_pipeline(TEST_MINHASH_PARAMS))
    return bool(set(out["buckets"][0].as_py()) & set(out["buckets"][1].as_py()))


# Recall: at TEST_MINHASH_PARAMS (b=26, r=11),
#   P(collide | char-J=0.95) = 1 - (1 - 0.95^11)^26 ≈ 1 - 2e-10
# so the assertion is effectively deterministic across all parametrizations.
@pytest.mark.parametrize("seed", range(20))
@pytest.mark.parametrize("target_j", [0.95, 0.97, 0.99])
def test_high_char_5gram_jaccard_pairs_share_lsh_bucket(seed: int, target_j: float):
    a, b = _make_pair_with_char_5gram_jaccard(seed, target_j)
    measured = _char_5gram_jaccard(a, b)
    # At n_chars=1000 the round() drift on k is well under 0.005. Wider drift
    # would mean the construction itself is broken, not the system.
    assert abs(measured - target_j) < 0.005, f"construction off-target: requested {target_j}, got {measured:.4f}"
    assert _shared_lsh_bucket(a, b), f"high-Jaccard pair (char-J={measured:.4f}) failed to share an LSH bucket"


# Precision: at (b=26, r=11),
#   P(collide | char-J=0.5) = 1 - (1 - 0.5^11)^26 ≈ 1.27%
#   P(collide | char-J=0.3) = 1 - (1 - 0.3^11)^26 ≈ 4.6e-5
# Over 50 seeds we expect ~0-1 collisions at J=0.5 and ~0 at J=0.3. Cap at 5
# leaves slack for parameter changes (e.g. dropping to b=20) without flaking,
# while still failing if precision degrades by ~5x.
@pytest.mark.parametrize("target_j", [0.3, 0.5])
def test_low_char_5gram_jaccard_rarely_collides(target_j: float):
    n_seeds = 50
    collisions = sum(_shared_lsh_bucket(*_make_pair_with_char_5gram_jaccard(seed, target_j)) for seed in range(n_seeds))
    assert collisions <= 5, f"{collisions}/{n_seeds} pairs collided at char-J={target_j} (expected ≤5)"


def test_high_char_5gram_jaccard_pair_clusters_end_to_end(tmp_path: Path):
    """A constructed J≈0.97 char-5-gram pair gets one shared dup_cluster_id.

    The parametrized LSH-level tests above cover recall across the Jaccard
    band; this single case exercises the full bucket → connected-components
    → per-source attr-output path.
    """
    a, b = _make_pair_with_char_5gram_jaccard(seed=0, target_j=0.97)
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    write_jsonl_file(
        [{"id": "doc_a", "text": a}, {"id": "doc_b", "text": b}],
        str(src_dir / "shard_0.jsonl.gz"),
    )

    source = _normalize(str(src_dir), str(tmp_path / "norm"))
    minhash = compute_minhash_attrs(source=source, output_path=str(tmp_path / "minhash"))
    dups = compute_fuzzy_dups_attrs(inputs=[minhash], output_path=str(tmp_path / "dups"), max_parallelism=1)

    by_id = _read_main_records(source)
    rows = _read_cluster_attrs(dups.sources[source.main_output_dir].attr_dir)
    by_source_id = {by_id[r["id"]]["source_id"]: r for r in rows if r["id"] in by_id}

    assert {"doc_a", "doc_b"} <= by_source_id.keys(), "both docs must have attr rows"
    cluster_ids = {by_source_id[s]["attributes"]["dup_cluster_id"] for s in ("doc_a", "doc_b")}
    assert len(cluster_ids) == 1, f"expected one shared dup_cluster_id; got {cluster_ids}"
    canonicals = [s for s in ("doc_a", "doc_b") if by_source_id[s]["attributes"]["is_cluster_canonical"]]
    assert len(canonicals) == 1, f"expected exactly one canonical; got {canonicals}"


# ---------------------------------------------------------------------------
# Real-world parser-variant regression tests.
#
# Fixtures live in the HF dataset at PARSER_VARIANTS_REPO (config
# parser_variants), pinned to PARSER_VARIANTS_REVISION in conftest.py. They
# contain text outputs of the same Wikipedia page extracted by trafilatura,
# html2text, and readability-lxml from a Common Crawl WARC capture. To
# refresh or extend, see resources/parser_variants/generate_test_examples.py
# (fetch + parse) and upload_test_examples.py (push to HF).
# ---------------------------------------------------------------------------


def _run_dedup_on_corpus(tmp_path: Path, docs: list[dict]) -> dict[str, dict]:
    """Run normalize -> minhash -> fuzzy_dups on a list of ``{id, text}`` docs."""
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    write_jsonl_file(docs, str(src_dir / "shard_0.jsonl.gz"))

    source = _normalize(str(src_dir), str(tmp_path / "norm"))
    minhash = compute_minhash_attrs(source=source, output_path=str(tmp_path / "minhash"))
    dups = compute_fuzzy_dups_attrs(inputs=[minhash], output_path=str(tmp_path / "dups"), max_parallelism=1)

    by_id = _read_main_records(source)
    rows = _read_cluster_attrs(dups.sources[source.main_output_dir].attr_dir)
    return {by_id[r["id"]]["source_id"]: r for r in rows if r["id"] in by_id}


def _cluster_id(by_source_id: dict[str, dict], source_id: str) -> str | None:
    """Return the dup_cluster_id for *source_id*, or None if it has no attr row (singleton)."""
    row = by_source_id.get(source_id)
    return row["attributes"]["dup_cluster_id"] if row else None


@pytest.mark.data_integration
def test_html_parser_variants_cluster_per_article(tmp_path: Path, parser_variants_docs, parser_variants_articles):
    """Trafilatura and readability outputs of one Wikipedia article cluster.

    Pinning current pipeline behavior on the HF-hosted fixtures: at the default
    MinHash params (num_perms=286, num_bands=26), trafilatura and readability
    extract sufficiently similar text from the same page (measured char-Jaccard
    ~0.89 on these fixtures) to share an LSH bucket and end up in the same
    dup_cluster_id. Cross-article pairs do not cluster (~0.22).

    The html2text variant is intentionally NOT asserted here; see
    ``test_html_parser_variants_all_three_cluster_per_article`` for the gap.
    """
    by_source_id = _run_dedup_on_corpus(tmp_path, parser_variants_docs)
    assert parser_variants_articles, "no parser-variant fixtures discovered"

    article_to_cluster: dict[str, str] = {}
    for article in parser_variants_articles:
        traf_id = f"{article}__trafilatura"
        read_id = f"{article}__readability"
        traf_cluster = _cluster_id(by_source_id, traf_id)
        read_cluster = _cluster_id(by_source_id, read_id)
        assert traf_cluster is not None, f"{traf_id} has no attr row (unexpected singleton)"
        assert read_cluster is not None, f"{read_id} has no attr row (unexpected singleton)"
        assert (
            traf_cluster == read_cluster
        ), f"{article}: trafilatura ({traf_cluster!r}) and readability ({read_cluster!r}) did not cluster together"
        article_to_cluster[article] = traf_cluster

    cluster_ids = list(article_to_cluster.values())
    assert len(set(cluster_ids)) == len(
        cluster_ids
    ), f"distinct articles ended up in the same cluster: {article_to_cluster}"


@pytest.mark.data_integration
@pytest.mark.xfail(
    strict=True,
    reason=(
        "html2text preserves Wikipedia nav/sidebar/edit-links/citations as inline text. "
        "The resulting char-Jaccard with trafilatura/readability is ~0.5, below the LSH "
        "threshold at b=26, r=11, so the pipeline does not cluster these as duplicates "
        "today. When boilerplate handling improves (pre-strip, threshold tune, or "
        "extractor-specific cleanup), this xfail flips and the main parser-variant test "
        "should be tightened to assert all three variants cluster."
    ),
)
def test_html_parser_variants_all_three_cluster_per_article(
    tmp_path: Path, parser_variants_docs, parser_variants_articles
):
    """Aspirational: every parser variant of one article shares one dup_cluster_id."""
    by_source_id = _run_dedup_on_corpus(tmp_path, parser_variants_docs)
    for article in parser_variants_articles:
        clusters = {_cluster_id(by_source_id, f"{article}__{p}") for p in ("trafilatura", "html2text", "readability")}
        assert (
            len(clusters) == 1 and None not in clusters
        ), f"{article}: parser variants split across clusters: {clusters}"


@pytest.mark.data_integration
def test_same_site_distinct_bodies_do_not_cluster(tmp_path: Path, same_site_distinct_docs):
    """Distinct articles from one site (heavy shared chrome) must not cluster.

    Pinned regression: even with a chrome-preserving parser (html2text) the
    pipeline must distinguish articles by their main content, not their
    template. Each input doc is expected to be a singleton — no attr row.
    Any clustering of a pair indicates over-merge based on shared chrome.

    The fixtures are Wikipedia pages on disjoint topics (Photography,
    Photosynthesis, Quantum mechanics, Roman Empire). BBC/Guardian/etc.
    would have been more typical "news boilerplate" candidates but none of
    them allow Common Crawl via robots.txt, so Wikipedia is the realistic
    floor for shared-template + distinct-body in the wild.
    """
    by_source_id = _run_dedup_on_corpus(tmp_path, same_site_distinct_docs)
    assert not by_source_id, f"distinct same-site articles clustered (over-merge): {sorted(by_source_id.keys())}"


@pytest.mark.data_integration
def test_wikipedia_revisions_cluster_per_article(tmp_path: Path, wikipedia_revisions_docs, wikipedia_revisions_articles):
    """Different temporal captures of one Wikipedia article must cluster.

    Recall regression on temporal drift: the dedup pipeline must recognise
    snapshots of the same URL across years (paragraphs added, citations
    refreshed) as the same logical document. All rows whose
    ``article_slug`` matches must share one ``dup_cluster_id``, and rows
    with different slugs must not cross-cluster.

    Fixture char-Jaccard between same-article revisions measures around
    0.76-0.85; cross-article around 0.21. The MinHash params keep the
    same-article pairs reliably above the LSH collision threshold.
    """
    by_source_id = _run_dedup_on_corpus(tmp_path, wikipedia_revisions_docs)
    assert wikipedia_revisions_articles, "no revision fixtures discovered"

    article_to_cluster: dict[str, str] = {}
    for article in wikipedia_revisions_articles:
        variants = [sid for sid in by_source_id if sid.startswith(f"{article}__")]
        assert variants, f"no attr rows for revisions of {article!r} (unexpected singletons)"
        clusters = {by_source_id[sid]["attributes"]["dup_cluster_id"] for sid in variants}
        assert len(clusters) == 1, f"{article}: revisions split across clusters: {clusters}"
        article_to_cluster[article] = clusters.pop()

    cluster_ids = list(article_to_cluster.values())
    assert len(set(cluster_ids)) == len(cluster_ids), f"distinct articles clustered together: {article_to_cluster}"


@pytest.mark.data_integration
def test_quote_inclusion_clusters_with_host_not_quoted(tmp_path: Path, quote_inclusion_corpus):
    """A host article with an inserted long quote of another must cluster with the host, not the source.

    Precision regression on citation patterns: a doc that quotes a chunk
    of another doc should remain distinct from the source unless the
    quote dominates. Construction at test time keeps the fixture minimal:
    fetch two real articles A and B; build ``A_with_B_quote`` by
    splicing ~1500 chars from the middle of B into the middle of A.

    Expected:
    * cluster(article_a, article_a_with_quote)  — host pair clusters
    * article_b is a singleton — source not over-merged with the quoter
    """
    by_doc_id = {r["doc_id"]: r["text"] for r in quote_inclusion_corpus}
    article_a = by_doc_id["article_a"]
    article_b = by_doc_id["article_b"]

    quote_chars = 1500
    b_mid = len(article_b) // 2
    quote = article_b[b_mid : b_mid + quote_chars]
    a_mid = len(article_a) // 2
    article_a_with_quote = article_a[:a_mid] + "\n\n" + quote + "\n\n" + article_a[a_mid:]

    docs = [
        {"id": "article_a", "text": article_a},
        {"id": "article_b", "text": article_b},
        {"id": "article_a_with_quote", "text": article_a_with_quote},
    ]
    by_source_id = _run_dedup_on_corpus(tmp_path, docs)

    a_cluster = _cluster_id(by_source_id, "article_a")
    a_with_quote_cluster = _cluster_id(by_source_id, "article_a_with_quote")
    b_cluster = _cluster_id(by_source_id, "article_b")

    assert a_cluster is not None, "article_a should not be a singleton"
    assert a_with_quote_cluster is not None, "article_a_with_quote should not be a singleton"
    assert (
        a_cluster == a_with_quote_cluster
    ), f"host A and A_with_quote did not cluster: {a_cluster!r} vs {a_with_quote_cluster!r}"

    assert b_cluster is None, (
        f"article_b clustered (over-merge with quoter): cluster={b_cluster!r}; "
        "the source of a quote should not be merged with the quoter unless the quote dominates"
    )
