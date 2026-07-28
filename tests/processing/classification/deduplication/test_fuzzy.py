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
from fray.current_client import set_current_client
from fray.local_backend import LocalClient
from marin.datakit.normalize import NormalizedData, generate_id, normalize_to_parquet
from marin.processing.classification.deduplication.fuzzy_dups import compute_fuzzy_dups_attrs
from marin.processing.classification.deduplication.fuzzy_minhash import (
    MinHashAttrData,
    MinHashParams,
    NgramKind,
    compute_minhash_attrs,
)
from marin.processing.classification.deduplication.fuzzy_verification import FuzzyVerificationParams
from zephyr.writers import write_jsonl_file, write_parquet_file

TEST_MINHASH_PARAMS = MinHashParams(
    num_perms=286,
    num_bands=26,
    ngram_size=5,
    ngram_kind=NgramKind.WORD,
    seed=42,
)
CHAR_MINHASH_PARAMS = TEST_MINHASH_PARAMS.model_copy(update={"ngram_kind": NgramKind.CHAR})


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
    """Return every Parquet row under *attr_dir*."""
    rows: list[dict] = []
    for pf in sorted(Path(attr_dir).glob("*.parquet")):
        rows.extend(pq.read_table(str(pf)).to_pylist())
    return rows


def _write_minhash_attr_dataset(
    *,
    output_dir: str,
    source_main_dir: str,
    rows: list[dict],
    text_by_id: dict[str, str],
) -> MinHashAttrData:
    """Write a one-shard MinHash attr dataset for focused fuzzy-dup tests."""
    attr_dir = os.path.join(output_dir, "outputs")
    Path(attr_dir).mkdir(parents=True, exist_ok=True)
    basename = "part-00000.parquet"
    write_parquet_file(rows, os.path.join(attr_dir, basename))
    Path(source_main_dir).mkdir(parents=True, exist_ok=True)
    write_parquet_file(
        [{"id": row["id"], "text": text_by_id[row["id"]]} for row in rows],
        os.path.join(source_main_dir, basename),
    )
    return MinHashAttrData(
        params=TEST_MINHASH_PARAMS,
        source_main_dir=source_main_dir,
        attr_dir=attr_dir,
        counters={},
    )


def _contained_candidate_pair() -> tuple[str, str, str, str]:
    """Return member/canonical texts and IDs with the canonical ordered first by CC."""
    member = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda"
    member_id = generate_id(member)
    member_order = dupekit.hash_xxh3_128(f"source_000|{member_id}".encode())
    for suffix in range(1_000):
        canonical = f"preface context {member} appendix context {suffix}"
        canonical_id = generate_id(canonical)
        if dupekit.hash_xxh3_128(f"source_000|{canonical_id}".encode()) < member_order:
            return member, member_id, canonical, canonical_id
    raise AssertionError("Could not construct a longer deterministic canonical")


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
    """Only the verified member receives a removal marker.

    Builds a direct candidate whose shorter member is an exact token-3 subset
    of the deterministic canonical and verifies:
    * exactly one member gets ``dup_doc=True``,
    * the marker names the retained representative,
    * unique docs have no attr row.
    """
    output_dir = fox_corpus["output_dir"]
    main_dir = os.path.join(output_dir, "single_source_main")
    member, member_id, canonical, canonical_id = _contained_candidate_pair()

    minhash = _write_minhash_attr_dataset(
        output_dir=os.path.join(output_dir, "single_source_minhash"),
        source_main_dir=main_dir,
        rows=[
            {"id": member_id, "buckets": ["shared"]},
            {"id": canonical_id, "buckets": ["shared"]},
        ],
        text_by_id={member_id: member, canonical_id: canonical},
    )
    dups = compute_fuzzy_dups_attrs(
        inputs=[minhash],
        output_path=os.path.join(output_dir, "single_source_dups"),
        max_parallelism=1,
    )

    rows = _read_cluster_attrs(dups.sources[main_dir].attr_dir)
    assert len(rows) == 1
    assert rows[0]["id"] == member_id
    attributes = rows[0]["attributes"]
    assert attributes["dup_doc"] is True
    assert attributes["dup_representative_id"] == canonical_id


def test_fuzzy_dups_multi_source_per_source_attr_trees(fox_corpus):
    """Two MinHashAttrData inputs produce two per-source attr trees.

    Cross-source exact-text duplicates have the same normalized ID in both
    datasets but remain distinct CC nodes. Exactly one side receives the
    verified removal marker.

    This test targets multi-source fuzzy dedup behavior directly. Normalization
    and MinHash generation already have separate coverage above.
    """
    train_main_dir = os.path.join(fox_corpus["output_dir"], "train_main")
    test_main_dir = os.path.join(fox_corpus["output_dir"], "test_main")
    arctic = "Arctic predators have superior auditory capabilities for hunting beneath snow."
    red = "Red canids inhabit northern territories worldwide."
    train_unique = "Newborn kits emerge sightless and vulnerable."
    test_unique = "Rapid runners represent the most diminutive wild dogs."
    train_mh = _write_minhash_attr_dataset(
        output_dir=os.path.join(fox_corpus["output_dir"], "mh_train"),
        source_main_dir=train_main_dir,
        rows=[
            {
                "id": generate_id(arctic),
                "buckets": ["shared-arctic"],
            },
            {
                "id": generate_id(red),
                "buckets": ["shared-red"],
            },
            {
                "id": generate_id(train_unique),
                "buckets": ["train-unique"],
            },
        ],
        text_by_id={generate_id(text): text for text in (arctic, red, train_unique)},
    )
    test_mh = _write_minhash_attr_dataset(
        output_dir=os.path.join(fox_corpus["output_dir"], "mh_test"),
        source_main_dir=test_main_dir,
        rows=[
            {
                "id": generate_id(arctic),
                "buckets": ["shared-arctic"],
            },
            {
                "id": generate_id(red),
                "buckets": ["shared-red"],
            },
            {
                "id": generate_id(test_unique),
                "buckets": ["test-unique"],
            },
        ],
        text_by_id={generate_id(text): text for text in (arctic, red, test_unique)},
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

    for shared_text in (
        "Arctic predators have superior auditory capabilities for hunting beneath snow.",
        "Red canids inhabit northern territories worldwide.",
    ):
        content_id = generate_id(shared_text)
        marked = [rows[content_id] for rows in (train_rows, test_rows) if content_id in rows]
        assert len(marked) == 1, f"{shared_text!r}: exactly one verified removal marker expected"
        assert marked[0]["attributes"]["dup_doc"] is True


def test_fuzzy_dups_retains_candidate_rejected_by_exact_verifier(fox_corpus):
    main_dir = os.path.join(fox_corpus["output_dir"], "rejected_main")
    left = "astronomy describes distant stars and planetary systems"
    right = "cooking combines seasonal produce with regional techniques"
    minhash = _write_minhash_attr_dataset(
        output_dir=os.path.join(fox_corpus["output_dir"], "rejected_minhash"),
        source_main_dir=main_dir,
        rows=[
            {"id": generate_id(left), "buckets": ["forced-collision"]},
            {"id": generate_id(right), "buckets": ["forced-collision"]},
        ],
        text_by_id={generate_id(left): left, generate_id(right): right},
    )

    dups = compute_fuzzy_dups_attrs(
        inputs=[minhash],
        output_path=os.path.join(fox_corpus["output_dir"], "rejected_dups"),
        max_parallelism=1,
    )

    assert _read_cluster_attrs(dups.sources[main_dir].attr_dir) == []
    decisions = _read_cluster_attrs(dups.decisions_dir)
    assert len(decisions) == 1
    assert decisions[0]["accepted"] is False
    assert decisions[0]["rejection"] in {
        "member_longer",
        "containment_below_threshold",
    }


def test_fuzzy_dups_writes_typed_empty_markers_without_candidates(fox_corpus):
    main_dir = os.path.join(fox_corpus["output_dir"], "singleton_main")
    text = "one document with no shared minhash bucket"
    minhash = _write_minhash_attr_dataset(
        output_dir=os.path.join(fox_corpus["output_dir"], "singleton_minhash"),
        source_main_dir=main_dir,
        rows=[{"id": generate_id(text), "buckets": ["unique"]}],
        text_by_id={generate_id(text): text},
    )

    dups = compute_fuzzy_dups_attrs(
        inputs=[minhash],
        output_path=os.path.join(fox_corpus["output_dir"], "singleton_dups"),
        max_parallelism=1,
    )

    attr_files = list(Path(dups.sources[main_dir].attr_dir).glob("*.parquet"))
    assert len(attr_files) == 1
    table = pq.read_table(attr_files[0])
    assert table.num_rows == 0
    assert table.schema.field("attributes").type.field("dup_doc").type == pa.bool_()
    assert dups.counters.get("dedup/fuzzy/verification/candidates", 0) == 0


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


def _canonical_assignment(minhash: MinHashAttrData, output_path: str) -> dict[str, tuple[str, bool]]:
    """Run fuzzy-dups and return ``{id -> (dup_cluster_id, is_canonical)}``."""
    dups = compute_fuzzy_dups_attrs(inputs=[minhash], output_path=os.path.join(output_path, "dups"), max_parallelism=4)
    assignments = {}
    for decision in _read_cluster_attrs(dups.decisions_dir):
        if not decision["accepted"]:
            continue
        assignment = (decision["component_id"], True)
        previous = assignments.setdefault(decision["canonical_id"], assignment)
        assert previous == assignment
        assignments[decision["member_id"]] = (decision["component_id"], False)
    return assignments


def test_fuzzy_dups_canonical_selection_is_deterministic(fox_corpus):
    """Two independent runs over the same input select the same survivors (marin#6798).

    Canonical selection is the min content-hash per component, so it must not
    depend on shard/link/reduce order or parallelism. Running fuzzy-dups twice
    over the same prepared MinHash input into separate output trees must yield a
    byte-identical ``{id -> (dup_cluster_id, is_canonical)}`` map, with
    exactly one canonical per cluster. A future canonical pick that leaked
    dict/set ordering or arrival order would break this.
    """
    output_dir = fox_corpus["output_dir"]
    main_dir = os.path.join(output_dir, "deterministic_main")
    member, member_id, canonical, canonical_id = _contained_candidate_pair()
    minhash = _write_minhash_attr_dataset(
        output_dir=os.path.join(output_dir, "deterministic_minhash"),
        source_main_dir=main_dir,
        rows=[
            {"id": member_id, "buckets": ["shared"]},
            {"id": canonical_id, "buckets": ["shared"]},
        ],
        text_by_id={member_id: member, canonical_id: canonical},
    )

    first = _canonical_assignment(minhash, os.path.join(output_dir, "run_a"))
    second = _canonical_assignment(minhash, os.path.join(output_dir, "run_b"))

    assert first == second, "fuzzy-dup canonical assignment differs between two runs over identical input"
    # Guard against a vacuous all-empty comparison and pin the
    # one-canonical-per-cluster rule.
    assert first, "expected at least one cluster member row"
    canonical_per_cluster: dict[str, int] = {}
    for cluster_id, is_canonical in first.values():
        canonical_per_cluster[cluster_id] = canonical_per_cluster.get(cluster_id, 0) + int(is_canonical)
    assert all(
        n == 1 for n in canonical_per_cluster.values()
    ), f"every cluster must have exactly one canonical; got {canonical_per_cluster}"


def test_fuzzy_dups_capped_does_not_raise_and_emits(fox_corpus):
    """A capped (non-converged) run warns but still produces a deterministic result (marin#6798).

    Builds a 5-node path graph (A-B-C-D-E, neighbors sharing one LSH bucket
    each) whose min component id needs >=2 iterations to reach both ends;
    ``cc_max_iterations=1`` cannot converge. The step must NOT raise -- with the
    id_norm-sorted bucket topology the capped result is deterministic (just
    incomplete) -- and must still emit verified marker rows under permissive
    verification thresholds.
    """
    main_dir = os.path.join(fox_corpus["output_dir"], "path_main")
    texts = [f"path node number {i} with distinct filler content here" for i in range(5)]
    # Chain the nodes: node i shares bucket b{i-1}{i} with its left neighbor and
    # b{i}{i+1} with its right neighbor, so links form a path, not a star.
    rows = []
    for i, text in enumerate(texts):
        buckets = []
        if i > 0:
            buckets.append(f"b{i - 1}{i}")
        if i < len(texts) - 1:
            buckets.append(f"b{i}{i + 1}")
        rows.append({"id": generate_id(text), "buckets": buckets})

    mh = _write_minhash_attr_dataset(
        output_dir=os.path.join(fox_corpus["output_dir"], "mh_path"),
        source_main_dir=main_dir,
        rows=rows,
        text_by_id={generate_id(text): text for text in texts},
    )

    dups = compute_fuzzy_dups_attrs(
        inputs=[mh],
        output_path=os.path.join(fox_corpus["output_dir"], "fuzzy_dups_path"),
        verification_params=FuzzyVerificationParams(
            minimum_member_containment=0,
            maximum_member_unique_ngrams=100,
            maximum_chars_per_token=100,
        ),
        cc_max_iterations=1,
        max_parallelism=4,
    )
    attr_rows = _read_cluster_attrs(dups.sources[main_dir].attr_dir)
    assert attr_rows, "capped run should still emit verified marker rows"


def test_text_cap_chars_truncates_mega_docs_only(tmp_path):
    """``text_cap_chars`` should change the MinHash signature for docs above
    the cap but leave smaller docs unaffected.

    Mega documents (e.g. PDF text dumps with O(10M) shingles) otherwise produce
    saturated MinHash signatures that band-collide with arbitrary other docs,
    forming CC false-positive blobs. The cap bounds signature density per doc.
    """
    # Build a tiny normalized dataset with:
    #   - one MEGA doc (text > cap → cap-buckets must differ from no-cap buckets)
    #   - one SMALL doc (text < cap → cap-buckets must equal no-cap buckets)
    cap_chars = 2_000  # small cap so we can easily exceed it without huge text
    rng = random.Random(7)
    alphabet = string.ascii_lowercase + " "
    mega_text = "".join(rng.choice(alphabet) for _ in range(cap_chars * 3))
    small_text = "".join(rng.choice(alphabet) for _ in range(cap_chars // 4))

    src_dir = tmp_path / "src"
    src_dir.mkdir()
    write_jsonl_file(
        [
            {"id": "mega", "text": mega_text, "source": "t"},
            {"id": "small", "text": small_text, "source": "t"},
        ],
        str(src_dir / "shard.jsonl.gz"),
    )
    norm = normalize_to_parquet(input_path=str(src_dir), output_path=str(tmp_path / "norm"))

    cap_mh = compute_minhash_attrs(
        source=norm,
        output_path=str(tmp_path / "mh_cap"),
        text_cap_chars=cap_chars,
    )
    nocap_mh = compute_minhash_attrs(
        source=norm,
        output_path=str(tmp_path / "mh_nocap"),
        text_cap_chars=None,
    )

    cap_by_id = {r["id"]: r["buckets"] for r in _read_cluster_attrs(cap_mh.attr_dir)}
    nocap_by_id = {r["id"]: r["buckets"] for r in _read_cluster_attrs(nocap_mh.attr_dir)}

    # normalize_to_parquet generates a deterministic xxh3 id from text; the
    # raw "mega"/"small" handles are renamed to source_id internally and
    # don't appear in minhash attrs.
    mega_id = generate_id(mega_text)
    small_id = generate_id(small_text)
    assert set(cap_by_id) == {mega_id, small_id} == set(nocap_by_id)

    # Small doc (under cap) → unaffected.
    assert sorted(cap_by_id[small_id]) == sorted(
        nocap_by_id[small_id]
    ), "doc smaller than the cap should produce identical buckets"
    # Mega doc (over cap) → different signature, no band overlap.
    cap_mega = set(cap_by_id[mega_id])
    nocap_mega = set(nocap_by_id[mega_id])
    assert cap_mega != nocap_mega, "doc over the cap should produce a different signature"
    assert not (cap_mega & nocap_mega), (
        "doc over the cap should share no LSH bands between cap / no-cap signatures "
        f"(intersection: {cap_mega & nocap_mega})"
    )

    # Params + version metadata.
    assert cap_mh.params.text_cap_chars == cap_chars
    assert nocap_mh.params.text_cap_chars is None
    assert cap_mh.version == "v3"


# ---------------------------------------------------------------------------
# Char-5-gram Jaccard recall / precision tests.
#
# The dupekit MinHash pipeline shingles by character (lib/dupekit/rust/src/
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
            ngram_kind={
                NgramKind.CHAR: dupekit.NgramKind.Char,
                NgramKind.WORD: dupekit.NgramKind.Word,
            }[params.ngram_kind],
            seed=params.seed,
        ),
        dupekit.Transformation.MinHashLSH(input_col="signature", output_col="buckets", num_bands=params.num_bands),
        dupekit.Transformation.SelectColumns(columns=["id", "buckets"]),
    ]


def _shared_lsh_bucket(text_a: str, text_b: str) -> bool:
    """Return True iff (a, b) share at least one MinHash-LSH bucket."""
    batch = pa.RecordBatch.from_pylist([{"id": "a", "text": text_a}, {"id": "b", "text": text_b}])
    out = dupekit.transform(batch, _dupekit_pipeline(CHAR_MINHASH_PARAMS))
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
    """Return accepted cluster assignments from normalize -> minhash -> fuzzy-dups."""
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    write_jsonl_file(docs, str(src_dir / "shard_0.jsonl.gz"))

    source = _normalize(str(src_dir), str(tmp_path / "norm"))
    minhash = compute_minhash_attrs(source=source, output_path=str(tmp_path / "minhash"))
    dups = compute_fuzzy_dups_attrs(inputs=[minhash], output_path=str(tmp_path / "dups"), max_parallelism=1)

    by_id = _read_main_records(source)
    assignments: dict[str, dict] = {}
    for decision in _read_cluster_attrs(dups.decisions_dir):
        if not decision["accepted"]:
            continue
        attributes = {"dup_cluster_id": decision["component_id"]}
        for document_id in (decision["canonical_id"], decision["member_id"]):
            if document_id in by_id:
                assignments[by_id[document_id]["source_id"]] = {"attributes": attributes}
    return assignments


def _cluster_id(by_source_id: dict[str, dict], source_id: str) -> str | None:
    """Return the accepted dup cluster for *source_id*, or None if it was retained."""
    row = by_source_id.get(source_id)
    return row["attributes"]["dup_cluster_id"] if row else None


@pytest.mark.data_integration
def test_html_parser_variants_do_not_cross_cluster_articles(
    tmp_path: Path, parser_variants_docs, parser_variants_articles
):
    """Accepted parser-variant pairs never cross article boundaries.

    The exact verifier intentionally retains same-article parser variants when
    each extraction carries unique token 3-grams. Any pair it does accept must
    still consist only of renderings of the same article.
    """
    by_source_id = _run_dedup_on_corpus(tmp_path, parser_variants_docs)
    assert parser_variants_articles, "no parser-variant fixtures discovered"

    articles_by_cluster: dict[str, set[str]] = {}
    for source_id, row in by_source_id.items():
        article, _ = source_id.rsplit("__", 1)
        cluster_id = row["attributes"]["dup_cluster_id"]
        articles_by_cluster.setdefault(cluster_id, set()).add(article)
    assert all(len(articles) == 1 for articles in articles_by_cluster.values()), articles_by_cluster


@pytest.mark.data_integration
@pytest.mark.xfail(
    strict=True,
    reason=(
        "Parser outputs preserve different navigation, citation, and boundary text. "
        "The precision-first exact verifier retains variants with any unique token "
        "3-grams; a separately evaluated boilerplate policy is needed to merge all three."
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
def test_wikipedia_revisions_do_not_cross_cluster_articles(
    tmp_path: Path, wikipedia_revisions_docs, wikipedia_revisions_articles
):
    """Accepted temporal-revision pairs never cross article boundaries.

    Revisions with refreshed text are deliberately retained by the
    precision-first verifier. This regression ensures the verifier never
    turns shared Wikipedia structure into a cross-article deletion.
    """
    by_source_id = _run_dedup_on_corpus(tmp_path, wikipedia_revisions_docs)
    assert wikipedia_revisions_articles, "no revision fixtures discovered"

    articles_by_cluster: dict[str, set[str]] = {}
    for source_id, row in by_source_id.items():
        article, _ = source_id.rsplit("__", 1)
        cluster_id = row["attributes"]["dup_cluster_id"]
        articles_by_cluster.setdefault(cluster_id, set()).add(article)
    assert all(len(articles) == 1 for articles in articles_by_cluster.values()), articles_by_cluster


@pytest.mark.data_integration
def test_quote_inclusion_does_not_remove_quoted_source(tmp_path: Path, quote_inclusion_corpus):
    """A host article containing a long quote must not remove the quoted source.

    Precision regression on citation patterns: a doc that quotes a chunk
    of another doc remains distinct from the source. The exact-subset rule may
    also retain the host variant because inserting the quote creates boundary
    token 3-grams; that conservative recall tradeoff is intentional.
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

    b_cluster = _cluster_id(by_source_id, "article_b")

    assert b_cluster is None, (
        f"article_b clustered (over-merge with quoter): cluster={b_cluster!r}; "
        "the source of a quote should not be merged with the quoter unless the quote dominates"
    )
