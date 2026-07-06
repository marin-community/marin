# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Stage-explorer queries: turn a resolved :class:`StoreLineage` into ducky SQL.

Every method builds SQL over the stage parquet (schemas validated live) and runs
it through :class:`~ducky.client.DuckyClient`, returning
plain JSON-serializable structures the dashboard renders. The dashboard never
sees raw SQL — it names a view + params and this layer composes the query.

Stage parquet layouts / schemas::

    normalize      <path>/outputs/main/*.parquet   {id, text, source_id, ...source cols}
    decontam       <path>/*.parquet                {id, attributes: {contaminated, max_overlap, ...}}
    cluster_assign <path>/*.parquet                {id, cluster_<view>, dist_5000, ...}
    quality        <path>/*.parquet                {id, score}
"""

from __future__ import annotations

import logging

from ducky.client import DuckyClient, QueryResult

from experiments.datakit.web_explorer.lineage import StoreLineage

logger = logging.getLogger(__name__)

# Samplers read a bounded candidate pool (LIMIT, cheap — no full scan of a
# multi-TB source), then seed-shuffle it (`ORDER BY hash(id || seed)`) and take
# n. Reproducible for a given seed, re-samplable by changing it. Since parquet
# is content-hash (id) ordered, the pool is already a representative slice.
DEFAULT_SEED = 7
_POOL_BROAD = 4000  # unfiltered / broad-search sampling
_POOL_FILTER = 500  # selective filters (a cluster, a quality-score range)
_POOL_RARE = 200  # rare filters (contamination)


def _sql_str(value: str) -> str:
    """Escape a string for embedding in single-quoted SQL."""
    return value.replace("'", "''")


class WebExplorer:
    """Query facade bound to one resolved store lineage + a ducky client."""

    def __init__(
        self,
        lineage: StoreLineage,
        ducky: DuckyClient,
        source_docs: dict[str, int] | None = None,
        dedup_attr: dict[str, str] | None = None,
    ):
        self.lineage = lineage
        self.ducky = ducky
        # Estimated docs per source (from the baked summary), used to sample the
        # store from cheap small sources first rather than scanning huge ones.
        self.source_docs = source_docs or {}
        # source -> per-source fuzzy-dup attribute dir ({id, attributes:
        # {dup_cluster_id, is_cluster_canonical}}), for dedup drill-down.
        self.dedup_attr = dedup_attr or {}

    # -- glob helpers -------------------------------------------------------
    def _normalize_glob(self, source: str) -> str:
        return f"{self.lineage.normalize[source]}/outputs/main/*.parquet"

    def _flat_glob(self, mapping: dict[str, str], source: str) -> str:
        return f"{mapping[source]}/*.parquet"

    # -- overview -----------------------------------------------------------
    def resolved_stages(self, source: str) -> dict[str, bool]:
        """Which stages have a resolved dataset for ``source``."""
        return {
            "normalize": source in self.lineage.normalize,
            "tokenize": source in self.lineage.tokenize,
            "decontam": source in self.lineage.decontam,
            "cluster_assign": source in self.lineage.cluster_assign,
            "quality": source in self.lineage.quality,
        }

    # -- normalized ---------------------------------------------------------
    # Char-length aggregates over the whole source would scan all text (multi-TB
    # for big sources), so they sample the first _SAMPLE rows; the exact doc
    # count comes from the baked summary (self.source_docs), no scan.
    _SAMPLE = 200_000

    def normalized_stats(self, source: str) -> dict:
        r = self.ducky.run(
            f"SELECT round(avg(len),1) AS avg_chars, min(len) AS min_chars, max(len) AS max_chars, "
            f"approx_quantile(len, 0.5) AS median_chars "
            f"FROM (SELECT length(text) AS len FROM read_parquet('{self._normalize_glob(source)}') LIMIT {self._SAMPLE})"
        )
        stats = r.dicts()[0]
        stats["docs"] = self.source_docs.get(source)
        stats["sampled_rows"] = self._SAMPLE
        return stats

    def normalized_length_hist(self, source: str, buckets: int = 20) -> QueryResult:
        # log-scaled char-length histogram over a bounded sample (floor bucketing).
        b = int(buckets)
        glob = self._normalize_glob(source)
        return self.ducky.run(
            f"WITH d AS (SELECT length(text) AS n FROM read_parquet('{glob}') LIMIT {self._SAMPLE}), "
            f"m AS (SELECT ln(max(n)+1) AS lg FROM d) "
            f"SELECT least(floor(ln(n+1)/(SELECT lg FROM m)*{b}), {b - 1}) AS bucket, "
            f"min(n) AS lo, max(n) AS hi, count(*) AS docs FROM d GROUP BY bucket ORDER BY bucket"
        )

    def normalized_samples(self, source: str, n: int = 20, search: str = "", seed: int = DEFAULT_SEED) -> QueryResult:
        # Only id + text are guaranteed across sources; other normalize columns
        # (source_id / uuid / source-specific fields) vary, so don't select them.
        where = f"WHERE text ILIKE '%{_sql_str(search)}%'" if search.strip() else ""
        return self.ducky.run(
            f"SELECT id, length(text) AS chars, substr(text, 1, 2000) AS text FROM "
            f"(SELECT id, text FROM read_parquet('{self._normalize_glob(source)}') {where} LIMIT {_POOL_BROAD}) "
            f"ORDER BY hash(id || '{int(seed)}') LIMIT {int(n)}"
        )

    # -- decontamination ----------------------------------------------------
    def decontam_stats(self, source: str) -> dict:
        # Sampled: scanning the contaminated flag over billions of rows is slow.
        # Contamination is rare, so a sampled rate can read 0% — it's a rough
        # gauge, not the exact count (labelled "sampled" in the UI).
        glob = self._flat_glob(self.lineage.decontam, source)
        r = self.ducky.run(
            f"SELECT round(100.0*avg(attributes.contaminated::int), 4) AS contaminated_pct, "
            f"round(avg(attributes.max_overlap), 4) AS avg_overlap, "
            f"round(max(attributes.max_overlap), 4) AS max_overlap "
            f"FROM (SELECT attributes FROM read_parquet('{glob}') LIMIT {self._SAMPLE})"
        )
        stats = r.dicts()[0]
        stats["docs"] = self.source_docs.get(source)
        stats["sampled_rows"] = self._SAMPLE
        return stats

    def decontam_samples(self, source: str, n: int = 20, seed: int = DEFAULT_SEED) -> QueryResult:
        """Seed-sample contaminated docs (from a bounded pool), joined to their text."""
        decon = self._flat_glob(self.lineage.decontam, source)
        norm = self._normalize_glob(source)
        return self.ducky.run(
            f"SELECT id, max_overlap, text FROM ("
            f"SELECT d.id AS id, round(d.attributes.max_overlap, 3) AS max_overlap, substr(n.text, 1, 2000) AS text "
            f"FROM (SELECT id, attributes FROM read_parquet('{decon}') "
            f"WHERE attributes.contaminated LIMIT {_POOL_RARE}) d "
            f"JOIN read_parquet('{norm}') n USING (id)) "
            f"ORDER BY hash(id || '{int(seed)}') LIMIT {int(n)}"
        )

    # -- quality classifier -------------------------------------------------
    def quality_hist(self, source: str, buckets: int = 20) -> QueryResult:
        glob = self._flat_glob(self.lineage.quality, source)
        b = int(buckets)
        return self.ducky.run(
            f"SELECT least(floor(score*{b}), {b - 1}) AS bucket, "
            f"round(min(score),3) AS lo, round(max(score),3) AS hi, count(*) AS docs "
            f"FROM (SELECT score FROM read_parquet('{glob}') LIMIT {self._SAMPLE}) GROUP BY bucket ORDER BY bucket"
        )

    def quality_samples(self, source: str, lo: float, hi: float, n: int = 20, seed: int = DEFAULT_SEED) -> QueryResult:
        qual = self._flat_glob(self.lineage.quality, source)
        norm = self._normalize_glob(source)
        return self.ducky.run(
            f"SELECT score, text FROM ("
            f"SELECT q.id AS id, round(q.score,4) AS score, substr(n.text,1,2000) AS text "
            f"FROM (SELECT id, score FROM read_parquet('{qual}') "
            f"WHERE score >= {float(lo)} AND score < {float(hi)} LIMIT {_POOL_FILTER}) q "
            f"JOIN read_parquet('{norm}') n USING (id)) "
            f"ORDER BY hash(id || '{int(seed)}') LIMIT {int(n)}"
        )

    # -- deduplication ------------------------------------------------------
    def _dedup_glob(self, source: str) -> str:
        return f"{self.dedup_attr[source]}/*.parquet"

    def dedup_examples(
        self, source: str, n_clusters: int = 6, per_cluster: int = 3, seed: int = DEFAULT_SEED
    ) -> list[dict]:
        """Example duplicate clusters for ``source`` with each member's text.

        The per-source fuzzy-dup attr is sparse (non-singletons only). We bound
        cost by grouping within the first ~500k-row window (never the full
        multi-shard set — that would risk an OOM), which surfaces the *dense*
        clusters (templated / near-identical content) that dominate dedup. Then
        we fetch each shown member's normalized text with an id ``IN`` filter.
        """
        if source not in self.dedup_attr or source not in self.lineage.normalize:
            return []
        attr, norm = self._dedup_glob(source), self._normalize_glob(source)
        rows = self.ducky.run(
            f"WITH win AS (SELECT id, attributes.dup_cluster_id d, attributes.is_cluster_canonical c "
            f"FROM read_parquet('{attr}') LIMIT 500000), "
            f"r AS (SELECT *, row_number() OVER (PARTITION BY d ORDER BY c DESC) rn, "
            f"count(*) OVER (PARTITION BY d) n FROM win), "
            f"big AS (SELECT DISTINCT d, n FROM r WHERE n >= 3 "
            f"ORDER BY hash(d || '{int(seed)}') LIMIT {int(n_clusters)}) "
            f"SELECT r.d AS cluster_id, r.n AS sampled_size, r.id AS doc_id, r.c AS canonical "
            f"FROM r JOIN big USING (d) WHERE r.rn <= {int(per_cluster)} ORDER BY r.n DESC, r.d, r.c DESC"
        ).dicts()
        if not rows:
            return []
        ids = [r["doc_id"] for r in rows]
        in_list = ", ".join(f"'{_sql_str(i)}'" for i in ids)
        text = {
            d["id"]: d["text"]
            for d in self.ducky.run(
                f"SELECT id, substr(text, 1, 1500) AS text FROM read_parquet('{norm}') WHERE id IN ({in_list})"
            ).dicts()
        }
        clusters: dict[str, dict] = {}
        for r in rows:
            cl = clusters.setdefault(
                r["cluster_id"], {"cluster_id": r["cluster_id"], "sampled_size": r["sampled_size"], "members": []}
            )
            cl["members"].append(
                {"doc_id": r["doc_id"], "canonical": bool(r["canonical"]), "text": text.get(r["doc_id"], "")}
            )
        return list(clusters.values())

    # -- final store --------------------------------------------------------
    def store_heatmap(self) -> dict:
        """cluster x quality bucket stats, straight from the store artifact (no query)."""
        # Provided by the server from the loaded ClusteredStoreData payload.
        raise NotImplementedError("store_heatmap is served from the store payload, not ducky")

    def store_cluster_samples(
        self, cluster: int, n: int = 12, max_sources: int = 8, seed: int = DEFAULT_SEED
    ) -> list[dict]:
        """Seed-sample docs assigned to ``cluster`` (cluster_view), joined to their text.

        A cluster spans all sources; joining every source's parquet at once is
        too heavy, so we probe resolved sources one at a time (cheap per-source
        join with predicate pushdown on the cluster column), seed-shuffle a
        bounded pool per source, and accumulate up to ``n`` across ``max_sources``.
        """
        view = self.lineage.cluster_view
        both = [s for s in self.lineage.source_names if s in self.lineage.cluster_assign and s in self.lineage.normalize]
        # Smallest sources first: a per-source join scans that source's whole
        # normalize parquet, so cheap sources return samples fastest.
        both.sort(key=lambda s: self.source_docs.get(s, 1 << 62))
        out: list[dict] = []
        for source in both[:max_sources]:
            remaining = n - len(out)
            if remaining <= 0:
                break
            assign = self._flat_glob(self.lineage.cluster_assign, source)
            norm = self._normalize_glob(source)
            res = self.ducky.run(
                f"SELECT cluster, text FROM ("
                f"SELECT a.cluster_{view} AS cluster, a.id AS id, substr(n.text,1,2000) AS text "
                f"FROM read_parquet('{assign}') a JOIN read_parquet('{norm}') n USING (id) "
                f"WHERE a.cluster_{view} = {int(cluster)} LIMIT {_POOL_FILTER}) "
                f"ORDER BY hash(id || '{int(seed)}') LIMIT {int(remaining)}"
            )
            for row in res.dicts():
                out.append({**row, "source": source})
        return out

    def store_bucket_samples(
        self, cluster: int, quality_bucket: int, n: int = 12, max_sources: int = 8, seed: int = DEFAULT_SEED
    ) -> list[dict]:
        """Sample docs in a specific (cluster, quality) bucket, with score + text.

        The bucket is ``cluster_<view> == cluster`` AND the quality score in that
        bucket's threshold range, so this needs cluster_assign ⋈ quality ⋈
        normalize per source. (Assignment view — it does not re-apply the
        decontam/dedup drops the store did, so counts are looser than the bucket
        totals.)
        """
        th = self.lineage.quality_thresholds
        q = int(quality_bucket)
        lo = 0.0 if q == 0 else th[q - 1]
        hi = 1.01 if q >= len(th) else th[q]
        view = self.lineage.cluster_view
        srcs = [
            s
            for s in self.lineage.source_names
            if s in self.lineage.cluster_assign and s in self.lineage.quality and s in self.lineage.normalize
        ]
        srcs.sort(key=lambda s: self.source_docs.get(s, 1 << 62))
        out: list[dict] = []
        for source in srcs[:max_sources]:
            remaining = n - len(out)
            if remaining <= 0:
                break
            assign = self._flat_glob(self.lineage.cluster_assign, source)
            qual = self._flat_glob(self.lineage.quality, source)
            norm = self._normalize_glob(source)
            res = self.ducky.run(
                f"SELECT score, text FROM ("
                f"SELECT a.id AS id, round(q.score, 4) AS score, substr(n.text, 1, 2000) AS text "
                f"FROM read_parquet('{assign}') a JOIN read_parquet('{qual}') q USING (id) "
                f"JOIN read_parquet('{norm}') n USING (id) "
                f"WHERE a.cluster_{view} = {int(cluster)} AND q.score >= {lo} AND q.score < {hi} LIMIT {_POOL_FILTER}) "
                f"ORDER BY hash(id || '{int(seed)}') LIMIT {int(remaining)}"
            )
            for row in res.dicts():
                out.append({**row, "source": source})
        return out
