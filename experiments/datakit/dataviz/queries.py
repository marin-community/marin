# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Stage-explorer queries: turn a resolved :class:`StoreLineage` into ducky SQL.

Every method builds SQL over the stage parquet (schemas validated live) and runs
it through :class:`~experiments.datakit.dataviz.ducky.DuckyClient`, returning
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

from experiments.datakit.dataviz.ducky import DuckyClient, QueryResult
from experiments.datakit.dataviz.lineage import StoreLineage

logger = logging.getLogger(__name__)


def _sql_str(value: str) -> str:
    """Escape a string for embedding in single-quoted SQL."""
    return value.replace("'", "''")


class Dataviz:
    """Query facade bound to one resolved store lineage + a ducky client."""

    def __init__(self, lineage: StoreLineage, ducky: DuckyClient, source_docs: dict[str, int] | None = None):
        self.lineage = lineage
        self.ducky = ducky
        # Estimated docs per source (from the baked summary), used to sample the
        # store from cheap small sources first rather than scanning huge ones.
        self.source_docs = source_docs or {}

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
    def normalized_stats(self, source: str) -> dict:
        r = self.ducky.run(
            f"SELECT count(*) AS docs, round(avg(length(text)),1) AS avg_chars, "
            f"min(length(text)) AS min_chars, max(length(text)) AS max_chars, "
            f"approx_quantile(length(text), 0.5) AS median_chars "
            f"FROM read_parquet('{self._normalize_glob(source)}')"
        )
        return r.dicts()[0]

    def normalized_length_hist(self, source: str, buckets: int = 20) -> QueryResult:
        # log-scaled char-length histogram (floor bucketing; +1 guards len 0).
        b = int(buckets)
        return self.ducky.run(
            f"WITH d AS (SELECT length(text) AS n FROM read_parquet('{self._normalize_glob(source)}')), "
            f"m AS (SELECT ln(max(n)+1) AS lg FROM d) "
            f"SELECT least(floor(ln(n+1)/(SELECT lg FROM m)*{b}), {b - 1}) AS bucket, "
            f"min(n) AS lo, max(n) AS hi, count(*) AS docs FROM d GROUP BY bucket ORDER BY bucket"
        )

    def normalized_samples(self, source: str, n: int = 20, search: str = "") -> QueryResult:
        where = f"WHERE text ILIKE '%{_sql_str(search)}%'" if search.strip() else ""
        return self.ducky.run(
            f"SELECT id, source_id, length(text) AS chars, substr(text, 1, 2000) AS text "
            f"FROM read_parquet('{self._normalize_glob(source)}') {where} "
            f"USING SAMPLE {int(n)} ROWS"
        )

    # -- decontamination ----------------------------------------------------
    def decontam_stats(self, source: str) -> dict:
        glob = self._flat_glob(self.lineage.decontam, source)
        r = self.ducky.run(
            f"SELECT count(*) AS docs, sum(attributes.contaminated::int) AS contaminated, "
            f"round(100.0*avg(attributes.contaminated::int), 4) AS contaminated_pct, "
            f"round(avg(attributes.max_overlap), 4) AS avg_overlap, "
            f"round(max(attributes.max_overlap), 4) AS max_overlap "
            f"FROM read_parquet('{glob}')"
        )
        return r.dicts()[0]

    def decontam_samples(self, source: str, n: int = 20) -> QueryResult:
        """Sample contaminated docs, joined back to their normalized text."""
        decon = self._flat_glob(self.lineage.decontam, source)
        norm = self._normalize_glob(source)
        return self.ducky.run(
            f"SELECT d.id, round(d.attributes.max_overlap, 3) AS max_overlap, "
            f"substr(n.text, 1, 2000) AS text "
            f"FROM read_parquet('{decon}') d JOIN read_parquet('{norm}') n USING (id) "
            f"WHERE d.attributes.contaminated ORDER BY d.attributes.max_overlap DESC LIMIT {int(n)}"
        )

    # -- quality classifier -------------------------------------------------
    def quality_hist(self, source: str, buckets: int = 20) -> QueryResult:
        glob = self._flat_glob(self.lineage.quality, source)
        b = int(buckets)
        return self.ducky.run(
            f"SELECT least(floor(score*{b}), {b - 1}) AS bucket, "
            f"round(min(score),3) AS lo, round(max(score),3) AS hi, count(*) AS docs "
            f"FROM read_parquet('{glob}') GROUP BY bucket ORDER BY bucket"
        )

    def quality_samples(self, source: str, lo: float, hi: float, n: int = 20) -> QueryResult:
        qual = self._flat_glob(self.lineage.quality, source)
        norm = self._normalize_glob(source)
        return self.ducky.run(
            f"SELECT round(q.score,4) AS score, substr(n.text,1,2000) AS text "
            f"FROM read_parquet('{qual}') q JOIN read_parquet('{norm}') n USING (id) "
            f"WHERE q.score >= {float(lo)} AND q.score < {float(hi)} "
            f"ORDER BY q.score DESC LIMIT {int(n)}"
        )

    # -- final store --------------------------------------------------------
    def store_heatmap(self) -> dict:
        """cluster x quality bucket stats, straight from the store artifact (no query)."""
        # Provided by the server from the loaded ClusteredStoreData payload.
        raise NotImplementedError("store_heatmap is served from the store payload, not ducky")

    def store_cluster_samples(self, cluster: int, n: int = 12, max_sources: int = 8) -> list[dict]:
        """Sample docs assigned to ``cluster`` (cluster_view), joined to their text.

        A cluster spans all sources; joining every source's parquet at once is
        too heavy, so we probe resolved sources one at a time (cheap per-source
        join with predicate pushdown on the cluster column) and accumulate up to
        ``n`` rows across at most ``max_sources`` sources.
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
                f"SELECT a.cluster_{view} AS cluster, substr(n.text,1,2000) AS text "
                f"FROM read_parquet('{assign}') a JOIN read_parquet('{norm}') n USING (id) "
                f"WHERE a.cluster_{view} = {int(cluster)} LIMIT {int(remaining)}"
            )
            for row in res.dicts():
                out.append({**row, "source": source})
        return out
