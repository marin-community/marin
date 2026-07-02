# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pre-baked catalog: named DuckDB views over common Marin data sources, plus
click-to-fill example queries for the dashboard.

Two data sources are wired in, each rooted at a configurable object-store prefix
(``DuckyConfig.finelog_root`` / ``datakit_root``):

- **finelog** — the log/stats store. One namespace per table, laid out flat as
  ``<root>/<namespace>/seg_L*.parquet`` (LSM segments; ``L<level>`` is a compaction
  level, not a date). Namespace directory names literally contain dots
  (``iris.task``), so the views live in a ``finelog`` schema with dotted, quoted
  names: ``finelog."iris.task"``.

- **datakit** — the normalized parquet datasets, laid out as
  ``<root>/<name>_<hash8>/outputs/main/*.parquet`` where ``<hash8>`` is an
  unguessable recipe hash, so each view globs ``<name>_*``. Only a curated,
  high-value subset is pre-baked as views; the rest are reachable via the browse
  example query.

Views are created eagerly (DuckDB binds a view's schema at ``CREATE`` time), so a
view over an absent or unreachable dataset fails to create — the runner treats that
as best-effort and skips it. The example queries are plain text and always
available regardless of which views materialized.
"""

from __future__ import annotations

import dataclasses

from ducky.config import DuckyConfig

FINELOG_SCHEMA = "finelog"
DATAKIT_SCHEMA = "datakit"

# finelog namespaces (table dir name → description). The base `log` table is defined by
# finelog; the `iris.*`/`zephyr.*` stats tables are defined by their producers. See
# lib/finelog/src/finelog/schema.py and lib/{iris,zephyr}/.../stats.py.
_FINELOG_NAMESPACES: tuple[tuple[str, str], ...] = (
    ("log", "Base log lines: key, source, data (trigram-indexed), epoch_ms, level, seq."),
    ("iris.task", "Per-attempt task resource usage (cpu/memory/disk) reported by workers."),
    ("iris.worker", "Per-worker heartbeat: host utilization, identity, device type."),
    ("iris.task_status", "Markdown status text pushed from running tasks (short retention)."),
    ("iris.profile", "CPU/memory/thread profile captures, one row per capture."),
    ("iris.provisioning", "Slice provisioning outcomes (ready/stockout/error/preempted)."),
    ("zephyr.stage", "Per-stage completion stats: throughput + aggregated resource usage."),
    ("zephyr.worker", "Per-shard stats emitted at start / sample interval / end."),
)

# Curated datakit normalized datasets (view name, step-name path segment, description).
# The step-name path segment is everything between the root and the `_<hash8>` suffix; it
# may contain a slash (a nested family/subset). Chosen for size / general usefulness — the
# full set (~100+) lives in lib/marin/src/marin/datakit/sources.py; browse it via the
# example query rather than baking every one as an eagerly-bound view. Which of these are
# present depends on what's replicated to `datakit_root`; the nemotron_cc_math_v1 subsets are
# in us-east5 today, the rest are canonical in eu-west4. Absent ones are skipped at startup.
_DATAKIT_SOURCES: tuple[tuple[str, str, str], ...] = (
    ("nemotron_cc_math_v1_4plus_mind", "nemotron_cc_math_v1/4plus_mind", "Nemotron-CC math, 4+ MIND high-quality."),
    ("nemotron_cc_math_v1_4plus", "nemotron_cc_math_v1/4plus", "Nemotron-CC math, quality 4+ subset."),
    ("finetranslations", "finetranslations", "Normalized translation corpus (~3.0T tokens)."),
    ("nemotron_cc_v2_high_quality", "nemotron_cc_v2/high_quality", "Nemotron-CC v2 high-quality subset."),
    ("finepdfs", "finepdfs", "FinePDFs normalized text (all languages)."),
    ("institutional_books", "institutional_books", "Institutional Books normalized text."),
    ("cp_peS2o", "cp/peS2o", "Common Pile peS2o (academic papers)."),
)


@dataclasses.dataclass(frozen=True)
class View:
    """A pre-baked DuckDB view over an object-store parquet dataset."""

    schema: str
    name: str
    description: str
    definition_sql: str
    """The ``read_parquet(...)`` relation the view selects from."""

    @property
    def qualified_name(self) -> str:
        """Fully-qualified, safely-quoted identifier, e.g. ``finelog."iris.task"``."""
        return f"{self.schema}.{_quote_ident(self.name)}"


@dataclasses.dataclass(frozen=True)
class ExampleQuery:
    """A ready-to-run SQL snippet the dashboard offers as a click-to-fill starter."""

    title: str
    description: str
    sql: str


@dataclasses.dataclass(frozen=True)
class Catalog:
    views: tuple[View, ...]
    examples: tuple[ExampleQuery, ...]


def _quote_ident(name: str) -> str:
    """Quote a SQL identifier, escaping embedded double quotes."""
    escaped = name.replace('"', '""')
    return f'"{escaped}"'


def _finelog_views(root: str) -> list[View]:
    root = root.rstrip("/")
    return [
        View(
            schema=FINELOG_SCHEMA,
            name=namespace,
            description=description,
            definition_sql=f"SELECT * FROM read_parquet('{root}/{namespace}/seg_L*.parquet')",
        )
        for namespace, description in _FINELOG_NAMESPACES
    ]


def _datakit_views(root: str) -> list[View]:
    root = root.rstrip("/")
    return [
        View(
            schema=DATAKIT_SCHEMA,
            name=name,
            description=f"{description} (normalized/{path})",
            definition_sql=f"SELECT * FROM read_parquet('{root}/{path}_*/outputs/main/*.parquet')",
        )
        for name, path, description in _DATAKIT_SOURCES
    ]


def _finelog_examples(views: list[View]) -> list[ExampleQuery]:
    by_name = {view.name: view.qualified_name for view in views}
    return [
        ExampleQuery(
            "Log level breakdown",
            "Row count per log level — reads only the tiny `level` column, so it stays cheap "
            "even though the `log` table is tens of GB (the `data` blob is ~80% of the bytes).",
            f"SELECT level, count(*) AS rows\nFROM {by_name['log']}\nGROUP BY level\nORDER BY level",
        ),
        ExampleQuery(
            "Recent errors",
            "Latest error/critical log lines (finelog level: DEBUG=1, INFO=2, WARNING=3, ERROR=4, "
            "CRITICAL=5). Filters on `level` and truncates `data` with left(), so only the ~100 "
            "matching rows materialize the text blob.",
            f"SELECT epoch_ms, source, key, left(data, 500) AS data\nFROM {by_name['log']}\n"
            f"WHERE level >= 4\nORDER BY seq DESC\nLIMIT 100",
        ),
        ExampleQuery(
            "Iris task resource usage",
            "Most recent per-task cpu/memory samples. Projects the handful of numeric columns "
            "it needs rather than SELECT *.",
            f"SELECT task_id, worker_id, ts, cpu_millicores, memory_mb, memory_peak_mb\n"
            f"FROM {by_name['iris.task']}\nORDER BY ts DESC\nLIMIT 100",
        ),
        ExampleQuery(
            "Worker utilization snapshot",
            "Latest heartbeats with cpu/memory and running task count.",
            f"SELECT worker_id, ts, status, cpu_pct, mem_bytes, running_task_count, device_variant\n"
            f"FROM {by_name['iris.worker']}\nORDER BY ts DESC\nLIMIT 100",
        ),
        ExampleQuery(
            "Zephyr stage throughput",
            "Completed zephyr stages ranked by byte rate.",
            f"SELECT execution_id, stage_name, status, elapsed, items, item_rate, byte_rate\n"
            f"FROM {by_name['zephyr.stage']}\nORDER BY ts DESC\nLIMIT 100",
        ),
    ]


def _datakit_examples(root: str) -> list[ExampleQuery]:
    root = root.rstrip("/")
    # A read_parquet template rather than a fixed view: which normalized datasets are present
    # varies by region/replication, so the sample/count examples take an editable <dataset>
    # placeholder the user fills from the browse query. Cheap by construction (glob metadata,
    # footer-only count, LIMIT + truncated text).
    dataset_glob = f"{root}/<dataset>_*/outputs/main/*.parquet"
    return [
        ExampleQuery(
            "Browse normalized datasets",
            "List every normalized dataset directory. Uses glob() (object listing only) over the "
            "first shard of each dataset, so it reads no parquet data.",
            "SELECT DISTINCT regexp_extract(file, 'normalized/(.+)_[0-9a-f]{8}/outputs', 1) AS dataset\n"
            f"FROM glob('{root}/**/outputs/main/part-00000-*.parquet')\n"
            "ORDER BY dataset",
        ),
        ExampleQuery(
            "Count rows in a dataset",
            "Total rows for one normalized dataset. count(*) is answered from parquet footers, so "
            "it doesn't read the id/text data. Replace <dataset> with a name from the browse query.",
            f"SELECT count(*) AS rows\nFROM read_parquet('{dataset_glob}')",
        ),
        ExampleQuery(
            "Sample normalized rows",
            "Peek at id/text from one normalized dataset. LIMIT reads only the first shard's first "
            "row group, and left() truncates the text. Replace <dataset> with a browsed name.",
            f"SELECT id, length(text) AS chars, left(text, 500) AS preview\n"
            f"FROM read_parquet('{dataset_glob}')\nLIMIT 20",
        ),
    ]


def _select_examples(
    examples: list[ExampleQuery], candidates: list[View], available: set[str] | None
) -> list[ExampleQuery]:
    """Keep only examples every view they reference is actually available.

    An example referencing a view that wasn't created (absent dataset, or a root outside the
    allowlist) would error if run, so it's dropped. Examples that reference no view (the
    datakit read_parquet/glob templates) are kept — the caller gates those on the source
    having any available view.
    """
    if available is None:
        return examples
    names = [view.qualified_name for view in candidates]
    return [e for e in examples if all(name in available for name in names if name in e.sql)]


def build_catalog(config: DuckyConfig, available: set[str] | None = None) -> Catalog:
    """Assemble the pre-baked catalog from the configured source roots.

    A source with no configured root contributes nothing. When ``available`` is given (the set
    of view identifiers the runner actually created), each source contributes only its
    available views, and only if it has at least one — so the dashboard never advertises a view
    that was skipped (absent dataset, or a root outside the bucket allowlist). ``available=None``
    returns the full potential catalog (what the runner iterates over to decide what to create).
    """
    views: list[View] = []
    examples: list[ExampleQuery] = []

    if config.finelog_root:
        finelog_views = _finelog_views(config.finelog_root)
        present = [v for v in finelog_views if available is None or v.qualified_name in available]
        if present:
            views.extend(present)
            examples.extend(_select_examples(_finelog_examples(finelog_views), finelog_views, available))

    if config.datakit_root:
        datakit_views = _datakit_views(config.datakit_root)
        present = [v for v in datakit_views if available is None or v.qualified_name in available]
        if present:
            views.extend(present)
            examples.extend(_datakit_examples(config.datakit_root))

    return Catalog(views=tuple(views), examples=tuple(examples))
