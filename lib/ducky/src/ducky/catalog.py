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
# example query rather than baking every one as an eagerly-bound view.
_DATAKIT_SOURCES: tuple[tuple[str, str, str], ...] = (
    ("finetranslations", "finetranslations", "Normalized translation corpus (~3.0T tokens)."),
    ("nemotron_cc_v2_high_quality", "nemotron_cc_v2/high_quality", "Nemotron-CC v2 high-quality subset."),
    (
        "nemotron_cc_v2_medium_quality",
        "nemotron_cc_v2/medium_quality",
        "Nemotron-CC v2 medium-quality subset (~2.1T tokens).",
    ),
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
            "Recent log lines",
            "Latest 100 rows from the base finelog log.",
            f"SELECT epoch_ms, level, source, key, data\nFROM {by_name['log']}\nORDER BY seq DESC\nLIMIT 100",
        ),
        ExampleQuery(
            "Iris task resource usage",
            "Most recent per-task cpu/memory samples.",
            f"SELECT task_id, attempt_id, worker_id, ts, cpu_millicores, memory_mb, memory_peak_mb\n"
            f"FROM {by_name['iris.task']}\nORDER BY ts DESC\nLIMIT 100",
        ),
        ExampleQuery(
            "Worker utilization snapshot",
            "Latest heartbeat per worker with cpu/memory and running task count.",
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


def _datakit_examples(root: str, views: list[View]) -> list[ExampleQuery]:
    root = root.rstrip("/")
    sample = views[0].qualified_name if views else "datakit.finetranslations"
    return [
        ExampleQuery(
            "Browse normalized datasets",
            "List every normalized dataset directory (one glob over the first shard of each).",
            "SELECT DISTINCT regexp_extract(file, 'normalized/(.+)_[0-9a-f]{8}/outputs', 1) AS dataset\n"
            f"FROM glob('{root}/**/outputs/main/part-00000-*.parquet')\n"
            "ORDER BY dataset",
        ),
        ExampleQuery(
            "Sample normalized rows",
            "Peek at id/text from a curated normalized dataset.",
            f"SELECT id, length(text) AS text_len, text\nFROM {sample}\nLIMIT 20",
        ),
    ]


def build_catalog(config: DuckyConfig) -> Catalog:
    """Assemble the pre-baked catalog from the configured source roots.

    A source with no configured root contributes nothing. Example queries reference the
    views by name (finelog) or the root directly (datakit browse), so they line up with
    whatever views got built.
    """
    views: list[View] = []
    examples: list[ExampleQuery] = []

    if config.finelog_root:
        finelog_views = _finelog_views(config.finelog_root)
        views.extend(finelog_views)
        examples.extend(_finelog_examples(finelog_views))

    if config.datakit_root:
        datakit_views = _datakit_views(config.datakit_root)
        views.extend(datakit_views)
        examples.extend(_datakit_examples(config.datakit_root, datakit_views))

    return Catalog(views=tuple(views), examples=tuple(examples))
