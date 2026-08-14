# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Object-storage usage scans and deletion-candidate reports."""

import dataclasses
import re
import time
from collections.abc import Callable
from datetime import UTC, datetime

from rigging.fsutil.listing import (
    DEFAULT_LISTING_WORKERS,
    ListingPhase,
    entry_mtime,
    is_child,
    metadata_listing_pages,
)
from rigging.fsutil.render import format_size

TIB = 1024**4
DAYS_PER_YEAR = 365.25
SECONDS_PER_YEAR = DAYS_PER_YEAR * 24 * 60 * 60
DEFAULT_USAGE_WORKERS = DEFAULT_LISTING_WORKERS
DEFAULT_PREFIX_DEPTH = 3


@dataclasses.dataclass(frozen=True)
class UsageStats:
    """Aggregate size, object count, and newest write time."""

    size_bytes: int = 0
    object_count: int = 0
    last_modified: datetime | None = None

    def __add__(self, other: "UsageStats") -> "UsageStats":
        modified = max(
            (value for value in (self.last_modified, other.last_modified) if value is not None),
            default=None,
        )
        return UsageStats(
            size_bytes=self.size_bytes + other.size_bytes,
            object_count=self.object_count + other.object_count,
            last_modified=modified,
        )


@dataclasses.dataclass(frozen=True)
class PrefixUsage:
    """A directory prefix and the recursively aggregated metadata below it."""

    prefix: str
    direct: UsageStats
    total: UsageStats
    children: tuple["PrefixUsage", ...]


@dataclasses.dataclass(frozen=True)
class UsageScan:
    """Complete metadata-only scan of one URL."""

    url: str
    root: PrefixUsage
    prefix_depth: int
    listing_pages: int
    elapsed_seconds: float


@dataclasses.dataclass(frozen=True)
class ScanProgress:
    """Monotonic progress observed while metadata pages complete."""

    phase: ListingPhase
    current_prefix: str
    listing_pages: int
    prefixes_completed: int
    prefixes_discovered: int
    stats: UsageStats
    elapsed_seconds: float


@dataclasses.dataclass(frozen=True)
class PrefixGroup:
    """One non-overlapping row in a threshold prefix breakdown."""

    label: str
    stats: UsageStats


@dataclasses.dataclass
class _MutablePrefix:
    prefix: str
    direct: UsageStats = UsageStats()
    children: dict[str, "_MutablePrefix"] = dataclasses.field(default_factory=dict)


def scan_usage(
    url: str,
    *,
    workers: int = DEFAULT_USAGE_WORKERS,
    prefix_depth: int = DEFAULT_PREFIX_DEPTH,
    progress: Callable[[ScanProgress], None] | None = None,
) -> UsageScan:
    """Scan object metadata below *url* and return a prefix tree.

    The tree retains at most *prefix_depth* path components. Memory is bounded
    by the namespace at that depth while preserving useful cleanup rollups.
    """
    if prefix_depth <= 0:
        raise ValueError("prefix_depth must be positive")
    started = time.monotonic()
    root = _MutablePrefix(prefix="")
    root_path: str | None = None
    observed = UsageStats()
    listing_pages = 0

    for page in metadata_listing_pages(url, workers=workers):
        now = time.monotonic()
        root_path = root_path or page.path.removesuffix("/")
        listing_pages = page.pages_completed
        for entry in page.entries:
            if entry.get("type") == "directory":
                continue
            stats = _object_usage(entry)
            if stats is None or not is_child(page.path, entry["name"]):
                continue
            _add_object(root, root_path, entry["name"], stats, prefix_depth)
            observed += stats
        if progress is not None:
            progress(
                ScanProgress(
                    phase=page.phase,
                    current_prefix=page.path,
                    listing_pages=page.pages_completed,
                    prefixes_completed=page.prefixes_completed,
                    prefixes_discovered=page.prefixes_discovered,
                    stats=observed,
                    elapsed_seconds=now - started,
                )
            )

    return UsageScan(
        url=url,
        root=_freeze(root),
        prefix_depth=prefix_depth,
        listing_pages=listing_pages,
        elapsed_seconds=time.monotonic() - started,
    )


def _object_usage(entry: dict) -> UsageStats | None:
    if entry["name"].endswith("/") and not entry.get("size"):
        return None
    return UsageStats(
        size_bytes=int(entry.get("size") or 0),
        object_count=1,
        last_modified=entry_mtime(entry),
    )


def _add_object(root: _MutablePrefix, root_path: str, name: str, stats: UsageStats, prefix_depth: int) -> None:
    root_prefix = root_path if root_path.endswith("/") else f"{root_path}/"
    if not name.startswith(root_prefix):
        raise ValueError(f"object {name!r} is outside scan root {root_path!r}")
    relative = name[len(root_prefix) :]
    directories = relative.split("/")[:-1]
    node = root
    for segment in directories[:prefix_depth]:
        prefix = f"{node.prefix}{segment}/"
        node = node.children.setdefault(segment, _MutablePrefix(prefix=prefix))
    node.direct += stats


def _freeze(node: _MutablePrefix) -> PrefixUsage:
    children = tuple(_freeze(child) for _, child in sorted(node.children.items(), key=lambda item: item[0].casefold()))
    total = node.direct
    for child in children:
        total += child.total
    return PrefixUsage(prefix=node.prefix, direct=node.direct, total=total, children=children)


def threshold_prefix_groups(scan: UsageScan, threshold_bytes: int) -> list[PrefixGroup]:
    """Return exact prefixes found by descending while groups exceed a threshold."""
    if threshold_bytes <= 0:
        raise ValueError("threshold_bytes must be positive")

    groups = []
    if scan.root.direct.object_count:
        groups.append(PrefixGroup(label="[root objects]", stats=scan.root.direct))
    for child in scan.root.children:
        groups.extend(_threshold_groups(child, threshold_bytes))
    if not groups and scan.root.total.object_count:
        groups.append(PrefixGroup(label="[objects]", stats=scan.root.total))
    return sorted(groups, key=lambda group: (-group.stats.size_bytes, group.label.casefold()))


def _threshold_groups(node: PrefixUsage, threshold_bytes: int) -> list[PrefixGroup]:
    if node.total.size_bytes < threshold_bytes or not node.children:
        return [PrefixGroup(label=node.prefix, stats=node.total)]

    groups = []
    if node.direct.object_count:
        groups.append(PrefixGroup(label=f"{node.prefix}[objects]", stats=node.direct))
    for child in node.children:
        groups.extend(_threshold_groups(child, threshold_bytes))
    return groups


def stale_tib_years(stats: UsageStats, now: datetime) -> float | None:
    """Return TiB-years since the newest write, or ``None`` when its time is unknown."""
    if stats.last_modified is None:
        return None
    current = now.replace(tzinfo=UTC) if now.tzinfo is None else now.astimezone(UTC)
    age_seconds = max(0.0, (current - stats.last_modified).total_seconds())
    return stats.size_bytes / TIB * age_seconds / SECONDS_PER_YEAR


def ranked_groups(groups: list[PrefixGroup], now: datetime) -> list[PrefixGroup]:
    """Return groups ordered by stale TiB-years, then raw size."""
    return sorted(
        groups,
        key=lambda group: (stale_tib_years(group.stats, now) or 0.0, group.stats.size_bytes),
        reverse=True,
    )


def render_usage_report(scan: UsageScan, *, threshold_bytes: int, generated_at: datetime) -> str:
    """Render a self-contained Markdown usage and deletion-candidate report."""
    now = generated_at.replace(tzinfo=UTC) if generated_at.tzinfo is None else generated_at.astimezone(UTC)
    groups = threshold_prefix_groups(scan, threshold_bytes)
    ranked = ranked_groups(groups, now)
    total = scan.root.total
    lines = [
        "# Object storage usage report",
        "",
        f"- Target: `{_markdown_code(scan.url)}`",
        f"- Generated: {now.isoformat(timespec='seconds')}",
        f"- Total: {format_size(total.size_bytes)} across {total.object_count:,} objects",
        f"- Scan: {scan.listing_pages:,} listing pages in {scan.elapsed_seconds:.1f}s (metadata only)",
        f"- Prefix threshold: {format_size(threshold_bytes)}; prefixes at or above it are expanded "
        f"through {scan.prefix_depth} path components",
        "",
        "## Ranked deletion candidates",
        "",
        "Score is stale TiB-years: TiB reclaimed x years since the newest write in the prefix.",
        "",
        "| Rank | Score | Size | Share | Objects | Last written | Age | Prefix |",
        "| ---: | ---: | ---: | ---: | ---: | --- | ---: | --- |",
    ]
    for rank, group in enumerate(ranked, start=1):
        score = stale_tib_years(group.stats, now)
        share = group.stats.size_bytes / total.size_bytes if total.size_bytes else 0.0
        lines.append(
            f"| {rank} | {_format_score(score)} | {format_size(group.stats.size_bytes)} | {share:.1%} | "
            f"{group.stats.object_count:,} | {_format_date(group.stats.last_modified)} | "
            f"{_format_age(group.stats.last_modified, now)} | `{_markdown_code(_display_path(scan.url, group.label))}` |"
        )

    lines.extend(
        [
            "",
            "## Prefix breakdown",
            "",
            "| Prefix | Size | Share | Objects | Last written |",
            "| --- | ---: | ---: | ---: | --- |",
        ]
    )
    for group in groups:
        share = group.stats.size_bytes / total.size_bytes if total.size_bytes else 0.0
        lines.append(
            f"| `{_markdown_code(_display_path(scan.url, group.label))}` | {format_size(group.stats.size_bytes)} | "
            f"{share:.1%} | {group.stats.object_count:,} | {_format_date(group.stats.last_modified)} |"
        )
    lines.append("")
    return "\n".join(lines)


def parse_byte_size(value: str) -> int:
    """Parse a positive byte size such as ``1TB`` or ``512GiB``."""
    match = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*([kmgtpe]?i?b)?\s*", value, flags=re.IGNORECASE)
    if match is None:
        raise ValueError(f"invalid byte size: {value!r}")
    number = float(match.group(1))
    unit = (match.group(2) or "B").upper()
    decimal_units = {"B": 0, "KB": 1, "MB": 2, "GB": 3, "TB": 4, "PB": 5, "EB": 6}
    binary_units = {"KIB": 1, "MIB": 2, "GIB": 3, "TIB": 4, "PIB": 5, "EIB": 6}
    if unit in decimal_units:
        result = int(number * 1000 ** decimal_units[unit])
    elif unit in binary_units:
        result = int(number * 1024 ** binary_units[unit])
    else:
        raise ValueError(f"invalid byte size: {value!r}")
    if result <= 0:
        raise ValueError("byte size must be positive")
    return result


def _display_path(url: str, label: str) -> str:
    if not label:
        return url
    separator = "" if url.endswith("/") else "/"
    return f"{url}{separator}{label}"


def _markdown_code(value: str) -> str:
    return value.replace("`", "\\`").replace("|", "\\|")


def _format_score(score: float | None) -> str:
    return "—" if score is None else f"{score:,.2f}"


def _format_date(value: datetime | None) -> str:
    return "unknown" if value is None else value.astimezone(UTC).date().isoformat()


def _format_age(value: datetime | None, now: datetime) -> str:
    if value is None:
        return "unknown"
    days = max(0, (now - value.astimezone(UTC)).days)
    if days < 60:
        return f"{days}d"
    if days < 730:
        return f"{days / 30.44:.1f}mo"
    return f"{days / DAYS_PER_YEAR:.1f}y"
