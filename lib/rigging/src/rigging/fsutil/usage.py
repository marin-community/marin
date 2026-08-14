# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Object-storage usage scans and deletion-candidate reports."""

import dataclasses
import re
import time
from collections.abc import Callable
from datetime import UTC, datetime

from rigging.fsutil.listing import DEFAULT_LISTING_WORKERS, entry_mtime, is_child, metadata_listing_pages
from rigging.fsutil.render import format_size

TIB = 1024**4
SECONDS_PER_YEAR = 365.25 * 24 * 60 * 60
DEFAULT_USAGE_WORKERS = DEFAULT_LISTING_WORKERS


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
    prefixes_scanned: int
    elapsed_seconds: float


@dataclasses.dataclass(frozen=True)
class ScanProgress:
    """Monotonic progress observed while directory listings complete."""

    prefixes_scanned: int
    prefixes_discovered: int
    stats: UsageStats


@dataclasses.dataclass(frozen=True)
class PrefixGroup:
    """One non-overlapping row in an adaptive prefix breakdown."""

    label: str
    stats: UsageStats


@dataclasses.dataclass
class _MutablePrefix:
    prefix: str
    direct: UsageStats = UsageStats()
    children: dict[str, "_MutablePrefix"] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass(frozen=True)
class _Component:
    label: str
    stats: UsageStats
    child: PrefixUsage | None


def scan_usage(
    url: str,
    *,
    workers: int = DEFAULT_USAGE_WORKERS,
    prefix_depth: int = 3,
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
    discovered: set[str] = set()
    scanned: set[str] = set()

    for path, entries in metadata_listing_pages(url, workers=workers):
        root_path = root_path or path.rstrip("/")
        scanned.add(path.rstrip("/"))
        for entry in entries:
            if entry.get("type") == "directory":
                if is_child(path, entry["name"]):
                    discovered.add(entry["name"].rstrip("/"))
                continue
            stats = _object_usage(entry)
            if stats is None or not is_child(path, entry["name"]):
                continue
            _add_object(root, root_path, entry["name"], stats, prefix_depth)
            observed += stats
        if progress is not None:
            progress(
                ScanProgress(
                    prefixes_scanned=len(scanned),
                    prefixes_discovered=len(discovered) + 1,
                    stats=observed,
                )
            )

    return UsageScan(
        url=url,
        root=_freeze(root),
        prefixes_scanned=len(scanned),
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
    relative = name.removeprefix(root_path.rstrip("/") + "/")
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


def adaptive_prefix_groups(scan: UsageScan, min_size_bytes: int) -> list[PrefixGroup]:
    """Partition a scan into readable, non-overlapping prefix groups.

    Every group is at least *min_size_bytes* unless an entire top-level prefix
    is smaller. Adjacent small siblings are collapsed into lexical ranges.
    """
    if min_size_bytes <= 0:
        raise ValueError("min_size_bytes must be positive")

    groups = []
    if scan.root.direct.object_count:
        groups.append(PrefixGroup(label="[root objects]", stats=scan.root.direct))
    for child in scan.root.children:
        groups.extend(_partition(child, min_size_bytes))
    if not groups and scan.root.total.object_count:
        groups.append(PrefixGroup(label="[objects]", stats=scan.root.total))
    return groups


def _partition(node: PrefixUsage, min_size_bytes: int) -> list[PrefixGroup]:
    if node.total.size_bytes < min_size_bytes or not node.children:
        return [PrefixGroup(label=node.prefix, stats=node.total)]
    if not node.direct.object_count and len(node.children) == 1:
        return _partition(node.children[0], min_size_bytes)

    components = []
    if node.direct.object_count:
        components.append(_Component(label="[objects]", stats=node.direct, child=None))
    components.extend(
        _Component(label=_segment(child.prefix), stats=child.total, child=child) for child in node.children
    )
    chunks = _size_chunks(components, min_size_bytes)
    if len(chunks) == 1 and len(chunks[0]) == len(components):
        return [PrefixGroup(label=node.prefix, stats=node.total)]

    groups = []
    for chunk in chunks:
        if len(chunk) == 1 and chunk[0].child is not None:
            groups.extend(_partition(chunk[0].child, min_size_bytes))
            continue
        stats = UsageStats()
        for component in chunk:
            stats += component.stats
        first = chunk[0].label
        last = chunk[-1].label
        suffix = first if first == last else f"{first} … {last}"
        groups.append(PrefixGroup(label=f"{node.prefix}{suffix}", stats=stats))
    return groups


def _size_chunks(components: list[_Component], min_size_bytes: int) -> list[list[_Component]]:
    chunks: list[list[_Component]] = []
    pending: list[_Component] = []
    pending_bytes = 0
    for component in components:
        if component.stats.size_bytes >= min_size_bytes:
            if pending:
                chunks.append(pending)
                pending = []
                pending_bytes = 0
            chunks.append([component])
            continue
        pending.append(component)
        pending_bytes += component.stats.size_bytes
        if pending_bytes >= min_size_bytes:
            chunks.append(pending)
            pending = []
            pending_bytes = 0
    if pending:
        chunks.append(pending)

    index = 0
    while len(chunks) > 1 and index < len(chunks):
        chunk_size = sum(component.stats.size_bytes for component in chunks[index])
        if chunk_size >= min_size_bytes:
            index += 1
            continue
        if index == 0:
            chunks[1] = chunks[0] + chunks[1]
            chunks.pop(0)
        else:
            chunks[index - 1].extend(chunks[index])
            chunks.pop(index)
            index -= 1
    return chunks


def stale_tib_years(stats: UsageStats, now: datetime) -> float | None:
    """Return TiB multiplied by years since the newest object write."""
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


def render_usage_report(scan: UsageScan, *, min_size_bytes: int, generated_at: datetime) -> str:
    """Render a self-contained Markdown usage and deletion-candidate report."""
    now = generated_at.replace(tzinfo=UTC) if generated_at.tzinfo is None else generated_at.astimezone(UTC)
    groups = adaptive_prefix_groups(scan, min_size_bytes)
    ranked = ranked_groups(groups, now)
    total = scan.root.total
    lines = [
        "# Object storage usage report",
        "",
        f"- Target: `{_markdown_code(scan.url)}`",
        f"- Generated: {now.isoformat(timespec='seconds')}",
        f"- Total: {format_size(total.size_bytes)} across {total.object_count:,} objects",
        f"- Scan: {scan.prefixes_scanned:,} prefixes in {scan.elapsed_seconds:.1f}s (metadata only)",
        f"- Grouping floor: {format_size(min_size_bytes)}; smaller siblings are combined, "
        "while small top-level prefixes remain visible",
        "",
        "## Ranked deletion candidates",
        "",
        "Score is stale TiB-years: TiB reclaimed x years since the newest write in the group. "
        "Range labels combine adjacent siblings and cannot be passed directly as deletion prefixes.",
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
    for group in sorted(groups, key=lambda item: item.label.casefold()):
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


def _segment(prefix: str) -> str:
    return prefix.rstrip("/").rsplit("/", 1)[-1] + "/"


def _display_path(url: str, label: str) -> str:
    return f"{url.rstrip('/')}/{label}" if label else url


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
    return f"{days / 365.25:.1f}y"
