# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Content-addressed caching for generated documentation.

Each doc entry is keyed by a hash of its source code and the hashes of its
dependency docs. If nothing changed, nothing regenerates.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

from agent_docs.graph import ClassInfo, FunctionInfo, RepoGraph

logger = logging.getLogger(__name__)

CACHE_FILE = "docs/agent/.cache.json"


@dataclass
class CacheEntry:
    source_hash: str
    dep_doc_hashes: list[str] = field(default_factory=list)
    doc_hash: str = ""
    tier: int = 3


@dataclass
class DocCache:
    entries: dict[str, CacheEntry] = field(default_factory=dict)

    def is_stale(self, key: str, source_hash: str, dep_doc_hashes: list[str] | None = None) -> bool:
        if key not in self.entries:
            return True
        entry = self.entries[key]
        if entry.source_hash != source_hash:
            return True
        if dep_doc_hashes is not None and entry.dep_doc_hashes != dep_doc_hashes:
            return True
        return False

    def update(self, key: str, source_hash: str, doc_hash: str, tier: int, dep_doc_hashes: list[str] | None = None):
        self.entries[key] = CacheEntry(
            source_hash=source_hash,
            dep_doc_hashes=dep_doc_hashes or [],
            doc_hash=doc_hash,
            tier=tier,
        )


def load_cache(repo_root: Path) -> DocCache:
    cache_path = repo_root / CACHE_FILE
    if not cache_path.exists():
        return DocCache()
    try:
        data = json.loads(cache_path.read_text())
        entries = {}
        for key, val in data.items():
            entries[key] = CacheEntry(
                source_hash=val["source_hash"],
                dep_doc_hashes=val.get("dep_doc_hashes", []),
                doc_hash=val.get("doc_hash", ""),
                tier=val.get("tier", 3),
            )
        return DocCache(entries=entries)
    except (json.JSONDecodeError, KeyError):
        logger.warning("Failed to load cache, starting fresh")
        return DocCache()


def save_cache(cache: DocCache, repo_root: Path) -> None:
    cache_path = repo_root / CACHE_FILE
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    data = {}
    for key, entry in sorted(cache.entries.items()):
        data[key] = {
            "source_hash": entry.source_hash,
            "dep_doc_hashes": entry.dep_doc_hashes,
            "doc_hash": entry.doc_hash,
            "tier": entry.tier,
        }
    cache_path.write_text(json.dumps(data, indent=2) + "\n")


def _combined_source_hash(items: list[FunctionInfo | ClassInfo]) -> str:
    """Hash the combined source of all items in a module."""
    combined = "\n".join(sorted(item.source_hash for item in items))
    return hashlib.sha256(combined.encode()).hexdigest()[:16]


def compute_stale_modules(graph: RepoGraph, cache: DocCache) -> set[str]:
    """Determine which modules need doc regeneration.

    A module is stale if any of its functions/classes have changed source hashes,
    or if the module itself is not in the cache.
    """
    stale: set[str] = set()
    for mod_name, mod in graph.modules.items():
        public_items: list[FunctionInfo | ClassInfo] = [f for f in mod.functions if f.is_public] + [
            c for c in mod.classes if c.is_public
        ]
        if not public_items:
            continue
        combined_hash = _combined_source_hash(public_items)
        if cache.is_stale(mod_name, combined_hash):
            stale.add(mod_name)

    # Propagate staleness: if module A imports from module B and B is stale,
    # A's docs should also be regenerated (callee context changed)
    propagated = True
    while propagated:
        propagated = False
        for mod_name, mod in graph.modules.items():
            if mod_name in stale:
                continue
            if mod.imports_from & stale:
                stale.add(mod_name)
                propagated = True

    return stale


def hash_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]
