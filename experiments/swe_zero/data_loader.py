# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Load and sample PRs from SWE-rebench V2 for execution-free rollout generation."""

import logging
import random
from dataclasses import dataclass, field

from datasets import Dataset, load_dataset

logger = logging.getLogger(__name__)

DEFAULT_DATASET_ID = "nebius/SWE-rebench-V2"


@dataclass
class PRRecord:
    """A single PR / task from SWE-rebench V2."""

    instance_id: str
    repo: str
    base_commit: str
    problem_statement: str
    pr_description: str
    patch: str
    test_patch: str
    language: str
    interface: str
    created_at: str
    fail_to_pass: list[str] = field(default_factory=list)
    pass_to_pass: list[str] = field(default_factory=list)

    @classmethod
    def from_hf_row(cls, row: dict) -> "PRRecord":
        return cls(
            instance_id=row["instance_id"],
            repo=row["repo"],
            base_commit=row["base_commit"],
            problem_statement=row["problem_statement"],
            pr_description=row.get("pr_description", ""),
            patch=row["patch"],
            test_patch=row["test_patch"],
            language=row.get("language", "unknown"),
            interface=row.get("interface", ""),
            created_at=str(row.get("created_at", "")),
            fail_to_pass=row.get("FAIL_TO_PASS", []),
            pass_to_pass=row.get("PASS_TO_PASS", []),
        )


class SWERebenchV2Loader:
    """Loads SWE-rebench V2 and provides sampling utilities."""

    def __init__(self, language_filter: str | None = None, dataset_id: str = DEFAULT_DATASET_ID):
        logger.info("Loading dataset %s from HuggingFace...", dataset_id)
        self._ds: Dataset = load_dataset(dataset_id, split="train")
        # Build lightweight indexes without copying full rows into memory.
        # The HF Dataset stays memory-mapped; we only store instance_id -> row index.
        # Use batch column access for speed (avoids per-row dict construction).
        instance_ids = self._ds["instance_id"]
        repos = self._ds["repo"]
        languages = self._ds["language"] if "language" in self._ds.column_names else [None] * len(self._ds)
        self._id_to_idx: dict[str, int] = {}
        self._by_repo: dict[str, list[str]] = {}
        for idx, (iid, repo, lang) in enumerate(zip(instance_ids, repos, languages, strict=True)):
            self._id_to_idx[iid] = idx
            lang = lang or "unknown"
            if language_filter and lang != language_filter:
                continue
            self._by_repo.setdefault(repo, []).append(iid)
        logger.info(
            "Loaded %d instances across %d repos (filter=%s)",
            len(self._id_to_idx),
            len(self._by_repo),
            language_filter,
        )

    @property
    def repos(self) -> list[str]:
        return sorted(self._by_repo.keys())

    def instances_for_repo(self, repo: str) -> list[str]:
        return self._by_repo.get(repo, [])

    def get(self, instance_id: str) -> PRRecord:
        idx = self._id_to_idx[instance_id]
        return PRRecord.from_hf_row(self._ds[idx])

    def sample_repos(self, n: int, min_prs: int = 10, seed: int = 42) -> list[str]:
        """Sample n repos that each have at least min_prs PRs."""
        rng = random.Random(seed)
        eligible = [r for r, ids in self._by_repo.items() if len(ids) >= min_prs]
        if len(eligible) < n:
            logger.warning("Only %d repos have >= %d PRs (requested %d)", len(eligible), min_prs, n)
            n = len(eligible)
        return rng.sample(eligible, n)

    def sample_prs(self, repo: str, n: int, seed: int = 42) -> list[PRRecord]:
        """Sample n PRs from a given repo."""
        rng = random.Random(seed)
        ids = self._by_repo.get(repo, [])
        if len(ids) < n:
            logger.warning("Repo %s has only %d PRs (requested %d)", repo, len(ids), n)
            n = len(ids)
        sampled = rng.sample(ids, n)
        return [self.get(iid) for iid in sampled]
