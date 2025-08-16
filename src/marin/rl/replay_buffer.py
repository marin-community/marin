import collections
import hashlib
import time
from typing import Iterable, Optional

import ray

from .datatypes import GroupKey, RolloutGroup, RolloutRecord


@ray.remote(max_concurrency=1)
class ReplayBuffer:
    """Minimal Ray actor implementing a grouped rollout replay buffer."""

    def __init__(
        self,
        root_path: str,
        *,
        compression: str = "zstd",
        capacity_groups: int = 50_000,
        target_group_size: int = 8,
        min_group_size: int = 2,
        seal_timeout_s: float = 30.0,
        max_per_replica: Optional[int] = None,
        accept_policy_versions: Optional[Iterable[str]] = None,
    ) -> None:
        self.root_path = root_path
        self.compression = compression
        self.capacity = capacity_groups
        self.target_size = target_group_size
        self.min_size = min_group_size
        self.seal_timeout_s = seal_timeout_s
        self.max_per_replica = max_per_replica
        self.accept_policy_versions = (
            set(accept_policy_versions) if accept_policy_versions else None
        )

        self.groups: dict[str, RolloutGroup] = {}
        self.strict: dict[GroupKey, list[str]] = {}
        self.mixed: dict[GroupKey, list[str]] = {}
        self.pending: dict[GroupKey, dict] = {}
        self.pending_acks: dict[str, float] = {}
        self.rr_keys_strict = collections.deque()
        self.rr_keys_mixed = collections.deque()

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------
    def add_rollout(self, r: RolloutRecord) -> None:
        key = GroupKey(r.environment, r.example_id, r.policy_version, r.segment_idx)
        if self.accept_policy_versions and key.policy_version not in self.accept_policy_versions:
            return
        agg = self.pending.setdefault(
            key,
            {
                "uids": set(),
                "rollouts": [],
                "by_replica": collections.Counter(),
                "created_ts": time.time(),
                "last_update_ts": time.time(),
            },
        )
        if r.rollout_uid in agg["uids"]:
            return
        if self.max_per_replica and agg["by_replica"][r.replica_id] >= self.max_per_replica:
            return
        agg["uids"].add(r.rollout_uid)
        agg["rollouts"].append(r)
        agg["by_replica"][r.replica_id] += 1
        agg["last_update_ts"] = time.time()
        self._maybe_seal(key, agg)

    # ------------------------------------------------------------------
    def _stable_group_id(self, key: GroupKey, uids: set[str]) -> str:
        h = hashlib.sha1()
        h.update(f"{key.environment}|{key.example_id}|{key.policy_version}|{key.segment_idx}".encode())
        for uid in sorted(uids):
            h.update(uid.encode())
        return h.hexdigest()

    def _index_group(self, key: GroupKey, gid: str) -> None:
        self.strict.setdefault(key, []).append(gid)
        self.rr_keys_strict.append(gid)

    def _maybe_seal(self, key: GroupKey, agg: dict, force: bool = False) -> None:
        ready = len(agg["rollouts"]) >= self.target_size
        timed_out = (
            time.time() - agg["created_ts"] >= self.seal_timeout_s
            and len(agg["rollouts"]) >= self.min_size
        )
        if not (ready or timed_out or force):
            return
        gid = self._stable_group_id(key, agg["uids"])
        group = RolloutGroup(
            id=gid,
            environment=key.environment,
            example_id=key.example_id,
            policy_version=key.policy_version,
            segment_idx=key.segment_idx,
            rollouts=list(agg["rollouts"]),
            sealed_ts=time.time(),
            metadata={
                "num_rollouts": len(agg["rollouts"]),
                "uids": sorted(agg["uids"]),
                "replicas": sorted(agg["by_replica"]),
            },
        )
        self.groups[gid] = group
        self._index_group(key, gid)
        self.pending[key] = {
            "uids": set(),
            "rollouts": [],
            "by_replica": collections.Counter(),
            "created_ts": time.time(),
            "last_update_ts": time.time(),
        }

    # Utility methods for tests ------------------------------------------------
    def flush(self) -> None:
        for key, agg in list(self.pending.items()):
            if agg["rollouts"]:
                self._maybe_seal(key, agg, force=True)

    def list_groups(self) -> list[RolloutGroup]:
        return list(self.groups.values())
