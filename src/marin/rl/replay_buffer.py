import collections
import hashlib
import time
import uuid
from collections.abc import Iterable

import jax
import numpy as np
import ray
import pyarrow as pa
import pyarrow.parquet as pq
from jaxtyping import PRNGKeyArray
from pathlib import Path

from .datatypes import GroupKey, RolloutGroup, RolloutRecord, RLExample
from .batch_maker import BatchMaker, _compute_advantages


class ReplayBuffer:
    """

    Basic replay buffer that stores rollouts in memory. This class is deliberately not a Ray actor
    (though it can be decorated with @ray.remote just-in-time or wrapped)

    We'll add support for writing to disk later.
    """

    def __init__(
        self,
        *,
        prng_key: PRNGKeyArray,
        min_group_size: int = 4,
    ) -> None:

        self.prng_key = prng_key
        self.min_group_size = min_group_size
        if min_group_size < 2:
            raise ValueError("min_group_size must be at least 2")

        self.rollout_groups: dict[GroupKey, list[RolloutRecord]] = {}

    def extend(self, rollouts: list[RolloutRecord]):
        for rollout in rollouts:
            key = GroupKey(rollout.environment, rollout.example_id)
            group = self.rollout_groups.get(key)
            if group is None:
                self.rollout_groups[key] = [rollout]
            else:
                group.append(rollout)

    def purge(self):
        self.rollout_groups = {}

    def sample(self, *, bsize: int, step: int):
        # TODO: packing?
        # TODO: should we track advantage stats online
        # TODO: log reward and advantage stats
        # find useful rollout groups (those with some non-zero advantage entry)
        useful_rollouts: list[RolloutRecord] = []
        for _key, group in self.rollout_groups.items():
            if len(group) <= 1:
                continue
            advantages = _compute_advantages(group)

            for rollout, advantage in zip(group, advantages, strict=False):
                if not np.allclose(advantage, 0.0):
                    useful_rollouts.append(rollout)

        this_prng = jax.random.fold_in(self.prng_key, step)
        n = len(useful_rollouts)
        if n == 0:
            return []
        # Permute indices to avoid object array issues
        perm = jax.random.permutation(this_prng, n)
        # Convert to Python ints
        import numpy as _np

        idx = list(map(int, _np.array(perm[:bsize])))
        # TODO: remove selected from pool
        return [useful_rollouts[i] for i in idx]


@ray.remote(max_concurrency=1)
class OldReplayBuffer:
    """Minimal Ray actor implementing a grouped rollout replay buffer.

    This buffer accumulates experiences and can forward them to a BatchMaker
    for creating training batches.
    """

    def __init__(
        self,
        root_path: str,
        *,
        compression: str = "zstd",
        capacity_groups: int = 50_000,
        target_group_size: int = 8,
        min_group_size: int = 2,
        seal_timeout_s: float = 30.0,
        max_per_replica: int | None = None,
        accept_policy_versions: Iterable[str] | None = None,
        batch_maker: BatchMaker | None = None,
    ) -> None:
        self.root_path = root_path
        self.compression = compression
        self.capacity = capacity_groups
        self.target_size = target_group_size
        self.min_size = min_group_size
        self.seal_timeout_s = seal_timeout_s
        self.max_per_replica = max_per_replica
        self.accept_policy_versions = set(accept_policy_versions) if accept_policy_versions else None
        self.batch_maker = batch_maker

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
        key = GroupKey(r.environment, r.example_id, r.policy_version)
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

        # Forward to BatchMaker if available
        if self.batch_maker is not None:
            self._forward_to_batch_maker(r)

        self._maybe_seal(key, agg)

    def _forward_to_batch_maker(self, rollout: RolloutRecord) -> None:
        """Forward a rollout to the BatchMaker for processing.

        Args:
            rollout: RolloutRecord to forward
        """
        try:
            self.batch_maker.add_rollout(rollout)
        except Exception as e:
            # Log error but don't fail the replay buffer
            print(f"Error forwarding rollout to BatchMaker: {e}")

    def create_batch(self, batch_size: int) -> list[RLExample] | None:
        """Create a batch of RL examples using the BatchMaker.

        Args:
            batch_size: Target size of the batch

        Returns:
            List of RL examples, or None if insufficient data
        """
        if self.batch_maker is None:
            return None

        return self.batch_maker.create_batch(batch_size)

    def write_batch_to_disk(self, batch: list[RLExample], batch_id: str) -> None:
        """Write a completed batch to disk as a parquet file.

        Args:
            batch: List of RL examples to write
            batch_id: Unique identifier for the batch
        """
        if not batch:
            return

        # Create batches directory
        batches_dir = Path(self.root_path) / "batches"
        batches_dir.mkdir(parents=True, exist_ok=True)

        # Convert batch to table format
        table = self._batch_to_table(batch, batch_id)

        # Write to parquet file
        filename = f"batch_{batch_id}_{int(time.time())}.parquet"
        filepath = batches_dir / filename

        pq.write_table(table, filepath, compression=self.compression)

    def _batch_to_table(self, batch: list[RLExample], batch_id: str) -> pa.Table:
        """Convert a batch of RL examples to pyarrow table format."""
        rows = []

        for i, example in enumerate(batch):
            row = {
                "batch_id": batch_id,
                "example_idx": i,
                "tokens": example.tokens.tolist(),
                "loss_mask": example.loss_mask.tolist(),
                "advantage": example.advantage.tolist(),
                "generator_log_probs": example.generator_log_probs.tolist(),
                "timestamp": time.time(),
            }
            rows.append(row)

        return pa.Table.from_pylist(rows)

    def create_and_store_batch(self, batch_size: int) -> str | None:
        """Create a batch and store it to disk.

        Args:
            batch_size: Target size of the batch

        Returns:
            Batch ID if successful, None otherwise
        """
        batch = self.create_batch(batch_size)
        if batch is None:
            return None

        # Generate batch ID
        batch_id = f"batch_{uuid.uuid4().hex[:8]}"

        # Store batch to disk
        self.write_batch_to_disk(batch, batch_id)

        return batch_id

    # ------------------------------------------------------------------
    def _stable_group_id(self, key: GroupKey, uids: set[str]) -> str:
        """Create a deterministic group ID that's stable across restarts.

        The group ID is a hash of:
        - Environment name, example ID, and policy version (the group key)
        - Sorted list of rollout UIDs in the group

        This ensures that:
        1. The same set of rollouts always produces the same group ID
        2. We can deduplicate groups across system restarts
        3. We have reproducible training data organization
        4. We can safely write groups to disk without fear of ID conflicts
        """
        h = hashlib.sha1()
        h.update(f"{key.environment}|{key.example_id}|{key.policy_version}".encode())
        for uid in sorted(uids):
            h.update(uid.encode())
        return h.hexdigest()

    def _index_group(self, key: GroupKey, gid: str) -> None:
        """Index a sealed group for sampling and retrieval.

        Groups are indexed in two ways:
        1. By their group key (environment, example_id, policy_version) for
           strict sampling (same problem, same policy)
        2. In a round-robin queue for mixed sampling across different problems

        This indexing enables efficient sampling strategies during training.
        """
        self.strict.setdefault(key, []).append(gid)
        self.rr_keys_strict.append(gid)

    def _maybe_seal(self, key: GroupKey, agg: dict, force: bool = False) -> None:
        """Seal a group of rollouts when it's ready for training."

        Sealing happens when:
        1. We have enough rollouts to form a meaningful training batch (target_size)
        2. The group has been waiting too long and we want to avoid indefinite delays
        3. We're forcing closure (e.g., during shutdown)

        Sealing creates a stable, immutable group that can be:
        - Sampled for training (via strict/mixed sampling)
        - Written to disk as a complete unit
        - Used for advantage computation across rollouts in the same group
        """

        ready = len(agg["rollouts"]) >= self.target_size
        timed_out = time.time() - agg["created_ts"] >= self.seal_timeout_s and len(agg["rollouts"]) >= self.min_size
        if not (ready or timed_out or force):
            return

        # Create a deterministic group ID that will be the same every time
        # for the same set of rollouts. This ensures reproducibility and
        # allows us to deduplicate groups across restarts.
        gid = self._stable_group_id(key, agg["uids"])

        # Create the sealed group with all metadata
        group = RolloutGroup(
            id=gid,
            environment=key.environment,
            example_id=key.example_id,
            policy_version=key.policy_version,
            rollouts=list(agg["rollouts"]),
            sealed_ts=time.time(),
            metadata={
                "num_rollouts": len(agg["rollouts"]),
                "uids": sorted(agg["uids"]),
                "replicas": sorted(agg["by_replica"]),
            },
        )

        # Store the sealed group and index it for sampling
        self.groups[gid] = group
        self._index_group(key, gid)

        # Reset the pending aggregation for this key, allowing new rollouts
        # to start forming the next group
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
