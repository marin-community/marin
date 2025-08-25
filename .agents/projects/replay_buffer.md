# Replay Buffer — Full Spec (Ray Actor + Parquet)
## Status

- [x] Data Types
- [x] ReplayBuffer actor skeleton

## Goals & Context

We need a **Ray Actor** that acts as a distributed replay buffer for async RL (GRPO/AIPO-style). It must:

* Persist rollouts to a blob store or filesystem (**GCS/S3/local**) using **Parquet**.
* Support **grouped rollouts** where a group is K rollouts for the **same (environment, example\_id, policy\_version)**.
* Allow **multiple environment replicas** to contribute rollouts for the same group key.
* Provide **deterministic sampling** for reproducibility.
* Enable **strict** or **mixed-policy** grouping.
* Recover cleanly after preemptions.
* Long Term: Support **partial rollouts** (segmented completions). p4


## Basic Flow

- env replicas write rollouts to a known location. notify replaybuffer of new data/pointers
- Typically there is one replay buffer per env type?
- ReplayBuffer reads rollouts from parquet.
- ReplayBuffer feeds them to BatchMaker
- Trainer asks BatchMaker for a batch. It reads from the replay buffer.



### What does trainer want?
Trainer wants a batch of size K consisting of packed sequences of S tokens. Will settle for padding if necessery.
Trainer can ask a BatchMaker for this batch. It does so via an intermediary.

### BatchMakers:

- RlooBatchMaker -> standard
- StableMixtureBatchMaker -> asks subbatchmakers?


## Stupidest thing that can work:

ReplayBuffer receives rollouts. Groups them by (env, problemid). Tries to main the best batch for each one.

- [ ] in memory "working set"
- [ ] append rollouts
- [ ] supports purge
- [ ] supports sample
- [ ] hardcodes grpo or similar

Smarter:

- [ ] stores replays to parquet. periodic flush of working set. data loss is possible but nothing major.
- [ ] support flush
- [ ] on restore, loads working set.
- [ ] when a batch is sampled, writes it disk. When a batch is requested, it first checks if that step is already available.
- [ ] batchMaker



---

## Data Types

Change the existing datatypes in datatypes.py to match the new requirements. The following dataclasses define the core data model.

Make these frozen.

```python
@dataclass
class RolloutRecord:
    environment: str
    example_id: str
    policy_version: str
    replica_id: str = "unknown"
    rollout_uid: str = ""
    token_count: int = 0
    reward: Optional[float] = None
    logprobs: Optional[np.ndarray] = None
    output_tokens: Optional[np.ndarray] = None
    metadata: Optional[dict] = None
    created_ts: float = field(default_factory=lambda: time.time())

@dataclass
class RolloutGroup:
    id: str
    environment: str
    example_id: str
    policy_version: str
    rollouts: list[RolloutRecord]
    sealed_ts: float
    metadata: dict

@dataclass(frozen=True)
class GroupKey:
    environment: str
    example_id: str
    policy_version: str

@dataclass(frozen=True)
class SampledBatch:
    batch_id: str
    group_ids: list[str]
    ts: float
```

---

## Implementation Sketch

```python
@ray.remote(max_concurrency=1)
class ReplayBuffer:
    def __init__(self, root_path: str, *, compression: str = "zstd", capacity_groups: int = 50_000,
                 target_group_size: int = 8, min_group_size: int = 2, seal_timeout_s: float = 30.0,
                 max_per_replica: Optional[int] = None, accept_policy_versions: Optional[Iterable[str]] = None):
        self.root_path = root_path
        self.compression = compression
        self.capacity = capacity_groups
        self.target_size = target_group_size
        self.min_size = min_group_size
        self.seal_timeout_s = seal_timeout_s
        self.max_per_replica = max_per_replica
        self.accept_policy_versions = set(accept_policy_versions) if accept_policy_versions else None
        self.groups = {}
        self.strict = {}
        self.mixed = {}
        self.pending = {}
        self.pending_acks = {}
        self.rr_keys_strict = collections.deque()
        self.rr_keys_mixed = collections.deque()

    def add_rollout(self, r: RolloutRecord):
        key = GroupKey(r.environment, r.example_id, r.policy_version)
        if self.accept_policy_versions and key.policy_version not in self.accept_policy_versions:
            return
        agg = self.pending.setdefault(key, {"uids": set(), "rollouts": [], "by_replica": collections.Counter(),
                                             "created_ts": time.time(), "last_update_ts": time.time()})
        if r.rollout_uid in agg["uids"]:
            return
        if self.max_per_replica and agg["by_replica"][r.replica_id] >= self.max_per_replica:
            return
        agg["uids"].add(r.rollout_uid)
        agg["rollouts"].append(r)
        agg["by_replica"][r.replica_id] += 1
        agg["last_update_ts"] = time.time()
        self._maybe_seal(key, agg)

    def _maybe_seal(self, key: GroupKey, agg: dict, force=False):
        ready = len(agg["rollouts"]) >= self.target_size
        timed_out = (time.time() - agg["created_ts"] >= self.seal_timeout_s) and len(agg["rollouts"]) >= self.min_size
        if not (ready or timed_out or force):
            return
        gid = self._stable_group_id(key, agg["uids"])
        group = RolloutGroup(id=gid, environment=key.environment, example_id=key.example_id,
                             policy_version=key.policy_version,
                             rollouts=list(agg["rollouts"]), sealed_ts=time.time(),
                             metadata={"num_rollouts": len(agg["rollouts"]), "uids": sorted(agg["uids"]),
                                       "replicas": sorted(agg["by_replica"])} )
        self.groups[gid] = group
        self._index_group(key, gid)
        self.pending[key] = {"uids": set(), "rollouts": [], "by_replica": collections.Counter(),
                             "created_ts": time.time(), "last_update_ts": time.time()}

    def _stable_group_id(self, key: GroupKey, uids: set) -> str:
        base = f"{key.environment}|{key.example_id}|{key.policy_version}|{'/'.join(sorted(uids))}"
        return "g-" + hashlib.blake2b(base.encode(), digest_size=12).hexdigest()

    def _index_group(self, key: GroupKey, gid: str):
        self.strict.setdefault(key, collections.deque()).append(gid)
        self.rr_keys_strict.append(key)
        mk = (key.environment, key.example_id)
        self.mixed.setdefault(mk, collections.deque()).append(gid)
        self.rr_keys_mixed.append(mk)
```

# Prioritized TODO Checklist (single list)

- [ ] Core data model #P1
  - [ ] Include `policy_version`, and deterministic `group_id` at seal time.
  - [ ] Dedupe semantics: rollout-level (`rollout_uid`) within aggregators and group-level (`group_id`) across the dataset.
  - [ ] Finalize Parquet schema and partitioning: `env/pv/seg`.
- [ ] Ray Actor interface #P1
  - [ ] Main actor
  - [ ] Implement `add_rollout(rollout: RolloutRecord)` to add rollouts to the buffer.
  - [ ] Implement `_maybe_seal(key, aggregator)` to seal groups based on size or timeout.
  - [ ] Implement `_stable_group_id(key, uids)` for deterministic group IDs.
  - [ ] Implement `_index_group(key, group_id)` to maintain strict/mixed indexes.
- [ ] Persistence \(write path\) #P1
  - [ ] Implement `write_rollout_groups(root_path, groups, compression)` that appends rows to a partitioned Parquet dataset.
  - [ ] Ensure idempotency \(skip if `group_id` already present via manifest\).
  - [ ] Add optional `manifest.jsonl` per partition: `{group_id, sealed_ts, counts, paths}`.
- [ ] Startup \& recovery #P2
  - [ ] Implement `rebuild_indexes()` by scanning manifests \(or partitions if manifest absent\).
  - [ ] Optionally replay an `ingest log` \(JSONL\) of pending, unsealed rollouts to rebuild aggregators.
- [ ] Group assembly \(aggregator\) #P2
  - [ ] Maintain `pending[GroupKey]` with `{uids, rollouts, by_replica, created_ts, last_update_ts}`.
  - [ ] Seal policy: `len(rollouts) >= target_group_size` or `(timeout && len >= min_group_size)`.
  - [ ] Enforce `max_per_replica` if set.
- [ ] Sampling #P3
  - [ ] Deterministic ordering by `group_id` with `seed`+`start_offset` for reproducible replay.
  - [ ] Strict vs mixed indexes; implement `on_policy_fraction` for hybrid batches.
- [ ] Indexing \& capacity #P3
  - [ ] Maintain `strict[(env, example_id, seg, pv)]` and `mixed[(env, example_id, seg)]` deques.
  - [ ] Round-robin over keys for fairness; evict oldest across buckets when over `capacity_groups`.
- [ ] Partial rollouts \& segments #P4
  - [ ] Store `is_last_segment`; allow training on partial segments.
  - [ ] \(Optional\) Support `append_to_group()` prior to sealing if using multi-stage ingestion.
- [ ] Observability \& retention #P4
  - [ ] Metrics: ingest/seal rates, time-to-seal, ready groups, sampling/ack latency, dedupe, evictions.
  - [ ] Retention policy: TTL or per-\(env,pv\) caps \(do not evict groups in un-acked batches\).
- [ ] Tests #P4
  - [ ] Idempotent writes, dedupe, timeout sealing, strict vs mixed sampling, seed+offset replay, recovery from restart.
- [ ] Add `append_to_group()` prior to sealing if using multi-stage ingestion.
- [ ] Add `segment_idx` and `is_last_segment` to `RolloutRecord` and `RolloutGroup` to support partial rollouts. p4

---

# Implementation Sketches (drop-in snippets)

> These sketches fill the TODOs in the spec and are designed to be lifted into your codebase with minimal changes.

## Parquet I/O helpers

```python
# parquet_io.py
from __future__ import annotations
import os, io, time
from typing import Iterable, List, Dict
import pyarrow as pa
import pyarrow.parquet as pq

PARTITION_COLS = ["environment", "policy_version", "segment_idx"]

def _rows_from_group(group: dict) -> List[Dict]:
    rows = []
    for r in group["rollouts"]:
        rows.append({
            "environment": group["environment"],
            "example_id": group["example_id"],
            "policy_version": group["policy_version"],
            "group_id": group["id"],
            "rollout_uid": r["rollout_uid"],
            "replica_id": r.get("replica_id", "unknown"),
            "created_ts": r.get("created_ts", time.time()),
            "sealed_ts": group["sealed_ts"],
            "token_count": r.get("token_count", 0),
            "reward": r.get("reward"),
            "logprobs": r.get("logprobs"),
            "output_tokens": r.get("output_tokens"),
            "metadata": r.get("metadata"),
        })
    return rows

def write_rollout_groups(root_path: str, groups: List[dict], compression: str = "zstd") -> int:
    """Append groups to a partitioned Parquet dataset at root_path.
    Returns number of rows written.
    """
    if not groups:
        return 0
    rows: List[Dict] = []
    for g in groups:
        rows.extend(_rows_from_group(g))
    table = pa.Table.from_pylist(rows)
    pq.write_to_dataset(
        table,
        root_path,
        partition_cols=PARTITION_COLS,
        compression=compression,
        use_dictionary=True,
        existing_data_behavior="overwrite_or_ignore",  # safe with manifest dedupe
    )
    return len(rows)
```

## Manifest index (optional but recommended)

```python
# manifest.py
import json, os
from dataclasses import dataclass
from typing import Iterable, Dict

@dataclass
class ManifestEntry:
    group_id: str
    sealed_ts: float
    num_rollouts: int
    rel_paths: list[str]  # files written by this group

class ManifestIndex:
    def __init__(self, root_path: str):
        self.root = root_path

    def _manifest_path(self, env: str, pv: str, seg: int) -> str:
        return os.path.join(self.root, f"env={env}", f"pv={pv}", f"seg={seg}", "manifest.jsonl")

    def add(self, env: str, pv: str, seg: int, e: ManifestEntry) -> None:
        mp = self._manifest_path(env, pv, seg)
        os.makedirs(os.path.dirname(mp), exist_ok=True)
        with open(mp, "a", encoding="utf-8") as f:
            f.write(json.dumps(e.__dict__) + "
")

    def scan(self) -> Dict[str, dict]:
        """Return group_id -> minimal metadata by walking all manifests."""
        out = {}
        for dirpath, _, filenames in os.walk(self.root):
            if "manifest.jsonl" in filenames:
                with open(os.path.join(dirpath, "manifest.jsonl"), "r", encoding="utf-8") as f:
                    for line in f:
                        rec = json.loads(line)
                        out[rec["group_id"]] = rec
        return out
```

## Index rebuild

```python
# inside ReplayBuffer
def rebuild_indexes(self) -> None:
    # Prefer manifests; fall back to scanning partitions if missing
    try:
        from .manifest import ManifestIndex  # adjust import to your layout
        mi = ManifestIndex(self.root_path)
        groups_meta = mi.scan()  # group_id -> {...}
        # NOTE: you still need to materialize per-group lightweight headers
        # (env, example_id, pv, seg). Read the minimal columns from Parquet.
    except Exception:
        groups_meta = {}

    # Minimal scan over dataset headers to rebuild ready indexes
    import pyarrow.dataset as ds
    dataset = ds.dataset(self.root_path, format="parquet", partitioning="hive")
    cols = ["group_id", "environment", "example_id", "policy_version", "sealed_ts"]
    scanner = dataset.scanner(columns=cols)
    batches = scanner.to_batches()
    seen = set()
    for batch in batches:
        tbl = batch.to_pydict()
        for gid, env, ex, pv in zip(tbl["group_id"], tbl["environment"], tbl["example_id"], tbl["policy_version"]):
            if gid in seen:
                continue
            seen.add(gid)
            # We index by group_id only; actual rollouts are fetched lazily via get_groups if you wire that to storage
            meta = {
                "id": gid, "environment": env, "example_id": ex, "policy_version": pv
            }
            self._index_header(meta)

def _index_header(self, header: dict) -> None:
    gid = header["id"]
    self.groups.setdefault(gid, header)  # header-only until fetched/expanded
    k = GroupKey(header["environment"], header["example_id"], header["policy_version"])
    self.strict.setdefault(k, collections.deque()).append(gid)
    mk = (k.environment, k.example_id)
    self.mixed.setdefault(mk, collections.deque()).append(gid)
```

## Timeout sealing / housekeeping

```python
# inside ReplayBuffer
def tick(self) -> None:
    now = time.time()
    for key, agg in list(self.pending.items()):
        if agg["rollouts"] and (now - agg["created_ts"]) >= self.seal_timeout_s and len(agg["rollouts"]) >= self.min_group:
            self._maybe_seal(key, agg, force=True)
```

## Hybrid sampling helper (on-policy fraction)

```python
# inside ReplayBuffer.sample_groups(...) before returning
if on_policy_fraction and not policy_version:
    n_on = max(0, min(num_groups, int(num_groups * on_policy_fraction)))
    # First, strict keys matching *current* policy versions present in memory
    choose_from(self.rr_keys_strict, self.strict, n_on)
    choose_from(self.rr_keys_mixed, self.mixed, num_groups - len(picked))
```

## Optional ingest log (pending durability)

```python
# ingest_log.py (optional)
import json, os

def append_ingest(root: str, r: dict) -> None:
    path = os.path.join(root, "ingest_logs", f"env={r['environment']}", f"pv={r['policy_version']}")
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, "pending.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps(r) + "
")

# On startup, replay these JSONL files to rebuild self.pending and then let normal sealing run.
```

## Tests to start with

* **Idempotency:** duplicate `rollout_uid` and duplicate `group_id` writes do not produce extra rows.
* **Timeout sealing:** groups seal after `seal_timeout_s` when `min_group_size` met.
* **Strict vs mixed sampling:** buckets and RR fairness verified.
* **Determinism:** same `seed`+`start_offset` → same batches.
* **Recovery:** after restart (`rebuild_indexes()`), ready groups equal pre-restart (minus any unsealed aggregators).
* **Capacity:** when over `capacity_groups`, oldest groups are evicted and never from un-acked batches.
*

---

### Additional Requirements for GRPO, Async-On-Policy, and Magistral Variants

#### 1. Provenance for Off-Policy Correction

* Store **actor policy logprobs** at generation time (μ) so trainers can do AIPO/IS corrections.
* Fields to add to `RolloutRecord`:

  * `actor_logprobs` (per-token array or aggregated sum)
  * `actor_policy_version` (policy ID used to generate the rollout)
  * Optionally `ref_logprobs` (reference policy) if KL penalties are desired.

#### 2. Recency Controls for “Close-to-On-Policy” Sampling

Add sampler parameters:

```
prefer_policy_versions: Optional[Iterable[str]] = None
max_policy_lag: Optional[int] = None         # max version diff from current
recent_n_versions: Optional[int] = None     # sample only from last N versions
recency_weighting: Literal["uniform","linear","exp"] = "uniform"
on_policy_fraction: Optional[float] = None  # already present in spec

```

Also log a histogram of observed policy lag in `stats()`.

#### 3. RLVR / Verifier Side-Channel

* Support post-generation scoring and constraints:

```
def upsert_scores(group_id: str, scores: Dict[str, float], flags: Optional[Dict[str, bool]] = None) -> None

```

* Write verifier scores to a **sidecar Parquet dataset** keyed by `(group_id, rollout_uid)` so raw rollouts remain immutable.

#### 4. Trainer Contract Notes (GRPO + Magistral Tweaks)

* Make sure buffer provides:

  * Per-token and/or summed logprobs
  * Token counts
  * Rewards (multiple components if needed)
* Trainers may implement:

  * Loss normalization by total tokens
  * Advantage normalization
  * No-KL variant
  * Clip-Higher variant

#### 5. Partial Rollout Continuity Metadata

* Add to `RolloutRecord`:

  * `continuation_of: Optional[str]` (prior `rollout_uid`)
  * `prefix_hash: Optional[str]` (hash of prefix tokens to verify continuity)

#### 6. Missing API Sketches to Fill

* `sample_groups()` — implement with recency controls and hybrid on-policy/mixed sampling.
* `get_groups(group_ids)` — fetch payloads by ID.
* `ack(batch_id, status)` — ensure no eviction of un-acked groups.
* `rebuild_indexes()` — rebuild both strict/mixed indexes from Parquet manifests.

#### 7. Storage Layout Note

* Keep raw rollout dataset **append-only**.
* Store verifier scores/flags in a **separate table** to allow re-scoring without rewriting large row groups.

#### 8. Observability

In `stats()`:

* `policy_lag_histogram`
* `ready_by_policy_version`
* `seal_time_ms_p50/p99`
* `generator_queue_len`
* `verifier_latency_ms`

#### 9. Infra Note

* These recency controls rely on **fast generator weight updates** (NCCL/DDMA or equivalent), as used in Magistral/LlamaRL.
