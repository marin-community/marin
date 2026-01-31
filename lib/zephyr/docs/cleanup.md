# Zephyr execution.py Cleanup Review

## P0: The `list[list[list]]` type plague

### Problem

The type `list[list[list]]` (shards of chunks of items) appears throughout execution.py with no type aliases, making the code nearly unreadable. Worse, the same variable `shards` silently changes type between `list[list[list]]` (in-memory data) and `list[list[ChunkRef]]` (disk references), and `isinstance` checks are used to distinguish them at runtime:

```python
# Line 597-598: runtime type sniffing on nested lists
if shards and isinstance(shards[0], list) and shards[0] and isinstance(shards[0][0], ChunkRef):
```

This pattern appears three times and is both fragile and unreadable. The type system provides zero help because the variable is untyped or typed as the wrong thing.

### Proposed solution

Introduce named wrapper types that make the shard/chunk/item hierarchy explicit and distinguish in-memory vs. on-disk data at the type level:

```python
@dataclass
class Chunk:
    """A list of items that form one unit of processing."""
    items: list[Any]

@dataclass
class Shard:
    """An ordered sequence of chunks assigned to a single worker."""
    chunks: list[Chunk]

@dataclass
class ShardedData:
    """All shards of in-memory data."""
    shards: list[Shard]

@dataclass
class ChunkRef:
    """Reference to a chunk stored on disk."""
    path: str
    count: int

    def read(self) -> Chunk: ...
    def write(chunk: Chunk, path: str) -> ChunkRef: ...  # classmethod

@dataclass
class ShardRefs:
    """All shards as disk references."""
    shards: list[list[ChunkRef]]

    def load(self) -> ShardedData: ...
```

This eliminates every `isinstance` check because the type itself tells you whether data is in memory or on disk. Function signatures become self-documenting:

```python
def _reshard(data: ShardedData, num_shards: int) -> ShardedData: ...
def _write_shards_to_disk(data: ShardedData, ...) -> ShardRefs: ...
def _load_shards_from_refs(refs: ...) -> ShardRefs: ...
```

---

## P0: `_materialize()` loads everything into the coordinator

### Problem

At line 658-661, `execute()` loads all shard data from disk into memory on the coordinator, then flattens it into a single list. For large datasets this defeats the purpose of disk-based chunking -- the coordinator becomes the bottleneck:

```python
if shards and isinstance(shards[0], list) and shards[0] and isinstance(shards[0][0], ChunkRef):
    shards = _load_all_shard_data(shards)
return _materialize(shards)
```

The stated intent of `materialize()` is "make sure things run," not "load all results into the coordinator's memory."

### Proposed solution

Return `ShardRefs` (or an iterator/lazy view) from `execute()` instead of a flattened list. The caller decides whether to load:

```python
def execute(self, dataset: Dataset, ...) -> ShardRefs:
    """Execute pipeline. Returns references to output chunks on disk.

    Call .load() or iterate to access data. Chunks are cleaned up
    when the context manager exits (unless preserve_chunks=True).
    """
    ...
    return final_shard_refs

# For callers that truly need everything in memory:
refs = ctx.execute(dataset)
all_data = refs.load()
```

For backward compatibility with `Backend.execute()`, that shim can call `.load()` and flatten. But the primary API should not force materialization.

---

## P1: ChunkRef should own its serialization

### Problem

`ChunkRef` is a plain dataclass with no behavior. The `_write_chunk` and `_read_chunk` module-level functions operate on paths, not on `ChunkRef` instances. This means:

1. Creating a `ChunkRef` and reading/writing are disconnected operations -- easy to get wrong.
2. `_write_chunk` returns a `dict` (!) that the caller manually unpacks into a `ChunkRef`.
3. The atomic-write logic in `_write_chunk` is not reusable or testable in isolation.

```python
# Current: awkward dict -> ChunkRef dance
meta = _write_chunk(chunk_path, item)
chunk_ref = ChunkRef(path=meta["path"], count=meta["count"])
```

### Proposed solution

Make `ChunkRef` a proper type with classmethods for construction and an instance method for reading:

```python
@dataclass(frozen=True)
class ChunkRef:
    path: str
    count: int

    @classmethod
    def write(cls, path: str, data: list) -> ChunkRef:
        """Atomically write data to path and return a reference."""
        ensure_parent_dir(path)
        temp_path = f"{path}.tmp"
        fs = fsspec.core.url_to_fs(path)[0]
        try:
            with fsspec.open(temp_path, "wb") as f:
                pickle.dump(data, f)
            fs.mv(temp_path, path)
        except Exception:
            with suppress(Exception):
                if fs.exists(temp_path):
                    fs.rm(temp_path)
            raise
        return cls(path=path, count=len(data))

    def read(self) -> list:
        """Load chunk data from disk."""
        with fsspec.open(self.path, "rb") as f:
            return pickle.load(f)
```

This is a straightforward refactor: replace `_write_chunk(path, data)` with `ChunkRef.write(path, data)`, and `_read_chunk(ref.path)` with `ref.read()`.

---

## P1: `_load_shards_from_refs` reads data that should stay as refs

### Problem

After each stage completes (line 641-642), the coordinator calls `_load_shards_from_refs` which reads every chunk from disk into memory, only to potentially write them back to disk for the next stage. This is wasteful for multi-stage pipelines:

```python
result_refs = coordinator.collect_results.remote().result()
shards = _load_shards_from_refs(result_refs, len(shard_refs))  # loads ALL data
```

### Proposed solution

`_load_shards_from_refs` should return `list[list[ChunkRef]]` (regroup refs by output shard) without loading data. Only load when actually needed (reshard, materialize). Rename to `_regroup_result_refs`:

```python
def _regroup_result_refs(
    result_refs: dict[int, list[tuple[dict, ChunkRef]]],
    input_shard_count: int,
) -> list[list[ChunkRef]]:
    """Regroup worker output refs by output shard index without loading data."""
    output_by_shard: dict[int, list[ChunkRef]] = defaultdict(list)
    for _input_idx, result_pairs in result_refs.items():
        for header, chunk_ref in result_pairs:
            output_by_shard[header["shard_idx"]].append(chunk_ref)

    num_output = max(max(output_by_shard.keys(), default=0) + 1, input_shard_count)
    return [output_by_shard.get(idx, []) for idx in range(num_output)]
```

---

## P1: Duplicate poll-loop pattern

### Problem

The "start stage, poll until done" pattern is copy-pasted three times: once in `execute()` (line 627-638), once in `_compute_join_aux` (line 884-894), and the shutdown/wait pattern appears in two places. Each copy has subtle differences (e.g., join version uses `ctx._coordinator` with `# type: ignore`).

### Proposed solution

Extract a helper:

```python
def _run_stage_on_coordinator(
    coordinator: ActorHandle,
    stage_name: str,
    tasks: list[ShardTask],
) -> dict[int, list[tuple[dict, ChunkRef]]]:
    """Submit tasks, poll until complete, return raw results."""
    coordinator.start_stage.remote(stage_name, tasks).result()
    while True:
        coordinator.check_heartbeats.remote()
        status = coordinator.get_status.remote().result()
        if status.get("fatal_error"):
            raise ZephyrWorkerError(status["fatal_error"])
        if status["completed"] >= status["total"]:
            break
        time.sleep(0.1)
    return coordinator.collect_results.remote().result()
```

---

## P2: `_SerializableShard` interface is implicit

### Problem

`_SerializableShard` implements `__iter__` and `iter_chunks()` to match what `run_stage` expects, but there is no shared protocol or type that documents this contract. If `run_stage` changes its expectations, this breaks silently.

### Proposed solution

Define a `Shard` protocol in `plan.py`:

```python
class ShardProtocol(Protocol):
    def iter_chunks(self) -> Iterator[list]: ...
    def __iter__(self) -> Iterator: ...
```

Have `_SerializableShard` implement it, and type-hint `StageContext.shard` accordingly.

---

## P2: `report_result` stores `list[tuple[dict, ChunkRef]]` but types say `list`

### Problem

`report_result` is typed as accepting `result: list` and `_results` is `dict[int, list]`. The actual data flowing through is `list[tuple[dict, ChunkRef]]`. This makes it impossible to understand the data flow without reading every call site.

### Proposed solution

Type the coordinator properly:

```python
StageResult = tuple[dict, ChunkRef]  # (header_dict, chunk_ref)

def report_result(self, worker_id: str, shard_idx: int, result: list[StageResult]) -> None: ...
```

---

## Migration strategy

1. **Phase 1** (P0 items): Introduce `ShardRefs` / `ShardedData` types. Refactor `execute()` to keep data as refs between stages. Fix `materialize()` to return refs. This is the biggest change but can be done in one pass since all the code is in `execution.py`.

2. **Phase 2** (P1 items): Move read/write onto `ChunkRef`. Extract poll-loop helper. These are mechanical refactors.

3. **Phase 3** (P2 items): Add `ShardProtocol`, tighten coordinator typing. Low risk, can happen anytime.

All changes are internal to `execution.py` except the return type of `execute()`, which affects `Backend.execute()` and tests. `Backend.execute()` can absorb the change by calling `.load()` + flatten internally.
