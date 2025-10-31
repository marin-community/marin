# Apache Beam Migration - Phase 0 Implementation Summary

**Date:** 2025-10-30
**Status:** ✅ Completed
**Phase:** Phase 0 - Core Library & First Migration

---

## Executive Summary

Successfully implemented the `ml-flow` library and migrated the FastText transform as a proof-of-concept for Beam-based data preprocessing in Marin. The implementation provides a clean Dataset API that wraps Apache Beam while maintaining compatibility with existing Marin patterns.

---

## Deliverables

### 1. ml-flow Library (`lib/ml-flow/`)

Created a standalone Python package providing high-level abstractions over Apache Beam:

#### **Core Components:**

- **`Dataset` API** (`dataset.py`): Fluent interface for data transformations
  - `from_text_files()` / `from_jsonl_files()` - Read data with compression support
  - `.map()`, `.flat_map()`, `.filter()` - Standard transformations
  - `.write_jsonl_gz()`, `.write_text_files()` - Write with automatic sharding
  - `.run_and_wait()` - Execute and wait for completion

- **`DataflowOptions`** (`options.py`): Typed configuration for Dataflow
  - FlexRS support for 40-60% cost savings
  - Auto-scaling with throughput-based algorithm
  - Sensible defaults for Marin workloads

- **I/O Transforms** (`io.py`): Custom Beam sources/sinks
  - `ReadJsonLines` - Read JSONL/JSONL.gz with fsspec
  - `ReadTextLines` - Read text files with compression
  - `WriteJsonLines` - Write JSONL.gz with automatic serialization

- **Testing Utilities** (`testing.py`): DirectRunner helpers for local testing

#### **Package Structure:**
```
lib/ml-flow/
├── pyproject.toml              # Package definition with apache-beam[gcp]>=2.69.0rc1
├── README.md                   # Usage guide and examples
├── src/ml_flow/
│   ├── __init__.py             # Public API exports
│   ├── dataset.py              # Core Dataset API (290 lines)
│   ├── io.py                   # File I/O transforms (140 lines)
│   ├── options.py              # DataflowOptions config (75 lines)
│   └── testing.py              # Test utilities (30 lines)
└── tests/
    ├── test_dataset.py         # 6 tests covering Dataset operations
    └── test_options.py         # 6 tests covering configuration
```

#### **Test Results:**
- **12/12 tests passing** with DirectRunner (Python 3.11)
- Covers: read/write, map/filter/flat_map, chaining, compression, options

---

### 2. FastText Transform Migration

Created Beam version of FastText to Dolma conversion as reference implementation:

**File:** `src/marin/transform/fasttext/transform_beam.py`

#### **Key Changes from Ray Version:**

| **Aspect** | **Ray (Original)** | **Beam (New)** |
|------------|-------------------|----------------|
| **Orchestration** | Manual `@ray.remote` + `ray.get()` | Automatic via Dataset API |
| **Resumption** | `@cached_or_construct_output` | Built into write transforms |
| **Parallelization** | Fixed resources | Auto-scaling (1-1000 workers) |
| **Cost** | Fixed cluster cost | FlexRS: 40-60% savings |
| **Lines of Code** | 129 lines | 185 lines (+43%) |
| **Complexity** | Manual coordination | Declarative pipeline |

#### **Implementation Highlights:**

```python
# Clean dataset-oriented approach
ds = Dataset.from_text_files(input_path, options)
ds = ds.map(add_line_numbers)
ds = ds.map(parse_fasttext_line)
ds = ds.filter(lambda x: x is not None)
ds.write_jsonl_gz(output_prefix)
ds.run_and_wait()
```

**Removed:**
- `@ray.remote` decorator
- `@cached_or_construct_output` decorator
- Manual `ray.get()` calls
- Resource specifications (handled by Dataflow)

**Gained:**
- Automatic scaling
- Cost optimization via FlexRS
- Built-in fault tolerance
- Declarative pipeline definition

---

## Technical Decisions

### 1. Apache Beam Version

**Decision:** Use `apache-beam[gcp]>=2.69.0rc1` (pre-release)

**Rationale:**
- Resolves protobuf version conflict with Levanter (requires protobuf>=6)
- Stable versions require protobuf<6, incompatible with Marin's training dependencies
- Pre-release 2.69.0rc1 supports protobuf>=6

**Trade-off:** Using pre-release version, but necessary for compatibility

---

### 2. Dependency Management

**Decision:** Make `ml-flow` an optional dependency (`beam` extra) with conflict markers

**Configuration in `pyproject.toml`:**
```toml
[project.optional-dependencies]
beam = ["ml-flow"]

[tool.uv]
conflicts = [
    [{ extra = "beam" }, { extra = "tpu" }],
    [{ extra = "beam" }, { extra = "tokenize_train" }],
]
```

**Rationale:**
- Beam and Levanter have protobuf conflicts
- Beam is for preprocessing, Levanter for training
- They don't need to be installed together
- Use separate environments or conflicting extras

---

### 3. Dataset-Oriented Design

**Decision:** Write datasets via `.write_jsonl_gz()` instead of processing directly in map functions

**Original Proposal:**
```python
FileProcessor.process_directory(
    input_pattern,
    output_path,
    process_fn=lambda in, out: write_file(in, out),  # Writes directly
)
```

**Implemented Design:**
```python
ds = Dataset.from_files(pattern)
ds = ds.map(transform)
ds.write_jsonl_gz(output_path)  # Writes dataset
ds.run_and_wait()
```

**Rationale:**
- More composable (can chain transformations)
- Separates concerns (transform vs. I/O)
- Follows Beam/Spark idioms
- Easier to test and reason about

---

### 4. Type Annotations

**Decision:** Use `TYPE_CHECKING` and string literals for Beam types

**Implementation:**
```python
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from apache_beam.runners.runner import PipelineResult

def run(self) -> "PipelineResult":
    return self._pipeline.run()
```

**Rationale:**
- `PipelineResult` not directly importable from `apache_beam`
- Avoids runtime import errors
- Maintains type safety for IDEs/mypy

---

## Testing Strategy

### Unit Tests (DirectRunner)
- **12 tests** covering core functionality
- Run locally with no cloud dependencies
- Fast execution (~3 seconds total)

### Workarounds Applied

1. **Compression Type:** Convert string to `CompressionTypes` enum
   ```python
   compression_map = {
       "none": CompressionTypes.UNCOMPRESSED,
       "gzip": CompressionTypes.GZIP,
   }
   ```

2. **Num Shards Bug:** Beam 2.69 fails with `None`, use `0` instead
   ```python
   shards = 0 if num_shards is None else num_shards
   ```

3. **Pipeline State:** Handle both string and enum representations
   ```python
   state = result.state if isinstance(result.state, str) else result.state.name
   ```

---

## Known Limitations

### 1. Dependency Conflicts
- **Issue:** Cannot install `beam` and `tpu`/`tokenize_train` extras together
- **Impact:** Need separate environments for preprocessing vs. training
- **Mitigation:** Use conflict markers, separate virtual environments, or Docker

### 2. Pre-release Apache Beam
- **Issue:** Using 2.69.0rc1 (not stable release)
- **Impact:** Potential bugs, API changes before stable
- **Mitigation:** Lock to specific RC version, test thoroughly, monitor for stable release

### 3. Python Version
- **Issue:** Tests initially failed with Python 3.13
- **Resolution:** Works with Python 3.11 (Marin's standard)
- **Note:** Likely fsspec/beam compatibility issue with 3.13

---

## Performance & Cost Analysis

### Expected Benefits (from migration plan):

| **Metric** | **Ray (Current)** | **Beam (Projected)** | **Improvement** |
|------------|-------------------|----------------------|-----------------|
| **Cost** | Fixed cluster (24h) | FlexRS auto-scaling | **40-60% savings** |
| **Scaling** | Manual (head node bottleneck) | Automatic (1-1000 workers) | **Better throughput** |
| **Fault Tolerance** | Manual retries | Built-in retries + spot VMs | **Better resilience** |
| **Code Complexity** | Manual orchestration | Declarative pipeline | **Simpler** |

### Example Cost Comparison (10TB workload):
- **Ray:** $1,824 (100 workers × 24h × $0.76/h)
- **Beam Standard:** $456 (50 workers × 12h × $0.76/h) = **-75%**
- **Beam FlexRS:** $368 (50 workers × 16h × $0.46/h) = **-80%**

---

## Next Steps (Phase 1)

### Immediate (Weeks 3-8):
1. **Migrate simple transforms** (5 conversation/legal transforms)
2. **Validate output parity** with Ray versions
3. **Run production workload** on small dataset with Dataflow
4. **Measure actual cost savings**

### Medium-term (Weeks 9-17):
1. **Migrate crawl transforms** (WARC processing, dedup)
2. **Migrate classification inference** (replace AutoscalingActorPool)
3. **Document migration patterns**

### Long-term (Weeks 18-22):
1. **End-to-end pipeline testing**
2. **Cost optimization tuning**
3. **Team training**

---

## Files Changed

### New Files:
```
lib/ml-flow/                               # New standalone library
├── pyproject.toml
├── README.md
├── src/ml_flow/
│   ├── __init__.py
│   ├── dataset.py
│   ├── io.py
│   ├── options.py
│   └── testing.py
└── tests/
    ├── test_dataset.py
    └── test_options.py

src/marin/transform/fasttext/
└── transform_beam.py                      # New Beam version

.agents/docs/
└── beam-implementation-phase0-summary.md  # This document
```

### Modified Files:
```
pyproject.toml                             # Added beam optional dependency
```

---

## Validation

### ✅ Completed:
- [x] ml-flow library implemented
- [x] 12/12 unit tests passing
- [x] FastText transform migrated
- [x] Documentation written

### 🔄 In Progress:
- [ ] Run FastText Beam version on actual data
- [ ] Validate output parity with Ray version
- [ ] Measure actual Dataflow costs

### 📋 Next:
- [ ] Migrate 5 more simple transforms
- [ ] Create migration recipe document
- [ ] Establish CI/CD for Beam tests

---

## Conclusion

Phase 0 is **complete** with a working `ml-flow` library and reference FastText migration. The implementation demonstrates:

1. **Clean API:** Dataset-oriented design is intuitive and composable
2. **Testing:** DirectRunner enables fast local development
3. **Compatibility:** Works with existing Marin patterns (fsspec, draccus, Dolma format)
4. **Foundation:** Ready to scale to remaining transforms

**Recommendation:** Proceed with Phase 1 (simple transform migrations) to validate cost savings and establish patterns before scaling to complex transforms.

---

## Resources

- **Migration Plan:** `.agents/docs/beam-migration-plan.md`
- **Codebase Outline:** `.agents/docs/transform-outline.md`
- **ml-flow README:** `lib/ml-flow/README.md`
- **Apache Beam Docs:** https://beam.apache.org/documentation/
- **Dataflow Docs:** https://cloud.google.com/dataflow/docs

---

**Author:** Claude Code
**Reviewer:** [Pending]
**Status:** ✅ Ready for Phase 1
