# Rust Code Inventory and Build Practices Research

## 1. Current Rust Code Inventory

### dupekit (`lib/dupekit/`)

**Purpose**: Optimized Rust code for text deduplication — bloom filters, hashing (blake2, blake3, xxh3), MinHash signatures, LSH bucketing, and a composable Arrow-based transformation pipeline.

**Rust source files** (`lib/dupekit/src/`):
- `lib.rs` — PyO3 module definition, exports all classes/functions
- `bloom.rs` — Bloom filter (set operations, persistence, union/intersection)
- `hashing.rs` — blake2, blake3, xxh3_64, xxh3_128 hash functions
- `minhash_ops.rs` — MinHash signature computation
- `ops.rs` — misc operations
- `pipeline.rs` — Composable `Transformation` pipeline over Arrow RecordBatches
- `marshaling.rs` — Benchmarking different Python↔Rust data marshaling strategies

**Cargo.toml** (`lib/dupekit/Cargo.toml`):
- Crate name: `dupekit`, lib name: `_native`, crate-type: `cdylib`
- Key deps: `pyo3 0.26` (abi3-py311 stable ABI), `arrow 57.1` (pyarrow feature), `parquet 57.1`, `blake2`, `blake3`, `xxhash-rust`, `regex`

**Build system**: Maturin (PEP 517 backend)
- `lib/dupekit/pyproject.toml`: `build-backend = "maturin"`, `requires = ["maturin>=1.5,<2.0"]`
- `module-name = "dupekit._native"` — compiled `.so` placed inside `dupekit/` Python package
- `python-source = "."` — Python source alongside Cargo.toml
- `[tool.uv].cache-keys` watches `Cargo.toml`, `Cargo.lock`, `src/**/*.rs`, `dupekit/*.pyi`

**Python interface**:
- `dupekit/__init__.py` — tries `from dupekit._native import *`; on `ImportError`, installs a stub module that raises `ImportError` with build instructions on any attribute access
- `dupekit/__init__.pyi` — full type stubs for all exported classes/functions

**Consumed by**:
- `lib/marin/src/marin/processing/classification/deduplication/exact.py`
- `lib/marin/src/marin/processing/classification/deduplication/fuzzy.py`
- `lib/marin/src/marin/processing/classification/deduplication/connected_components.py`
- `lib/marin/src/marin/processing/classification/decon.py`
- All dupekit tests in `lib/dupekit/tests/`

**Workspace status**: dupekit is **not** a uv workspace member. It is an optional dependency:
- Root `pyproject.toml:53`: `dupekit = { path = "lib/dupekit", editable = true }` under `[tool.uv.sources]`
- Root `pyproject.toml:51`: comment: "dupekit and harbor are optional dependencies (not workspace members)"
- `lib/marin/pyproject.toml`: listed under `[project.optional-dependencies] dedup = ["dupekit"]`

### Iris Dockerfile Rust

The Iris Dockerfile (`lib/iris/Dockerfile:134`) installs Rust via `rustup` — this is for building Python dependencies that require Rust at compile time (e.g., `cryptography`), not for any Iris-specific Rust code.

### Docker images (cluster/vllm)

Both `docker/marin/Dockerfile.cluster` and `docker/marin/Dockerfile.vllm`:
- Multi-stage build: `rust:1.91-slim` as `rust-builder` stage
- Copies full Rust toolchain into final image
- Builds dupekit via `uv pip install -e lib/dupekit --system`

## 2. Current Build System Integration

### How Rust is built today

1. **Dev mode**: `cd lib/dupekit && maturin develop --release` or simply `uv sync` from workspace root (uv detects maturin backend and invokes it)
2. **Docker/cluster**: `uv pip install -e lib/dupekit --system` in Dockerfile — uv invokes maturin which invokes cargo
3. **No CI workflows**: Zero GitHub Actions workflows build or test Rust code. No `.github/workflows/` file references dupekit, cargo, maturin, or Rust.

### Key configuration files

| File | Role |
|------|------|
| `lib/dupekit/Cargo.toml` | Rust crate manifest |
| `lib/dupekit/Cargo.lock` | Pinned Rust dependencies |
| `lib/dupekit/pyproject.toml` | Maturin build config + Python metadata |
| `lib/dupekit/dupekit/__init__.py` | Fallback stub when native ext missing |
| `lib/dupekit/dupekit/__init__.pyi` | Type stubs |
| `.gitignore` | Ignores `*.so` and `target/` |

### Stable ABI (abi3)

dupekit uses `pyo3/abi3-py311` — one compiled `.so` works across Python 3.11+. This is a major advantage for wheel distribution: only one wheel per platform (not per Python version).

## 3. Best Practices from Major Projects

### Project comparison

| Project | Build backend | Rust binding | Wheel strategy | Optional Rust? |
|---------|--------------|--------------|----------------|----------------|
| **polars** | maturin | PyO3 | maturin-action CI, PyPI wheels | No — Rust required |
| **pydantic-core** | maturin | PyO3 | maturin-action CI, PyPI wheels | No |
| **ruff** | maturin | standalone binary | maturin-action CI, PyPI wheels | N/A (pure Rust binary) |
| **tiktoken** | setuptools-rust | PyO3 | cibuildwheel CI, PyPI wheels | No — but pre-built wheels cover most platforms |
| **cryptography** | maturin (since ~v42) | PyO3/cffi | cibuildwheel CI, PyPI wheels | No — Rust required at build time, not runtime |

### Standard pattern: dev mode vs user mode

**Dev mode** (building from source):
```bash
# maturin develop builds .so in-place, linked to current Python
maturin develop --release
# OR: pip/uv install -e . triggers maturin as PEP 517 backend
uv pip install -e .
```

**User mode** (pre-built wheels):
```bash
# Users install from PyPI — no Rust compiler needed
pip install dupekit
```

### How projects publish platform-specific wheels

The standard pipeline:

1. **CI builds**: GitHub Actions matrix across `{linux-x86_64, linux-aarch64, macos-x86_64, macos-aarch64, windows-x86_64}`
2. **maturin-action** or **cibuildwheel**: builds wheels in manylinux containers (Linux) or native runners (macOS/Windows)
3. **abi3**: single wheel per platform when using stable ABI
4. **Publish**: upload to PyPI on tagged releases, or attach to GitHub Releases

Typical CI workflow (maturin-action):
```yaml
jobs:
  build:
    strategy:
      matrix:
        os: [ubuntu-latest, macos-13, macos-14, windows-latest]
    runs-on: ${{ matrix.os }}
    steps:
      - uses: actions/checkout@v4
      - uses: PyO3/maturin-action@v1
        with:
          command: build
          args: --release --out dist
          manylinux: 2_28
      - uses: actions/upload-artifact@v4
        with:
          name: wheels-${{ matrix.os }}
          path: dist/*.whl

  publish:
    needs: build
    runs-on: ubuntu-latest
    steps:
      - uses: actions/download-artifact@v4
      - uses: PyO3/maturin-action@v1
        with:
          command: upload
          args: --non-interactive --skip-existing dist/*
```

### The "optional Rust" pattern

dupekit already implements this pattern well via `dupekit/__init__.py`:
- Tries to import `_native`
- On `ImportError`, replaces the module with a stub that raises on use
- This means `import dupekit` succeeds even without the compiled extension

This is similar to how `cryptography` handles it — Rust is needed at build time but the resulting `.so` is self-contained. The difference is that without pre-built wheels, users must have Rust installed.

Maturin does **not** have built-in support for a `MATURIN_NO_RUST` env var or pure-Python fallback mode (see [maturin#1839](https://github.com/PyO3/maturin/issues/1839)). The standard approach is:
1. Publish pre-built wheels for common platforms
2. Fall back to source build (requires Rust) for exotic platforms
3. Optionally provide a pure-Python package under a different name

## 4. Existing Tooling

| Tool | Purpose | Used by |
|------|---------|---------|
| **maturin** | PEP 517 build backend for Rust+Python | polars, pydantic-core, cryptography, **dupekit** |
| **setuptools-rust** | setuptools plugin for Rust extensions | tiktoken (legacy pattern) |
| **PyO3/maturin-action** | GitHub Action for building/publishing maturin wheels | polars, pydantic-core |
| **cibuildwheel** | Generic multi-platform wheel builder | tiktoken, cryptography |
| **abi3** (stable ABI) | One wheel per platform across Python versions | **dupekit**, pydantic-core |

### maturin vs setuptools-rust

maturin is the modern choice. It:
- Requires less configuration
- Has built-in cross-compilation support
- Has `generate-ci` for scaffolding GitHub Actions
- Handles manylinux compliance automatically
- dupekit already uses maturin — no migration needed

### maturin generate-ci

```bash
maturin generate-ci github > .github/workflows/dupekit-wheels.yml
```

Generates a complete CI workflow with platform matrix, manylinux builds, and PyPI publish. Supports customization flags: `--pytest`, `--zig` (cross-compile), platform selection.

## 5. Mode Switching Patterns

### How the Python import path switches

dupekit's current `__init__.py` already handles this:

```python
try:
    from dupekit._native import *  # built .so present
except ImportError:
    # stub module that raises on use
    sys.modules[__name__] = _StubModule(__name__)
```

This is the standard pattern. The `.so` file is either:
- **Dev mode**: built in-place by `maturin develop` into `lib/dupekit/dupekit/_native.abi3.so`
- **Installed mode**: placed in `site-packages/dupekit/_native.abi3.so` by wheel install

### Gitignored artifacts

`.gitignore` already covers:
- `*.so` — compiled extensions
- `target/` — Rust build directory

### No env vars needed

The current pattern doesn't use env vars for mode switching — it relies purely on whether the `.so` exists at import time. This is the cleanest approach.

## 6. Gaps and Recommendations

### What's missing today

1. **No CI for Rust code**: No GitHub Actions workflow builds, tests, or lints dupekit. A `cargo fmt` or `cargo clippy` regression would go unnoticed.
2. **No pre-built wheels**: Every install (dev, Docker, cluster) builds from source. This is ~60-90s of compile time per install.
3. **No PyPI publishing**: dupekit is not published to PyPI. External users can't `pip install dupekit`.
4. **No Rust tests in CI**: The Rust unit tests (if any) are never run. Only the Python-level pytest tests exercise the code.

### What's already good

1. **abi3 stable ABI**: Only one wheel per platform needed.
2. **maturin build backend**: Modern, well-supported tooling.
3. **Graceful fallback**: `__init__.py` stub pattern handles missing extension cleanly.
4. **uv cache-keys**: Rebuilds triggered automatically when Rust sources change.
5. **Separate Cargo.lock**: Pinned Rust dependencies for reproducibility.

## Open Questions

1. Should dupekit be published to PyPI as a standalone package, or only consumed internally?
2. What platforms need wheels? (Linux x86_64 is the primary target for cluster use; macOS for dev)
3. Should we add `cargo test`, `cargo clippy`, and `cargo fmt --check` to a CI workflow?
4. Is there appetite for a Cargo workspace if more Rust crates are added in the future?
5. The Iris Dockerfile installs Rust just for build-time deps — could pre-built wheels for those deps eliminate that?
