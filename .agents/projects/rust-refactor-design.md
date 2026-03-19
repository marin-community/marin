# Rust Refactor and CI Wheel Build System — Design

## 1. Problem

All Rust code lives under `lib/dupekit/` alongside Python workspace members.
There is no Cargo workspace, no CI wheel builds, and every install (dev, Docker, cluster)
compiles from source (~60-90s). External contributors and Docker builds need a full Rust
toolchain even though dupekit uses abi3 stable ABI and could ship a single wheel per platform.

Key files:
- `lib/dupekit/Cargo.toml:1` — standalone crate, no workspace
- `lib/dupekit/pyproject.toml:1` — maturin build backend
- `pyproject.toml:53` — `dupekit = { path = "lib/dupekit", editable = true }` under `[tool.uv.sources]`
- `docker/marin/Dockerfile.cluster:14` — `FROM rust:1.91-slim AS rust-builder` stage
- `docker/marin/Dockerfile.vllm:14` — same rust-builder stage
- `.github/workflows/dupekit-unit-tests.yaml` — runs pytest only, no wheel build

## 2. Proposed Solution

### 2.1 Directory restructure

Move `lib/dupekit/` → `rust/dupekit/`. Add a Cargo workspace at `rust/Cargo.toml`.
The Python package directory (`dupekit/` containing `__init__.py`, `__init__.pyi`)
stays **inside** the crate directory as `rust/dupekit/dupekit/`, exactly as it is today.
Maturin's `python-source = "."` setting keeps working.

New layout:
```
rust/
├── Cargo.toml          # workspace root
└── dupekit/
    ├── Cargo.toml      # crate manifest (unchanged except workspace ref)
    ├── Cargo.lock
    ├── pyproject.toml   # maturin config (unchanged)
    ├── src/             # Rust sources
    │   ├── lib.rs
    │   ├── bloom.rs
    │   ├── hashing.rs
    │   ├── minhash_ops.rs
    │   ├── ops.rs
    │   ├── pipeline.rs
    │   └── marshaling.rs
    ├── dupekit/         # Python package
    │   ├── __init__.py
    │   └── __init__.pyi
    ├── tests/           # Python tests
    └── README.md
```

### 2.2 Dev-mode vs user-mode switch

**Recommended approach: Option A — gitignored override in `[tool.uv.sources]`.**

The simplest mechanism that works with uv:

1. The checked-in `pyproject.toml:53` changes to point at the new location:
   ```toml
   dupekit = { path = "rust/dupekit", editable = true }
   ```

2. A dev who wants to build from source runs `uv sync` — uv invokes maturin on `rust/dupekit/`.

3. For "user mode" (pre-built wheel, no Rust needed), the developer creates a gitignored
   override file `uv.override.toml` at the repo root:
   ```toml
   [tool.uv.sources]
   dupekit = { index = "dupekit-wheels" }
   ```
   And in the root `pyproject.toml`, add a secondary index pointing to PyPI (or a GitHub
   Releases-based simple index):
   ```toml
   [[tool.uv.index]]
   name = "dupekit-wheels"
   url = "https://pypi.org/simple"
   ```

**However**, uv does not support `uv.override.toml` natively. The actual mechanism is simpler:

**Revised approach: script-based mode switch.**

A small shell script `scripts/rust-mode.sh` toggles between dev and user mode by
rewriting one line in the root `pyproject.toml` (the `dupekit` source entry). The file
`.rust-dev-mode` (gitignored) acts as a flag.

```bash
#!/usr/bin/env bash
# scripts/rust-mode.sh — toggle dupekit between source and wheel install
set -euo pipefail
MODE="${1:-status}"
ROOT="$(git rev-parse --show-toplevel)"
PYPROJECT="$ROOT/pyproject.toml"

case "$MODE" in
  dev)
    # Source build from rust/dupekit/
    sed -i 's|^dupekit = .*|dupekit = { path = "rust/dupekit", editable = true }|' "$PYPROJECT"
    touch "$ROOT/.rust-dev-mode"
    echo "Switched to dev mode (source build). Run: uv sync"
    ;;
  user)
    # Pre-built wheel from PyPI
    sed -i 's|^dupekit = .*|dupekit = { version = ">=0.1.0" }|' "$PYPROJECT"
    rm -f "$ROOT/.rust-dev-mode"
    echo "Switched to user mode (pre-built wheel). Run: uv sync"
    ;;
  status)
    if [ -f "$ROOT/.rust-dev-mode" ]; then echo "dev"; else echo "user"; fi
    ;;
esac
```

Add to `.gitignore`:
```
.rust-dev-mode
```

**Why this approach over the others:**
- **B (wrapper package)**: Adds a whole extra package for a one-line config change. Overkill.
- **C (env var + custom backend)**: Maturin doesn't support this. Would need a custom PEP 517 wrapper.
- **D (two dep groups)**: uv doesn't support conditional source resolution per dep group.
- **A (script)**: Simple, explicit, no magic. One script, one gitignored flag. Developers
  default to dev mode (the `pyproject.toml` is checked in with the source path).

The default checked-in state is **dev mode** (`path = "rust/dupekit"`), because all active
contributors have Rust installed. User mode is opt-in for CI/Docker/external use.

### 2.3 CI workflow for wheel builds

New workflow: `.github/workflows/dupekit-wheels.yaml`

```yaml
name: Dupekit - Build Wheels

on:
  push:
    branches: [main]
    paths:
      - "rust/dupekit/**"
      - ".github/workflows/dupekit-wheels.yaml"
    tags:
      - "dupekit-v*"
  pull_request:
    paths:
      - "rust/dupekit/**"
      - ".github/workflows/dupekit-wheels.yaml"

concurrency:
  group: ${{ github.workflow }}-${{ github.event.pull_request.number || github.ref }}
  cancel-in-progress: true

permissions:
  contents: write  # for creating releases

jobs:
  build:
    strategy:
      matrix:
        include:
          - os: ubuntu-latest
            target: x86_64
          - os: ubuntu-latest
            target: aarch64
          - os: macos-13
            target: x86_64
          - os: macos-14
            target: aarch64
    runs-on: ${{ matrix.os }}
    steps:
      - uses: actions/checkout@v4

      - uses: PyO3/maturin-action@v1
        with:
          command: build
          args: --release --out dist --manifest-path rust/dupekit/Cargo.toml
          target: ${{ matrix.target }}
          manylinux: 2_28

      - uses: actions/upload-artifact@v4
        with:
          name: wheels-${{ matrix.os }}-${{ matrix.target }}
          path: dist/*.whl

  # Build sdist for source fallback
  sdist:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: PyO3/maturin-action@v1
        with:
          command: sdist
          args: --out dist --manifest-path rust/dupekit/Cargo.toml
      - uses: actions/upload-artifact@v4
        with:
          name: sdist
          path: dist/*.tar.gz

  # Publish to PyPI on tag push
  publish:
    if: startsWith(github.ref, 'refs/tags/dupekit-v')
    needs: [build, sdist]
    runs-on: ubuntu-latest
    environment: pypi
    permissions:
      id-token: write  # trusted publishing
    steps:
      - uses: actions/download-artifact@v4
        with:
          path: dist
          merge-multiple: true
      - uses: pypa/gh-action-pypi-publish@release/v1
        with:
          packages-dir: dist/

  # Attach wheels to GitHub Release (always on tag)
  release:
    if: startsWith(github.ref, 'refs/tags/dupekit-v')
    needs: [build, sdist]
    runs-on: ubuntu-latest
    steps:
      - uses: actions/download-artifact@v4
        with:
          path: dist
          merge-multiple: true
      - uses: softprops/action-gh-release@v2
        with:
          files: dist/*
```

**Wheel destination**: Publish to PyPI under `dupekit`. This is the standard approach and
lets anyone `pip install dupekit` without Rust. GitHub Releases are a secondary mirror.
On PRs, wheels are build-tested but not published (just uploaded as artifacts for verification).

### 2.4 pyproject.toml integration

#### Root `pyproject.toml` changes

```toml
# Line 53 — update path
dupekit = { path = "rust/dupekit", editable = true }
```

No other root pyproject.toml changes needed. The `[tool.uv.workspace]` members list
does not include dupekit (it's already excluded — it's an optional non-workspace dep).

When in user mode (script switches it):
```toml
dupekit = { version = ">=0.1.0" }
```

#### `lib/marin/pyproject.toml:180-182` — no changes needed

```toml
dedup = [
    "dupekit",
]
```

This already just names the package. uv resolves it via `[tool.uv.sources]` in the root.

#### `rust/dupekit/pyproject.toml` changes

Update the `cache-keys` glob paths (they're relative to the package dir, so unchanged),
and update the error message in `__init__.py` to reference the new path:

```python
# rust/dupekit/dupekit/__init__.py line 25
_INSTALL_MSG = (
    "dupekit native extension is not installed. "
    "Build it with: cd rust/dupekit && uv sync && maturin develop --release"
)
```

### 2.5 Docker changes

#### Current state (`docker/marin/Dockerfile.cluster:13-64,89-101`)

The Dockerfile copies the entire Rust toolchain (~800MB) into the final image and builds
dupekit from source.

#### New approach: pre-built wheel in Docker

```dockerfile
# Remove the rust-builder stage entirely (lines 13-14):
# - FROM rust:1.91-slim AS rust-builder

# Remove Rust toolchain copy (lines 59-64):
# - COPY --from=rust-builder ...
# - ENV PATH=... RUSTUP_HOME=... CARGO_HOME=...

# Replace source build (line 98) with wheel install:
# Before:
#   && uv pip install -e lib/dupekit --system \
# After:
#   && uv pip install dupekit --system \
```

This works because:
1. CI publishes wheels to PyPI on tagged releases
2. `uv pip install dupekit` fetches the pre-built abi3 wheel
3. No Rust toolchain needed in the Docker image → smaller image, faster builds

**Transition period**: Until the first wheel is published to PyPI, keep the Rust builder
stage but update the path to `rust/dupekit/`:
```dockerfile
  && uv pip install -e rust/dupekit --system \
```

Apply the same changes to `docker/marin/Dockerfile.vllm`.

### 2.6 Cargo workspace

New file: `rust/Cargo.toml`

```toml
[workspace]
members = ["dupekit"]
resolver = "2"

[profile.release]
lto = "fat"
codegen-units = 1
strip = "symbols"
```

Update `rust/dupekit/Cargo.toml` to declare workspace membership:

```toml
[package]
name = "dupekit"
version = "0.1.0"
edition = "2021"

# No change needed — being listed in workspace.members is sufficient.
# The Cargo.lock stays in rust/dupekit/ for maturin compatibility.
```

**Note on Cargo.lock placement**: maturin expects `Cargo.lock` next to the crate's
`Cargo.toml`, not at the workspace root. Keep `rust/dupekit/Cargo.lock` where it is.
If maturin resolves this from the workspace root in the future, move it then.

Actually, with a Cargo workspace, `Cargo.lock` should be at the workspace root
(`rust/Cargo.lock`). Maturin respects this — it walks up to find the workspace root.
Move `lib/dupekit/Cargo.lock` → `rust/Cargo.lock`.

The `[profile.release]` settings enable:
- `lto = "fat"` — whole-program link-time optimization for smaller/faster binaries
- `codegen-units = 1` — better optimization at cost of compile time (fine for CI)
- `strip = "symbols"` — smaller wheel size

### 2.7 Existing CI workflow update

`.github/workflows/dupekit-unit-tests.yaml` needs path updates:

```yaml
on:
  push:
    branches:
      - main
  pull_request:
    paths:
      - rust/dupekit/**                           # was: lib/dupekit/**
      - .github/workflows/dupekit-unit-tests.yaml

# ...
      - name: Test dupekit
        run: |
          cd rust/dupekit && uv run --frozen --group test pytest tests/ -v
```

Also add Rust-level CI (cargo test, clippy, fmt) — either in the same workflow or a
new one. Recommend adding to the existing workflow:

```yaml
  rust-checks:
    runs-on: ubuntu-latest
    timeout-minutes: 10
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
        with:
          components: clippy, rustfmt
      - name: cargo fmt
        run: cargo fmt --manifest-path rust/dupekit/Cargo.toml -- --check
      - name: cargo clippy
        run: cargo clippy --manifest-path rust/dupekit/Cargo.toml -- -D warnings
      - name: cargo test
        run: cargo test --manifest-path rust/dupekit/Cargo.toml
```

## 3. Implementation Plan

### Step 1: Directory move and Cargo workspace (no parallelism deps)

**Files to create/modify:**
- `git mv lib/dupekit rust/dupekit` (the entire directory)
- Create `rust/Cargo.toml` (workspace root, contents in §2.6)
- Move `rust/dupekit/Cargo.lock` → `rust/Cargo.lock`
- Verify `rust/dupekit/Cargo.toml` works within workspace: `cd rust && cargo check`

**Tests:** `cd rust/dupekit && cargo test` (Rust-level), `cd rust/dupekit && uv run --group test pytest tests/ -v` (Python-level)

### Step 2: Update all references to `lib/dupekit` → `rust/dupekit`

**Files to modify:**
- `pyproject.toml:53` — change `path = "lib/dupekit"` → `path = "rust/dupekit"`
- `docker/marin/Dockerfile.cluster:90,98` — update `COPY` and `uv pip install` paths
- `docker/marin/Dockerfile.vllm:90,98` — same changes
- `.github/workflows/dupekit-unit-tests.yaml:8,39` — update paths
- `rust/dupekit/dupekit/__init__.py:25` — update error message path

**Depends on:** Step 1

**Tests:** `uv sync` from repo root succeeds; `uv run python -c "import dupekit"` works

### Step 3: Add dev-mode/user-mode switch

**Files to create/modify:**
- Create `scripts/rust-mode.sh` (contents in §2.2)
- Add `.rust-dev-mode` to `.gitignore`
- `chmod +x scripts/rust-mode.sh`

**Depends on:** Step 2

**Tests:** Run `scripts/rust-mode.sh dev && uv sync` and `scripts/rust-mode.sh user` (user mode only testable after PyPI publish). Verify `scripts/rust-mode.sh status` returns correct mode.

### Step 4: CI wheel build workflow

**Files to create:**
- `.github/workflows/dupekit-wheels.yaml` (contents in §2.3)

**Depends on:** Step 1 (needs `rust/dupekit/` path)

**Tests:** Push to a branch, verify the workflow runs and produces wheel artifacts for all 4 platform targets. Download a wheel artifact and verify `pip install <wheel>` works.

### Step 5: Add Rust checks to existing CI

**Files to modify:**
- `.github/workflows/dupekit-unit-tests.yaml` — add `rust-checks` job (contents in §2.7)

**Depends on:** Step 2 (needs updated paths)

**Tests:** Push to a branch, verify cargo fmt/clippy/test pass in CI.

### Step 6: Docker optimization (post-PyPI publish)

**Files to modify:**
- `docker/marin/Dockerfile.cluster` — remove rust-builder stage, use `uv pip install dupekit`
- `docker/marin/Dockerfile.vllm` — same

**Depends on:** Step 4 (wheels must be published to PyPI first). This step should be
deferred until after the first `dupekit-v*` tag is pushed and wheels are on PyPI.

**Tests:** Build Docker image locally, verify `python -c "import dupekit; print(dupekit.Bloom)"` inside container.

### Parallelism

```
Step 1 ──→ Step 2 ──→ Step 3
              │
              └──→ Step 5
Step 1 ──→ Step 4
Step 4 ──→ Step 6 (deferred)
```

- Steps 1→2 are sequential (2 needs the moved files).
- Steps 3, 4, 5 can all run in parallel after their respective dependencies.
- Step 6 is deferred until PyPI wheels exist.

Steps 1+2 could be a single commit/PR. Steps 3, 4, 5 are independent and can be
separate PRs that land in any order after 1+2 merge.

## 4. Risks and Open Questions

1. **Cargo.lock location with workspace**: maturin may expect `Cargo.lock` at the crate
   level, not workspace root. Need to test. If maturin complains, keep `Cargo.lock` in
   `rust/dupekit/` and don't create a workspace-level lock.

2. **PyPI name `dupekit`**: Need to verify the name is available on PyPI. If taken,
   use `marin-dupekit` and update the `[project] name` in `rust/dupekit/pyproject.toml`.

3. **maturin `--manifest-path` with workspace**: The `maturin-action` `--manifest-path`
   flag should point to the crate's `Cargo.toml`, not the workspace root. Verify this
   works with the workspace layout.

4. **uv cache-keys after move**: The cache-keys in `rust/dupekit/pyproject.toml` use
   relative paths (`Cargo.toml`, `src/**/*.rs`). These should still work since they're
   relative to the package directory. But verify `uv sync` correctly detects Rust source
   changes and triggers rebuilds.

5. **Docker transition**: Between merging the directory move and publishing the first
   wheel to PyPI, Docker builds must still use source compilation. The intermediate state
   (Step 2) handles this by updating the path. Step 6 is explicitly deferred.

6. **Cross-compilation for aarch64**: The `maturin-action` supports cross-compilation
   for `linux-aarch64` via QEMU on `ubuntu-latest`. This is slow (~10-15 min) but works.
   If build times are problematic, consider `zig` cross-compilation linker or native
   ARM runners.

7. **Trusted publishing on PyPI**: Step 4's publish job uses PyPI trusted publishing
   (OIDC). This requires configuring a "trusted publisher" on pypi.org/manage for the
   `dupekit` project, pointing to the GitHub repo and workflow file. Someone with PyPI
   access needs to do this before the first publish.
