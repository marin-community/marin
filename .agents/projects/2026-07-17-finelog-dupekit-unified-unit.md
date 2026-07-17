# Fold finelog + dupekit unit tests into unified-unit (issue #7254, Steps 2–3)

Status: plan v2, revised after codex + fable design reviews. Ready to implement
pending the two maintainer decisions in "Open decisions".
Branch: `weaver/issue-7254`. Step 1 (dupekit workspace split) shipped in PR #7268.

## Goal

Issue #7254 wants all Python unit tests to run through `unified-unit.yaml`.
Step 1 made `dupekit` a workspace member with a prebuilt native wheel (mirroring
`finelog`). This plan covers the remainder:

- **Step 2**: a shared Rust build setup both `finelog` and `dupekit` use so their
  CI test legs build the native extension from source when a PR changes Rust.
- **Step 3**: wire that into `unified-unit.yaml` through a generic per-leg setup
  hook, so `finelog`/`dupekit` join the unified matrix and future suites
  (`levanter-torch`, `levanter-tpu`, `iris-e2e-smoke`) can fold in the same way.
  (Migrating those three is explicitly out of scope here.)

## The wmoss constraint (the crux)

> If a PR changes Rust *and* Python that uses that Rust change, we can't run the
> Python CI against the prebuilt wheel — it fails, since the wheel lacks the new
> Rust. (This broke previously.)

Today `dupekit-unit.yaml` tests `marin-dupekit` against the prebuilt
`marin-dupekit-native` wheel, so a same-PR Rust change is invisible. finelog has
the same latent hazard and it has bitten before — `lib/finelog/tests/
test_embedded.py:81-86` documents a `finelog_server` wheel that predated a
`LogEntry.seq` field.

Resolution: when a PR touches a native crate's Rust sources, every selected leg
whose tests exercise that native builds it from source (via `scripts/rust_mode.py
dev`) instead of pulling the wheel.

## Current state (verified against the tree, corroborated by both reviews)

- `unified-unit.yaml`: a `select` job (checkout `fetch-depth: 0`) runs
  `infra/ci/select_tests.py` → a matrix of legs `{label, package, extras,
  test_paths}`; the `unit` job runs `uv run --package <pkg> <extras> --group test
  pytest -n auto --dist=worksteal <test_paths>` (no `--frozen`, so it
  re-resolves). Three bespoke jobs (`levanter_torch`, `levanter_tpu`,
  `iris_e2e_smoke`) are gated on a `suites` output. `unit_tests` aggregates.
- `select_tests.py`: `SCOPES = (rigging, haliax, iris, fray, levanter, zephyr,
  marin)`. `SOURCE_ROOTS` and `TEST_DIR` are comprehensions over `SCOPES` /
  `UV_PACKAGE`. dupekit/finelog/ducky are not scopes. `BROAD_TRIGGERS` (`uv.lock`,
  `pyproject.toml`, `select_tests.py`, `unified-unit.yaml`) run the full matrix.
  The diff uses `git diff --name-only {base}...HEAD`.
- `dupekit-unit.yaml`: `changes` (dorny/paths-filter) → `check-user-mode` (guards
  against a committed dev-mode path source in dupekit AND finelog pyprojects) →
  `unit-test` (pytest vs prebuilt wheel, py3.12 + 3.13) + `rust-checks` (cargo
  fmt/clippy/test on `lib/dupekit/rust`).
- **finelog Python tests run nowhere in CI**; **finelog has no cargo checks**.
- `rust_mode.py` toggles `marin-dupekit-native` and `marin-finelog-server`
  between dev (path source → build from source) and user (prebuilt wheel) across
  three pyprojects. The `check-user-mode` guard blocks committing a dev-mode
  source.
- Native modules: `dupekit_native` (single light crate) and `finelog_server`
  (a heavy datafusion/arrow cargo workspace `pyext`; its Cargo.toml notes the
  *link alone* is multi-minute). finelog's Rust also compiles its own proto copy
  (`lib/finelog/rust/proto/`, `pyext/build.rs`); both `Cargo.lock` files are
  committed under `rust/` and appear in each crate's uv `cache-keys`.
- **Who exercises the native (verified):**
  - `marin-dupekit-native`: `dupekit` (its own tests via the `dupekit` proxy) and
    `marin` (`tests/processing/classification/deduplication/test_fuzzy.py`,
    `tests/datakit/test_decon.py` call dupekit).
  - `marin-finelog-server`: `finelog` (`test_embedded.py`) and `iris` (declares
    `marin-finelog-server` as a runtime dep at `lib/iris/pyproject.toml:17`; the
    controller starts the in-process native server and `lib/iris/tests/conftest.py`
    drives it).
  - Transitive-only carriers (`fray`/`levanter`/`zephyr` depend on `marin-iris`,
    `marin` depends on `marin-iris`) install the finelog wheel but do not exercise
    the native in unit tests — they stay on the wheel.

## Design

### Source-build policy (resolves the wmoss constraint, incl. cross-lib)

Two facts from selection drive it:

- `native_changed` = the set of native crates whose `lib/<pkg>/rust/**` changed
  in the diff. Computed with rename detection disabled (`git diff --name-only
  --no-renames {base}...HEAD`) so a file moved out of the crate still surfaces.
- `SOURCE_LEGS[native]` = the scopes whose tests exercise that native:
  - `dupekit` (`lib/dupekit/rust`) → `{dupekit, marin}`
  - `finelog` (`lib/finelog/rust`) → `{finelog, iris}`

A selected leg gets source-build mode iff its scope is in `SOURCE_LEGS[n]` for
some `n ∈ native_changed`. This fixes the blocker both reviews raised: a
`lib/dupekit/rust` change now source-builds the `marin` leg too, and a
`lib/finelog/rust` change source-builds the `iris` leg — no leg tests a stale
wheel for a native its tests exercise.

Crucially, source-build is gated **only** on `native_changed`, decoupled from
broad triggers and full runs: a `uv.lock` / root-`pyproject.toml` bump runs the
full matrix but does **not** source-build (the crate source is unchanged, so the
wheel is byte-equivalent to a source build). This keeps the finelog compile off
the highest-traffic CI paths — it runs only on actual `lib/finelog/rust` edits,
which are rare. On a `--run-all-tests` push to main the diff base is
`github.event.before`, so `native_changed` is still computed and source-build
stays conditional; a manual dispatch with no base defaults to source-building
both (conservative).

### Step 2 — shared Rust setup script

New `infra/ci/setup_rust_ext.sh`:

```sh
#!/usr/bin/env bash
# Build the native Rust extensions from source for a CI test leg, so a same-PR
# Rust change is exercised by the Python tests (a prebuilt wheel would not carry
# it — issue #7254). Invoked by unified-unit.yaml for legs flagged `setup: rust`.
# uv.lock is re-resolved (and rewritten in-tree) by the leg's `uv run`; that is
# expected and discarded with the runner.
set -euo pipefail
python3 scripts/rust_mode.py dev   # flip [tool.uv.sources] to local rust/ paths
```

`rust_mode.py dev` flips both native packages; uv compiles only the one the
tested package depends on (a metadata hook may run for the other crate — cheap).
The toolchain and cargo cache are provided by the workflow (below), not this
script, so the script stays a thin, reusable "prepare source build" step.

### Step 3a — generic per-leg setup in `unified-unit.yaml`

Matrix legs gain two optional declarative fields: `setup` (a capability tag,
default `""`) and `timeout` (minutes, default `15`). The `unit` job:

```yaml
    timeout-minutes: ${{ matrix.timeout || 15 }}
    steps:
      - uses: actions/checkout@v5
      - uses: astral-sh/setup-uv@v7
        with: { python-version: "3.12", enable-cache: true }
      # --- Rust source-build setup (only legs flagged setup: rust) ---
      - if: matrix.setup == 'rust'
        uses: dtolnay/rust-toolchain@29eef336d9b2848a0b548edc03f92a220660cdb8  # stable, pinned to match rust-checks
      - if: matrix.setup == 'rust'
        uses: Swatinem/rust-cache@<pin>
        with:
          workspaces: |
            lib/dupekit/rust
            lib/finelog/rust
      - if: matrix.setup == 'rust'
        run: bash infra/ci/setup_rust_ext.sh
      # --- pytest (unchanged) ---
      - run: uv run --package ${{ matrix.package }} ${{ matrix.extras }} --group test pytest ... ${{ matrix.test_paths }}
```

The mechanism is a fixed, declarative capability tag (`setup: rust`) with a
gated, hard-coded step group — not an opaque `bash ${{ matrix.setup }}` (both
reviews flagged the shell-injection surface of the latter). A future suite adds
its own tag (`setup: torch`) and gated block. The source-build *decision* lives
in `select` (which already has the diff and `fetch-depth: 0`); the workflow owns
*how* the tag is implemented.

### Step 3b — `select_tests.py` changes

1. Add `dupekit`, `finelog` to `SCOPES` and `UV_PACKAGE` (`marin-dupekit`,
   `marin-finelog`). `SOURCE_ROOTS`/`TEST_DIR` follow automatically (they are
   comprehensions). Joining the import graph is a net gain: a dupekit/finelog
   *source* change now also selects downstream tests in `marin`/`iris` that
   import them.
2. Force trigger for Rust-only changes (invisible to the Python import graph),
   scoped to non-graph dirs so it stays **additive** to normal module seeding:
   force scope `s` on a change under `lib/<s>/rust/**`, `lib/<s>/pyproject.toml`,
   or `lib/<s>/config/**`. Source files under `lib/<s>/src/**` keep flowing
   through the existing source-root seeding (so downstream selection survives) —
   the force must not consume/replace that path.
3. Compute `native_changed` (rename-safe) and attach `setup="rust"` +
   `timeout=30` to every leg whose scope is in `SOURCE_LEGS[n]` for
   `n ∈ native_changed`. Thread `setup`/`timeout` through `matrix_leg` /
   `scope_legs` / `compute_matrix` / `full_matrix` (defaults `""` / `15`).
4. Add `infra/ci/setup_rust_ext.sh` and `scripts/rust_mode.py` to
   `BROAD_TRIGGERS`, so a change to the build machinery re-runs the full matrix
   and exercises a source build.

### Step 3c — retire `dupekit-unit.yaml`, add cargo coverage for finelog

- Delete `dupekit-unit.yaml` (its Python `unit-test` job → unified matrix leg).
- New `rust-checks.yaml`, modeled on dupekit-unit's job graph so branch
  protection never hangs: an always-run `changes` job (dorny/paths-filter) whose
  downstream `cargo` and `check-user-mode` jobs are conditioned on the filter
  (a skipped job still reports a status). It runs `cargo fmt --check`, `cargo
  clippy --workspace --all-targets -D warnings`, `cargo test --workspace` for
  **both** `lib/dupekit/rust` and `lib/finelog/rust` (closes finelog's gap;
  `--workspace --all-targets` covers finelog's `pyext` member and bin), and moves
  the `check-user-mode` guard verbatim. Path filter: `lib/dupekit/rust/**`,
  `lib/finelog/rust/**`, the three pyprojects, `scripts/rust_mode.py`,
  `infra/ci/setup_rust_ext.sh`, and `.github/workflows/rust-checks.yaml` itself.
- `ducky-unit.yaml` is left as-is (pure Python, not in this issue's scope).

## Open decisions (need maintainer input before/at merge)

- **D1 — Python 3.13 coverage.** dupekit-unit tests 3.12 + 3.13; unified-unit is
  3.12-only. Both libs publish 3.13 wheels. Recommend preserving coverage by
  emitting targeted 3.13 legs for dupekit/finelog (a matrix `python` field on
  those legs; the `unit` job reads `${{ matrix.python || '3.12' }}`). Call this
  out explicitly in the PR body — dropping it should be a choice, not an
  accident.
- **D2 — required-check migration (maintainer-only, branch protection).**
  Deleting `dupekit-unit.yaml` orphans any required check named for its jobs, and
  making a `rust-checks.yaml` job required without the always-run pattern would
  hang non-Rust PRs. The plan keeps the `unit-tests` aggregate as the Python
  required check (unchanged name) and uses the always-run `changes` pattern for
  rust-checks. The branch-protection update (drop old `dupekit-unit` checks, add
  the rust-checks job) must be done by a maintainer atomically with the merge — I
  cannot change branch protection.

## Risk driving the cost model (fable B1)

finelog's cold source build (datafusion/arrow) may approach or exceed a normal
leg budget. Mitigations, all in this plan (not deferred): source-build is gated
on actual `lib/finelog/rust` changes only (rare, off the hot paths);
`Swatinem/rust-cache` persists the cargo target dir from day one; source-build
legs carry `timeout: 30`. Implementation step 0: measure the cold finelog build
in a CI dry-run. If a cached-miss cold build still overruns even 30 min on the
2-vCPU runner, fall back to the **build-once alternative**: a single
`native-wheels` job (gated on `native_changed`, generous timeout, persistent
cache) builds the wheel from source and uploads it; consumer legs install that
wheel via `UV_FIND_LINKS` and stay Rust-free. That is airtight and cheaper for a
heavy crate but adds a cross-job artifact dependency; preferred only if the
measured cost forces it.

## Other implementation notes (from review)

- **finelog native is test-group-only** (`[dependency-groups] test`), unlike
  dupekit's hard runtime dep — both resolve correctly (wheel in user mode, source
  in dev mode).
- **dupekit bench tests self-skip** without `--run-benchmark`
  (`tests/bench/conftest.py`), so the leg runs `lib/dupekit/tests/` with `--group
  test` exactly as today.
- **finelog `-n auto` in ini** layers under the unified CLI `-n auto
  --dist=worksteal` (pytest prepends addopts; CLI wins) — leave finelog's ini
  unchanged to avoid changing standalone developer behavior.
- **Rename/delete edges**: `--no-renames` surfaces a file moved out of `rust/`;
  deleted `.py` sources still seed their module (path math, no file read), so the
  package still force-selects. Deep downstream propagation of a *deleted* module
  is a pre-existing `select_tests` limitation, not introduced here.
- **Optional (N7)**: add `lib/finelog/` to `iris-e2e-smoke`'s
  `EXTRA_SUITE_TRIGGERS` so a finelog change also smoke-tests the iris path.

## Rollout / testing

0. Measure finelog cold source-build time in a CI dry-run (drives D-fallback).
1. Land select_tests + unified-unit + `setup_rust_ext.sh` + `rust-checks.yaml`
   together; delete `dupekit-unit.yaml`.
2. Verify on this PR via the printed `select` matrix: (a) `lib/dupekit/rust` touch
   → `dupekit` and `marin` legs both flagged `setup: rust`; (b) `lib/finelog/rust`
   touch → `finelog` and `iris` legs flagged; (c) `lib/finelog/src` touch →
   `finelog` leg on the wheel (no setup), plus any import-selected downstream; (d)
   `uv.lock` bump → full matrix, no `setup: rust` anywhere; (e) unrelated PR →
   neither lib selected.
3. Confirm `unit_tests` stays green; coordinate the branch-protection change (D2)
   with a maintainer before removing the old required checks.
