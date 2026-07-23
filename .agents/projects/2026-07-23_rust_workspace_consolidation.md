# Consolidating native crates into a shared Rust workspace

Tracking issue: [#7563](https://github.com/marin-community/marin/issues/7563).

## Summary

Marin has grown three independent native crate trees, each a self-contained
Cargo workspace under its owning Python package:

- `lib/finelog/rust` — the finelog log store, query engine, and Connect/RPC
  server. Ships a standalone `finelog-server` binary and a PyO3 extension
  (`finelog_server`) in the `marin-finelog-server` wheel.
- `lib/dupekit/rust` — text-dedup kernels. Ships the `dupekit_native`
  extension in the `marin-dupekit-native` wheel.
- `lib/iris/rust` — the controller endpoint-proxy dataplane (route parsing,
  JWT/IAP/session/federation verification, streaming). Ships the `iris_native`
  extension in the `marin-iris-native` wheel. Landing via
  [`[iris] Move endpoint proxy dataplane into Rust`](https://github.com/marin-community/marin/commit/2ae74b46e); not yet on `main`.

The layout keeps each wheel isolated but gives us no place for shared internal
crates, one dependency-version policy, one lint/profile policy, or one release
mechanism. The most concrete cost today: finelog and iris each carry their own
`jsonwebtoken`-based EdDSA verification (`lib/finelog/rust/src/server/auth.rs`,
1053 lines; `lib/iris/rust/src/auth.rs`, 541 lines), and finelog's module doc
already notes it re-implements "the same shape as rigging's `server_auth`."
Three copies of one security-sensitive routine is the motivating duplication.

This document proposes moving all native crates into a single top-level `rust/`
Cargo workspace with reusable internal crates and thin PyO3 extension crates
that stay independently buildable and publishable, and records the ownership,
dependency, lockfile, versioning, CI, and migration rules. The direction (merge
the trees and take the churn) is settled by the issue owner; the work here is
how to do it without regressing wheel isolation, dependency direction, or
release independence.

## Goals and non-goals

Goals (from the issue "Done when"):

1. One workspace layout with documented crate ownership, dependency boundaries,
   lockfile policy, and versioning model.
2. Two PyO3 extensions sharing one internal crate while building and publishing
   as separate Python distributions.
3. CI runs shared fmt/lint/test once, then builds only affected wheels.
4. Local `uv` development and Docker builds stay reproducible with no prior
   PyPI release.
5. A migration plan covering finelog, dupekit, and iris.

Non-goals:

- Publishing internal Rust crates to crates.io. Internal crates are path-only
  and compiled into whichever wheels depend on them.
- Merging the wheels. Each Python distribution keeps its own name, version,
  owner, and release cadence.
- Changing any wheel's public Python API or module name.

## Current state

| Crate tree | Cargo shape | Wheel | Module | pyo3 | Heavy shared deps |
|---|---|---|---|---|---|
| `lib/finelog/rust` | workspace: `finelog` lib+bin, member `pyext` | `marin-finelog-server` | `finelog_server` | 0.29 | jsonwebtoken 10, axum 0.7, hyper/hyper-rustls (ring), arrow/parquet 58, datafusion 53 |
| `lib/dupekit/rust` | single crate | `marin-dupekit-native` | `dupekit_native` | 0.26 | arrow/parquet 57.1, blake3 |
| `lib/iris/rust` | workspace: `iris-proxy` lib, member `pyext` | `marin-iris-native` | `iris_native` | 0.29 | jsonwebtoken 10, axum 0.7, hyper/hyper-rustls (ring), blake3, moka, reqwest |

Overlap the crates do not currently share:

- `jsonwebtoken = "10"` with `default-features = false, features = ["use_pem", "rust_crypto"]` — byte-identical in finelog and iris. Both verify EdDSA tokens with an audience guard.
- The `hyper` + `hyper-rustls` + `hyper-util` client stack pinned to the `ring` rustls provider — identical in finelog and iris, and load-bearing (a second provider pulls the aws-lc C/asm toolchain into the images).
- `axum = "0.7"`, `blake3`, `serde`, `tokio`, `arrow`/`parquet` (at drifting versions: finelog 58, dupekit 57.1).
- `pyo3` at two versions (0.29 for finelog/iris, 0.26 for dupekit).

Build and release tooling that assumes the per-package layout:

- Root `pyproject.toml`: `[tool.uv.workspace] exclude = ["lib/finelog/rust", "lib/dupekit/rust"]` and per-native `[tool.uv.sources]` path entries under RUST-DEV markers.
- `scripts/rust_mode.py`: toggles path sources for `marin-dupekit-native` and `marin-finelog-server` between local source build ("dev") and published wheel ("user").
- `infra/ci/select_tests.py`: `NATIVE_PACKAGES = {"dupekit": "lib/dupekit/rust", "finelog": "lib/finelog/rust"}`; a change under a crate's `rust/` tree force-selects that scope for a source build. Only the owning scope is source-built.
- `.github/workflows/rust-checks.yaml`: one `dupekit-rust` job and one `finelog-rust` job, each running fmt/clippy/test against its own `--manifest-path`.
- Per-package release workflows (`dupekit-release-wheels.yaml`, `finelog-release-wheels.yaml`, and iris's new `iris-native-release-wheels.yaml`) drive per-package `build_package.py` scripts (zig-cross manylinux + macOS matrix).
- Two committed locks: `lib/finelog/rust/Cargo.lock`, `lib/dupekit/rust/Cargo.lock`. No cargo entry in `.github/dependabot.yml` (only `uv` and `npm`), so today's Rust dependency graph is not auto-updated at all.
- Docker: `lib/finelog/deploy/Dockerfile` does `COPY lib/finelog/rust/ ./` then `cargo build -p finelog --bin finelog-server`; iris's Dockerfile copies `lib/iris/rust/` and runs `maturin build`.

## Proposed layout

One workspace at the repo root:

```
rust/
├── Cargo.toml                 # [workspace] — members, workspace.package,
│                              #   workspace.dependencies, profiles, lints
├── Cargo.lock                 # single committed lock for all native code
├── rust-toolchain.toml        # pinned stable toolchain (fmt/clippy determinism)
├── crates/                    # internal libraries: publish = false, path-only
│   └── marin-jwt/             #   shared EdDSA/JWT verify + audience/exp policy
│                              #   (marin-http, marin-arrow: future candidates,
│                              #    extract only when behavior is shared — see below)
├── finelog/                   # finelog server lib + `finelog-server` binary
│   ├── src/…  proto/…
├── finelog-pyext/             # cdylib → finelog_server → marin-finelog-server
│   ├── Cargo.toml  pyproject.toml  src/lib.rs  *.pyi
├── dupekit-pyext/             # cdylib → dupekit_native → marin-dupekit-native
│   ├── Cargo.toml  pyproject.toml  src/lib.rs
├── iris-proxy/                # iris endpoint-proxy dataplane lib
└── iris-pyext/                # cdylib → iris_native → marin-iris-native
    ├── Cargo.toml  pyproject.toml  src/lib.rs  *.pyi
```

The wheel's maturin `pyproject.toml` moves next to its `*-pyext` crate. The
build driver (`build_package.py`) and `rust_mode.py` path sources retarget from
`lib/<pkg>/rust` to `rust/<pkg>-pyext`; the wheel name, module name, and public
API are unchanged.

### Crate taxonomy

Three roles, distinguished by naming so ownership and dependency rules are
legible from the path alone:

- Internal crate (`rust/crates/<name>`): a reusable library. `publish = false`,
  `version = "0.0.0"`, never released to crates.io. Compiled from source into
  every wheel that depends on it.
- Component library (`rust/<component>`): a package's own non-trivial Rust logic
  that is not (yet) shared — e.g. `finelog` (server/store/query) and
  `iris-proxy`. May also be `publish = false`. A standalone binary (finelog's
  `finelog-server`) lives here.
- Extension crate (`rust/<component>-pyext`): a thin `cdylib` that wraps a
  component library (or an internal crate directly, as dupekit does) and is the
  maturin build target for exactly one wheel. Nothing depends on a pyext.

### Ownership rules

Physical co-location under `lib/<pkg>/` is replaced by explicit ownership:

- `.github/CODEOWNERS` maps `rust/finelog*` and `rust/iris-proxy` / `rust/iris-pyext`
  to their existing package owners, and `rust/crates/**` to a platform owner set
  (the group that reviews cross-cutting security/runtime code — the natural home
  for `marin-jwt`).
- Each wheel's Python distribution keeps its current name and owner. The wheel
  is the ownership unit for release; the crate path is the ownership unit for
  code review.
- An internal crate that a second package wants to depend on requires sign-off
  from the internal crate's owners (they take on the compatibility obligation).

## Dependency boundaries

The Python dependency direction is `{iris, haliax} → {levanter, zephyr} → marin`;
native code must not create reverse edges. Rules:

1. The crate dependency graph is a DAG. Extension crates are sinks: no crate
   depends on a `*-pyext`. No `*-pyext` depends on another `*-pyext`.
2. Internal crates (`rust/crates/**`) depend only on third-party crates and
   other internal crates. They never depend on a component library or an
   extension crate. This keeps them leaves (or near-leaves) that any package can
   consume without importing that package's world.
3. A shared internal crate must not encode one consumer's identity. `marin-jwt`
   verifies tokens against a supplied key set and audience; it does not know
   "finelog" or "iris." Consumer-specific policy (which audiences are trusted,
   which CIDRs bypass) stays in the component library.
4. Because internal crates carry no Python-level dependency, sharing `marin-jwt`
   between `finelog-pyext` and `iris-pyext` creates no Python edge between
   `marin-finelog` and `marin-iris`. Dependency direction is preserved as long as
   the rules above hold — but they are not self-enforcing (see below).
5. Internal crates stay layer-neutral: no PyO3, no Python-driven build scripts,
   no consumer-specific types. A crate that pulls in `pyo3` or a component's
   domain types is no longer a leaf and cannot be shared freely.

Cargo does not enforce most of this. It rejects dependency cycles, and PyO3's
`links = "python"` constraint forces one pyo3 version across any single
`cargo build`, but Cargo will happily let an internal crate depend on a
component, or `iris-proxy` depend on `finelog`, if a manifest says so. The
boundary rules therefore need a CI architecture check: a small script over
`cargo metadata` that asserts an explicit allowed-edge matrix — each crate is
tagged with its role (internal / component / pyext) and Python layer, and any
edge outside the permitted set fails the build. `cargo-deny` covers dependency
bans and duplicate-version policy but does not express this graph, so the
allowed-edge check is its own step. Cargo gives us cycle-freedom and pyo3
singleness for free; everything else is the CI check's job.

## Versioning model

- Internal and component crates: `version = "0.0.0"`, `publish = false`. They
  have no independent version because they are never distributed on their own.
  This removes a whole class of crates.io version-bump bookkeeping. Where a
  binary or extension reports a version (finelog's `finelog-server` exposes one,
  and `CARGO_PKG_VERSION` on a `0.0.0` component would regress it), embed the
  wheel/image version and git revision explicitly via a `build.rs` env stamp, so
  provenance survives the move to unversioned component crates.
- Wheels: unchanged. Each Python distribution keeps independent SemVer, its own
  release workflow, and its own tag namespace (`finelog-v*`, `dupekit-v*`,
  `iris-native-v*`). `build_package.py` still stamps the resolved version into
  the extension's `pyproject.toml` at build time.
- Consequence of sharing: a change to `marin-jwt` requires re-releasing every
  wheel that links it (finelog and iris) to ship the change — that is inherent
  to sharing code, and is exactly the coupling we want (one fix, both planes).
  It does not force a dupekit release: dupekit does not depend on `marin-jwt`,
  so its wheel and version are untouched. "Change one component, publish only
  the affected wheels" holds at the wheel granularity, driven by the crate DAG.
- Coordinated-release contract for shared-crate changes. Because a shared fix
  spans multiple wheels but each wheel publishes on its own workflow, a security
  fix could ship in one plane and silently miss the other. The affected-wheel
  computation (CI model, below) emits the exact set of wheels a change touches;
  a shared-crate change must re-release all of them, and CI flags any affected
  wheel whose version was not bumped in the same change so the set can never be
  partially published.

## Lockfile policy

- One committed `rust/Cargo.lock` for the whole workspace, with `resolver = "2"`.
  Every wheel and the finelog binary resolve against this one lock.
- A single lock does not by itself force one version of a dependency: if two
  members request semver-incompatible ranges (`arrow = "57.1"` vs `"58"`), the
  lock records both and Cargo builds both. Version unification comes from two
  things layered on the lock: (a) `[workspace.dependencies]` that every member
  inherits with `dep.workspace = true`, so a dependency is declared once; and
  (b) a `cargo-deny` `bans.multiple-versions` policy for the crates we insist be
  singletons (pyo3, arrow, parquet, jsonwebtoken, rustls provider), with an
  explicit allowlist for any duplicate we knowingly tolerate. The lock plus these
  two is what makes "one arrow, one pyo3" real; the lock alone is not.
- Replaces the two per-crate locks. Today they drift unchecked — arrow 58 vs
  57.1, pyo3 0.29 vs 0.26; the singleton policy turns that drift into a CI
  failure instead of a silent divergence.
- Reproducibility needs pinning on top of the lock. Build with `--locked`
  (cargo) / `--frozen` (uv) everywhere — CI, Docker, release — so a build fails
  instead of silently updating the lock. Pin the toolchain to an exact version
  in `rust/rust-toolchain.toml` (a floating `stable` channel reformats and
  relints differently across releases), and pin maturin and the builder images
  used for release wheels.
- Add a `cargo` ecosystem entry to `.github/dependabot.yml` pointed at `/rust`.
  This is net-new coverage; there is no cargo dependabot today.
- Trade-off: a shared-dependency bump touches the single lock and so rebuilds
  every dependent wheel. That is the point of a shared lock; the affected-wheel
  CI selection (below) keeps the blast radius to the wheels that actually
  changed.

## Build and local development

Unchanged model, retargeted paths:

- `scripts/rust_mode.py`: the `TARGETS` path sources move from
  `lib/<pkg>/rust` to `rust/<pkg>-pyext`. `dev` mode still injects
  `marin-<pkg>-native = { path = "rust/<pkg>-pyext" }` so `uv sync` builds the
  extension from source and tracks live edits; `user` mode still resolves the
  published wheel. The RUST-DEV markers and the CI "user-mode" guard are
  unchanged in spirit.
- Root `pyproject.toml`: `[tool.uv.workspace] exclude` lists the `rust/*-pyext`
  directories (maturin projects must stay out of the uv workspace, same as
  today). The pure-Python fronts (`marin-finelog`, `marin-dupekit`,
  `marin-iris`) remain uv workspace members built from source.
- maturin `manifest-path` stays `Cargo.toml` (the pyext's own manifest); the
  pyext's path dependency on internal crates is resolved through the workspace.
  No `uv sync` compiles Rust unless the package is in dev mode — the extension
  is a pre-built wheel otherwise.

## Docker builds

The one real behavioral change. A wheel or binary that lives in a shared
workspace needs the workspace root, the shared lock, and its member manifests in
the build context.

- Copy the whole `rust/` tree, then `cargo build -p finelog --bin finelog-server
  --locked` from the workspace root. `-p` scopes the *compile* to one package
  and its dependencies, so unrelated crates are not built — but Cargo loads
  *every* member manifest listed in the workspace `Cargo.toml` before it honors
  `-p`. A build context that copies only a subset of members (e.g. finelog's
  Dockerfile copying `rust/finelog*` but not `rust/dupekit-pyext`) fails at
  workspace load with `No such file or directory` on the absent member. Verified
  in the prototype: the partial copy errors, the whole-tree copy builds. So the
  Dockerfile copies `rust/` wholesale.
- The whole `rust/` tree is source-only and small (the per-package `target/` is
  already dockerignored). Cargo's `--mount=type=cache` registry and target caches
  keep incremental builds fast; the extra source trees cost nothing to copy, and
  `-p` means they are not compiled.
- Copy `rust/rust-toolchain.toml` into the context and build `--locked` so the
  image pins the same toolchain and lock as CI.
- No prior PyPI release is required: the image builds the binary/extension from
  the copied source against the committed lock, exactly as today.
- Alternative considered and rejected: a per-wheel `cargo build` outside a
  workspace (git-vendoring shared crates). It reintroduces version drift and
  defeats the point of consolidation. A generated pruned-workspace manifest (what
  `maturin sdist` does automatically) is a possible future optimization if the
  copied context ever grows large enough to matter; today it does not.

## CI model

Two concerns, split cleanly:

1. Shared fmt/lint/test — one job over the whole workspace, replacing the
   per-crate `dupekit-rust`/`finelog-rust` jobs:

   ```
   cargo fmt --all --check
   cargo clippy --workspace --all-targets --locked -- -D warnings
   cargo test --workspace --locked
   cargo deny check           # bans + multiple-version singleton policy
   <allowed-edge graph check> # role/layer matrix over cargo metadata
   ```

   Runs when `rust/**` changes. One toolchain, one cache, one policy. The
   existing "user-mode" guard (no committed dev-mode path source) is retained.
   `cargo test --workspace` unifies features across members, so it exercises the
   logic but not each wheel's real feature set — the per-wheel build in (2)
   covers that.

2. Build only affected wheels. A wheel is affected when its pyext crate or any
   crate it transitively depends on changed. Two input classes:

   - Crate-local changes → DAG selection. A small script over `cargo metadata`
     maps changed files → owning crate → dependent wheels. A change under
     `rust/crates/marin-jwt` selects both the finelog and iris wheel builds; a
     change under `rust/dupekit-pyext` selects only the dupekit wheel.
   - Global inputs → all-wheel fan-out. The root `Cargo.toml`, `Cargo.lock`,
     `rust-toolchain.toml`, shared `[profile.*]`, the build drivers, and any
     crate deletion cannot be attributed to one package, so they conservatively
     select every wheel. (Precise lock-driven selection — diffing the old and new
     resolved graphs to rebuild only wheels whose resolved deps changed — is a
     later optimization; start conservative.)
   - Each affected wheel gets an isolated production-mode `maturin build` plus a
     wheel install + `import` smoke, so an abi3 / `extension-module` / feature
     regression that the workspace test masks is caught per wheel.
   - `infra/ci/select_tests.py` extends the same idea for the Python unit legs:
     `NATIVE_PACKAGES` becomes a crate→scopes map, and a shared-crate change
     force-selects a source build for every dependent scope (today only the
     owning scope is source-built — that rule cannot survive shared crates,
     since a `marin-jwt` change must exercise both finelog and iris).
   - Release workflows stay per-wheel and independently dispatchable; their
     `pull_request` build-smoke `paths` filters expand to include the internal
     crates the wheel links, and the affected-wheel set drives the
     coordinated-release check (versioning model).

The net is one lint/test pass for the whole native tree, then a fan-out that
rebuilds and import-tests exactly the wheels a change can affect.

## Prototype

To de-risk the load-bearing mechanic — two PyO3 extensions sharing one internal
crate, each building and publishing as a separate wheel from one workspace with
one lock — I built a minimal instance of the proposed layout and exercised it
end to end.

Layout: a workspace with an internal `marin-jwt` crate (`publish = false`, real
`jsonwebtoken` EdDSA verify) consumed by two extension crates, `finelog-pyext`
(module `finelog_server`, wheel `marin-finelog-server`) and `dupekit-pyext`
(module `dupekit_native`, wheel `marin-dupekit-native`), each with its own
maturin `pyproject.toml`.

Verified with cargo 1.97.1 / maturin 1.14.1:

```
# one workspace build, one lock, marin-jwt compiled once as a local member
$ cargo build --workspace
   Compiling marin-jwt v0.0.0 (/tmp/rust-proto/crates/marin-jwt)
   Compiling finelog-native v0.2.0 (/tmp/rust-proto/finelog-pyext)
   Compiling dupekit-native v0.1.2 (/tmp/rust-proto/dupekit-pyext)
    Finished `dev` profile in 40.64s
$ ls *Cargo.lock */Cargo.lock          # single root lock, no per-crate locks
Cargo.lock

# two independent wheels at independent versions, each from its own pyproject
$ (cd finelog-pyext && maturin build --release)
📦 marin_finelog_server-0.2.0-cp312-abi3-manylinux_2_34_x86_64.whl
$ (cd dupekit-pyext && maturin build --release)
📦 marin_dupekit_native-0.1.2-cp312-abi3-manylinux_2_34_x86_64.whl

# each sdist vendors the shared internal crate and a pruned workspace manifest
$ tar tzf marin_finelog_server-0.2.0.tar.gz | grep -E 'jwt|Cargo'
  marin_finelog_server-0.2.0/Cargo.lock
  marin_finelog_server-0.2.0/Cargo.toml            # members trimmed to
  marin_finelog_server-0.2.0/crates/marin-jwt/…    #   [crates/marin-jwt, finelog-pyext]
  marin_finelog_server-0.2.0/finelog-pyext/Cargo.toml

# the sdist builds a wheel standalone — no surrounding workspace, no prior
# marin-jwt release anywhere
$ tar xzf marin_finelog_server-0.2.0.tar.gz && cd marin_finelog_server-0.2.0
$ maturin build --release
📦 marin_finelog_server-0.2.0-cp312-abi3-manylinux_2_34_x86_64.whl
```

Findings:

- One `cargo build --workspace` produces exactly one root `Cargo.lock`; the
  internal `marin-jwt` appears as a local member (`version = "0.0.0"`, no
  registry source). All members resolve against that one lock; unifying a
  specific dependency to one version still needs `[workspace.dependencies]` plus
  the `cargo-deny` singleton policy (see Lockfile policy).
- The two wheels build from their own `pyproject.toml`s at independent versions
  (0.2.0 and 0.1.2) as `abi3` wheels portable across CPython ≥ 3.12.
- maturin vendors path-dependency source into each sdist and rewrites the
  workspace `Cargo.toml` to only the crates that wheel links — the finelog sdist
  ships `crates/marin-jwt` and `finelog-pyext` and omits `dupekit-pyext`
  entirely. The extracted sdist builds a wheel standalone. This is the direct
  proof of "publish as separate distributions" and "reproducible without a prior
  PyPI release": a wheel and its sdist never require the shared crate to be
  released on its own.
- maturin emits a benign note about `[project]` metadata (license) living at the
  workspace level; each pyext `pyproject.toml` keeps its own `[project]` table,
  so this is cosmetic. Real crates already carry these fields.

A second pass added a component crate between the extension and the internal
crate — `finelog-pyext → finelog → marin-jwt`, the real finelog shape — and
tightened every build to `--locked`:

- The three-level chain builds with `--locked`, and the finelog sdist vendors all
  three levels (`crates/marin-jwt/src/lib.rs`, `finelog/src/lib.rs`,
  `finelog-pyext/src/lib.rs`) with the workspace members pruned to
  `["crates/marin-jwt", "finelog", "finelog-pyext"]`. The extracted sdist builds
  a wheel standalone under `--locked`.
- The Docker partial-copy failure is real: a context that copies `finelog*` and
  `crates/` but not `dupekit-pyext` (still a listed member) fails at workspace
  load with `No such file or directory`; copying the whole `rust/` tree builds.
  This is why the Docker section copies `rust/` wholesale.
- Adding a member left the committed lock stale, and `--locked` correctly refused
  to build until the lock was regenerated — the guardrail behaving as intended.

Scope of the prototype: it validates the workspace/lock mechanics, the
three-level path-dependency chain, sdist self-containment and standalone builds,
and the Docker copy rule. It does not yet exercise the real crates' generated
protos and static assets, the zig cross-compile matrix, the `uv` dev-mode source
build, or an actual multi-stage Dockerfile. Those become explicit gates in the
migration: a standalone locked sdist build and an isolated wheel-import test for
each real package, on the pinned maturin version (manifest pruning is maturin
behavior, so the maturin pin is load-bearing).

## Migration plan

Phased so each PR lands green on its own. The version unifications are separated
from the move, and the move retargets every build consumer in one atomic PR — a
half-moved tree is the one state that cannot be green.

Phase 0 — land iris native first (in flight). The iris endpoint-proxy crate
(`2ae74b46e`) is mid-flight on its own branch. Let it land under
`lib/iris/rust` as designed; this consolidation moves it afterward. Merging
consolidation into an unlanded crate would couple two large reviews.

Phase 1a — unify `pyo3` in place, before any move (a hard prerequisite).
PyO3's `links = "python"` means a single `cargo build`/`cargo test` over a
workspace cannot contain two pyo3 versions, so dupekit (0.26) must move to 0.29
to share a workspace with finelog/iris at all. Do this as its own PR against the
current `lib/dupekit/rust` layout: bump, fix any 0.26→0.29 API breaks, run the
dupekit suite. This isolates a real API-migration risk from the mechanical move.

Phase 1b — the mechanical move, all consumers retargeted atomically. In one PR:
`git mv` each `lib/<pkg>/rust` tree into `rust/` (`rust/finelog`,
`rust/finelog-pyext`, `rust/dupekit-pyext`, `rust/iris-proxy`, `rust/iris-pyext`),
add the workspace `Cargo.toml`/`rust-toolchain.toml` and the single
`rust/Cargo.lock`, and in the same commit retarget every consumer that names the
old paths — `rust_mode.py`, the root `pyproject.toml` exclude/sources,
`rust-checks.yaml`, each release workflow (working dir + `paths` filters) and its
`build_package.py`, the finelog and iris Dockerfiles, `select_tests.py`, and
CODEOWNERS. Leaving `rust-checks.yaml` or a release workflow on a stale
`--manifest-path lib/<pkg>/rust` is the failure codex flagged: the job runs
against a path that no longer exists. No shared crate and no arrow bump yet —
this PR reproduces today's wheels modulo the pyo3 already unified in 1a. `arrow`
stays split (finelog 58, dupekit 57.1) for now, allowlisted in the `cargo-deny`
duplicate policy; a workspace tolerates the duplicate, and forcing dupekit's
kernels onto arrow 58 is a separate tested change (Phase 1c, optional/deferred).

Phase 2 — extract the first shared crate (`marin-jwt`), behind a compatibility
gate. Two large auth implementations sharing a dependency does not prove
identical validation semantics, so before extraction write a JWT behavior matrix
and a regression suite covering: the algorithm allowlist (EdDSA only), `kid`
handling, audience and issuer checks, `exp`/`nbf` and clock-skew leeway, key
rotation (multiple valid keys), malformed-claim and malformed-token error
mapping, and each consumer's bypass policy (finelog's CIDR layer, iris's
IAP/session/federation paths). Extract the EdDSA verify/audience/exp core out of
finelog's `auth.rs` into `rust/crates/marin-jwt`, leaving consumer-specific
policy in the component libraries, point `finelog` and `iris-proxy` at it, and
require the regression suite to pass identically before and after. This is the
first payoff: one verified JWT implementation across the control and log planes.
`marin-http` and any further shared crate follow the same gate, and only once a
stable shared API (not merely shared dependencies) is identified.

Phase 3 — CI fan-out. Replace the per-crate rust jobs with the workspace
lint/test job, `cargo deny`, the allowed-edge graph check, and the
`cargo metadata`-driven affected-wheel selection with per-wheel isolated
build/import; extend `select_tests.py` to source-build all dependent scopes on a
shared-crate change; add the `cargo` dependabot entry and the coordinated-release
version-bump check.

Release land-order caveat (maintainer-only, unavoidable): no wheel is renamed,
so unlike the original finelog/dupekit native splits, this migration does not
need a new PyPI project or a pending-publisher dance. But the pyo3 bump (1a) and
any resolved-dependency change from the move (1b) alter a wheel's build inputs,
so a nightly cut should be dispatched and the root `uv.lock` refreshed and
committed so `uv sync --frozen` stays green. Flagged for a maintainer; the agent
cannot dispatch release workflows.

## Risks and trade-offs

- Churn. Every native path moves once. Mitigated by separating the risky part
  (the pyo3 API bump, Phase 1a) from a pure mechanical `git mv` (Phase 1b) that
  carries no logic change, so history follows the files and the move PR is
  reviewable as a rename.
- Shared-lock coupling. A dependency bump rebuilds all dependent wheels. This is
  the intended trade for version unification; affected-wheel selection bounds
  the CI cost, and dependabot makes the bumps visible and small.
- Docker context. Builds copy the workspace skeleton instead of one crate. Small
  source trees, `-p`-scoped builds, and cache mounts keep this cheap.
- Ownership legibility. Co-location under `lib/<pkg>` is lost; CODEOWNERS plus
  the `<component>` / `<component>-pyext` / `crates/<name>` naming convention
  replaces it.

## Future direction: growing iris into Rust

The endpoint-proxy move put iris's dataplane (route parsing, JWT/IAP/session/
federation verification, streaming) in Rust while the control plane (endpoint
registration, federation ownership, reconcile, scheduling) stayed in Python.
A natural follow-on is pulling more of iris across that seam. The workspace makes
this more feasible and shapes how to do it.

- The workspace is where an internal iris crate graph can grow. `iris-proxy`
  today is one lib crate; as more moves over it splits into internal crates
  (`iris-scheduler`, `iris-reconcile`, `iris-store`) that reuse the same
  `rust/crates/*` primitives finelog uses. Without consolidation, iris would
  re-invent a nested workspace under `lib/iris/rust` (the endpoint-proxy commit
  already had to, with `[workspace] members = ["pyext"]`). A larger iris-in-Rust
  is the strongest consumer of shared crates and so the strongest argument for
  the merge.
- Move compute kernels, keep glue in Python. The parts worth moving are
  self-contained, perf-sensitive, and narrow-interface: the reconcile diff engine
  (desired vs. observed → actions is a pure function), the scheduler's
  constraint/bin-packing core, token mint/verify, RPC (de)serialization hot
  paths, state projections — the same shape as finelog's store/query engine. The
  parts to leave in Python are the churny glue that talks to many external
  systems (k8s, GCP, the DB, the actor mailboxes); each is a wide FFI surface
  that trades Python's iteration speed for little perf gain. The dataplane /
  control-plane split is the right default seam, and "more iris in Rust" means
  moving more kernels across it while keeping the seam itself.
- One component, two consumption modes — finelog already proves it. The `finelog`
  lib is surfaced as both a standalone `finelog-server` binary and the
  `finelog_server` PyO3 extension. iris can follow the same path from one crate
  graph: an `iris-core` lib surfaced in-process via `iris-pyext` (shared memory,
  low-latency calls; the endpoint proxy's PyO3/Tokio listener is the foot in the
  door) and/or out-of-process as a binary Python drives over RPC (clean
  lifecycle, independent deploy). The design should not pick one; the workspace
  supports both.
- The risk to watch is FFI-surface sprawl: grow along the proto/RPC boundaries
  that already exist. Fine-grained Python objects threaded through PyO3 make the
  per-type boundary maintenance cost swamp the perf win. This is the same reason
  `marin-jwt` stays consumer-agnostic — layer-neutral internal crates are what
  keep the FFI surface narrow as the Rust footprint grows.

## Open questions for review

1. Internal crate home for `marin-jwt`: `rust/crates/` as proposed, or a
   dedicated `rust/crates/auth/` namespace anticipating session/IAP/federation
   helpers alongside JWT? The iris auth surface (IAP, session, federation) is
   larger than finelog's, so the shared crate may want to be `auth` with `jwt` as
   one module.
2. Rigging export path: the issue owner wants the shared auth reusable "from
   rigging as needed." Rigging is Python; is the intended reuse a PyO3 surface on
   the shared auth crate (its own tiny wheel), or does rigging call finelog/iris's
   existing extension? This decides whether the shared auth crate needs a pyext,
   and is the one open question that changes the crate taxonomy (an internal crate
   that also ships its own wheel).
3. `arrow`/`parquet` unification (finelog 58, dupekit 57.1): the plan keeps them
   split and allowlisted through the move (Phase 1b) and treats the dupekit 58
   bump as a separate tested change (Phase 1c). Confirm that ordering, or bump
   dupekit to 58 up front if its kernels are already known-good on 58.
4. Confirm the platform-owner set for `rust/crates/**` in CODEOWNERS — the group
   that takes the compatibility obligation for shared security-sensitive code.
