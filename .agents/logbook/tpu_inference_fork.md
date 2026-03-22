# tpu-inference Fork Logbook

## 2026-03-21 — Fork Setup, Fast Bootstrap, and Packaging Investigation

### Context

The `vllm_load_fast` branch had accumulated ~4,300 lines of Marin-side
monkeypatches to work around three TPU startup bottlenecks in `tpu-inference`.
Codex identified these over 36 iterative debug runs (documented in
`codex_vllm_refactor.md`):

1. Model bootstrap: `_get_nnx_model()` does expensive concrete random-init
   for Llama (20+ min). Fix: abstract-state via `nnx.eval_shape()` (3s).
2. Sampling RNG: `nnx.Rngs(...).params()` stalls on abstract state. Fix:
   direct `jax.random.key(seed)`.
3. KV cache allocation: first `sharded_allocate()` takes 19 min regardless
   of cache size. Fix: use upstream `num_gpu_blocks_override`.

Today's goal: move the fixes into a `tpu-inference` fork so Marin only needs
~10 lines of config pass-through instead of 4,300 lines of monkeypatches.

### Fork Created

- **Repo**: `marin-community/tpu-inference`
  (forked from `vllm-project/tpu-inference`)
- **Branch strategy** (documented in `docs/dev-guide/forking-policy.md`):
  - `main` — tracks upstream, never commit Marin changes
  - `feat/*` — one per logical change, clean for upstream PRs
  - `marin` — integration branch, merges all feat branches, what Marin pins to
- **Default branch**: changed to `marin` on GitHub (required for Actions to
  discover workflows on that branch)

### Feature Branch: `feat/fast-tpu-bootstrap`

Three changes to `tpu-inference`, all opt-in via `model_loader_extra_config`:

1. **Abstract dummy bootstrap** (`model_loader.py`):
   - `TpuBootstrapConfig` dataclass with validation
   - `_use_abstract_dummy_bootstrap()` — gated on `model_bootstrap="abstract_dummy"`,
     `load_format="dummy"`, architecture in `_ABSTRACT_BOOTSTRAP_ARCHITECTURES`
     (currently only `LlamaForCausalLM`), no quantization
   - Returns `nnx.eval_shape(create_abstract_model)` instead of concrete random-init

2. **Bootstrap-safe RNG** (`tpu_runner.py`):
   - `_init_sampling_rng()` — uses `jax.random.key(seed)` directly when
     abstract bootstrap is active, avoiding `nnx.Rngs(...).params()` stall

3. **Bootstrap-aware routing** (`model_loader.py`):
   - `_BOOTSTRAP_JAX_ROUTING_ALLOWLIST` (currently only `Qwen3MoeForCausalLM`)
   - When `prefer_jax_for_bootstrap=True`, overrides `_VLLM_PREFERRED_ARCHITECTURES`
     for allowlisted models so they route to `flax_nnx` instead of PyTorch wrapper

KV cache: deliberately NOT in the fork per Codex review. Use upstream
`num_gpu_blocks_override` from Marin side instead.

### Codex Review (Two Rounds)

**Round 1**: Narrowed abstract bootstrap to Llama only (Qwen3/Qwen3MoE
already implement `LoadableWithIterator`), typed config, removed KV clamp,
added tests.

**Round 2**: Narrowed routing to explicit allowlist (was checking full JAX
registry, would reroute GptOss). Added `model_config.quantization` check
(TPU quantization via `tpu_int8` wasn't blocked).

**Final verdict**: "No remaining correctness issue in the reviewed diff."

### CI Setup

- **Workflow**: `release.yml` on `marin` branch builds wheel via
  `python -m build --wheel`, uploads to GitHub Releases as prerelease
- **Versioning**: `0.0.0.dev{YYYYMMDD}+{short_sha}`
- **Pre-commit**: also runs on push to `marin` branch
- **Upstream workflows**: `check_ready_label.yml` and `pre-PR.yml` left
  as-is (Google's review process, harmless on fork)

### Marin-Side Changes

Reverted `vllm_load_fast` branch to main, deleted all monkeypatch files
(-4,355 lines), added:

- `vllm_server.py`: +8 lines in `_engine_kwargs_to_cli_args()` to pass
  `model_loader_extra_config` and `num_gpu_blocks_override` through to CLI
- `vllm_smoke_test.py`: added `--engine-kwargs-json` and `--load-format dummy`
  support
- `VLLM_NATIVE_PIP_PACKAGES`: changed from `("vllm-tpu",)` to `()` since
  deps now come from pyproject.toml
- `pyproject.toml`: added fork wheel URL to `vllm` extra
- `forking-policy.md`: updated with fork strategy and branch table

### Packaging Investigation — Where We Got Stuck

#### Attempt 1: `vllm-tpu==0.13.2.post6` + fork wheel (jax 0.8.0)

- Fork was based on upstream `main` which requires `jax==0.9.2`
- `vllm-tpu==0.13.2.post6` requires `jax==0.8.0`
- `uv sync` fails: jax version conflict

#### Attempt 2: Upgrade jax to 0.9.2

- Senior engineers approved jax upgrade
- Updated all jax pins in pyproject.toml (0.8.0 → 0.9.2)
- Also bumped torch (2.9→2.10), torchvision (0.24→0.25), triton (3.5→3.6)
  to match `vllm==0.18.0` requirements
- `uv sync` succeeded locally
- **On Iris worker**: `vllm-tpu==0.13.2.post6` crashed with
  `PackageNotFoundError: No package metadata was found for vllm`
  — the old `vllm-tpu` wheel is incompatible with jax 0.9.2

#### Attempt 3: Drop `vllm-tpu`, use stock `vllm==0.18.0` + fork

- Rationale: `vllm-tpu` bundles vllm + tpu-inference. Stock vllm has
  TPU platform support via plugin system. tpu-inference registers itself
  via entry_points. Should work separately.
- `uv sync` succeeded
- **On Iris worker**: `tpu_inference not found` — fork wheel's
  `install_requires` was empty, missing runtime deps
- Fixed: added `tpu-info`, `torchax`, `flax`, `qwix`, `runai-model-streamer`
- **On Iris worker**: `tpu_inference` now imports OK, but
  `torchvision::nms operator does not exist`
- Root cause: stock `vllm==0.18.0` hard-depends on `torchvision==0.25.0`
  from default PyPI (CUDA variant). On TPU workers, CUDA ops don't exist.
  The lockfile resolves CPU torchvision for `tpu` extra, but when both
  `tpu` and `vllm` extras are active, the `vllm` dep pulls the CUDA wheel.

#### Why `vllm-tpu` avoids this

`vllm-tpu` is NOT just stock vllm repackaged. It's a custom build that:
- Strips GPU-specific deps (torchvision with CUDA, flashinfer, CUDA toolkits)
- Adds TPU deps (tpu-inference, jax)
- May patch vllm internals for TPU

Stock `vllm` from PyPI assumes GPU and includes CUDA-dependent torchvision.
This is the fundamental reason `vllm-tpu` exists as a separate package.

### Current State

**Blocked on**: packaging. Cannot use stock `vllm==0.18.0` on TPU due to
CUDA torchvision dependency. Cannot use `vllm-tpu==0.13.2.post6` with
jax 0.9.2 (incompatible).

**Options**:
1. Use `vllm-tpu==0.13.3` (latest stable, jax 0.8.1) + rebase fork on
   `v0.13.3` tag. Pragmatic, works now, but old jax.
2. Build our own `vllm-tpu` wheel from stock vllm with GPU deps stripped.
   More work but gives us latest vllm + latest jax.
3. Override torchvision resolution to force CPU variant when both `tpu`
   and `vllm` extras are active. Needs uv expertise.
4. Wait for Google to release `vllm-tpu` with jax 0.9.x support.

**Recommendation**: Option 1 for immediate unblocking, option 2 or 4 longer term.

### What's Deployed on the Fork

| Commit | Branch | What |
|--------|--------|------|
| `690e8048` | `feat/fast-tpu-bootstrap` | Initial: abstract bootstrap, RNG, KV clamp |
| `03eb4529` | `feat/fast-tpu-bootstrap` | Review round 1: narrow scope, typed config, tests |
| `00f5efc1` | `feat/fast-tpu-bootstrap` | Review round 2: routing allowlist, quantization guard |
| `5620d202` | `feat/fast-tpu-bootstrap` | Style: isort/yapf formatting |
| `b3953b2c` | `marin` only | CI: wheel build workflow |
| `a290b83b` | `marin` only | Style: license header fix |
| `39d57610` | `marin` only | Fix: strip install_requires |
| `e718c30d` | `marin` only | Fix: add tpu-info dep back |
| `2d9ddeea` | `marin` only | Fix: add torchax, flax, qwix, runai deps |

### Iris Jobs Run Today

| Job | Config | Result |
|-----|--------|--------|
| `/ahmed/fast-bootstrap-baseline` | default loading, v6e-4 us-east1 | Failed: `vllm` binary not found (ran locally) |
| `/ahmed/fast-bootstrap-test` | dummy + tpu_bootstrap, v6e-4 us-east1 | Failed: same |
| `/ahmed/fast-boot-baseline-v3` | default, v6e-4 us-east1 | Failed: `tpu_inference not found` (empty install_requires) |
| `/ahmed/fast-boot-test-v3` | fast bootstrap, v6e-4 us-east1 | Failed: same |
| `/ahmed/fast-boot-debug` | `import tpu_inference`, v6e-4 us-east1 | **Succeeded**: tpu_inference imports OK |
| `/ahmed/fast-boot-debug2` | `from vllm.platforms.tpu import TpuPlatform` | Failed: cascading import error |
| `/ahmed/fast-boot-debug3` | print `vllm/platforms/tpu.py` source | Succeeded: showed vllm expects `from tpu_inference.platforms import TpuPlatform` |
| `/ahmed/fast-boot-debug4` | full traceback of TpuPlatform import | Failed: `No module named 'torchax'` |
| `/ahmed/fast-boot-baseline-v5` | default, v6e-4 (after adding torchax dep) | Failed: `torchvision::nms does not exist` (CUDA torchvision on TPU) |
| `/ahmed/fast-boot-test-v5` | fast bootstrap, v6e-4 | Failed: same torchvision issue |

### Key Learnings

1. `vllm-tpu` is not just stock vllm + tpu-inference bundled. It's a custom
   build with GPU deps stripped. You can't trivially replace it with stock vllm.

2. The `tpu-inference` fork's `install_requires` must include all runtime
   imports that stock vllm doesn't provide (torchax, flax, qwix, etc.) but
   must NOT include deps that conflict with Marin pins (jax, gcsfs, torch).

3. uv lockfile conflict resolution between extras is complex. When both `tpu`
   and `vllm` extras are active, the stricter dep (vllm's CUDA torchvision)
   can override the correct one (tpu's CPU torchvision).

4. The `_VLLM_PREFERRED_ARCHITECTURES` flag for Qwen3MoE was added by a
   Google engineer with no perf justification in the commit message. Both
   JAX and PyTorch paths compile to XLA on TPU.

5. Upstream `tpu-inference` main is developed against unreleased vllm. The
   latest released `vllm-tpu` (0.13.3) uses jax 0.8.1. jax 0.9.2 is only
   on unreleased main.

---

## 2026-03-22 — `abstract_load` Bootstrap Mode (Real Weight Loading)

### Problem

The `abstract_dummy` mode added in the previous session creates an abstract
model via `nnx.eval_shape()` and returns immediately — no weights, no JIT.
This works for RL (caller injects weights via `_sync_weights()`), but for
`vllm serve` (evaluators, smoke tests) we need the fork to create the
abstract model fast AND then load real weights.

The existing `else` branch in `_get_nnx_model()` already does
`eval_shape → load_weights → JIT`. The fix is a control-flow change to
make this path available via `additional_config`, not a new loader.

### Changes to Fork (`feat/fast-tpu-bootstrap-v0.13.2`)

#### Commit `c24fb86e` — feat: add abstract_load bootstrap mode

1. **`TpuBootstrapConfig`** — added `"abstract_load"` to valid values

2. **`_resolved_bootstrap_mode()`** — replaced boolean
   `_use_abstract_dummy_bootstrap()` with a mode-resolver returning
   `"default"` / `"abstract_dummy"` / `"abstract_load"`. Non-default modes
   now **raise ValueError** instead of silently falling back when conditions
   aren't met (unsupported arch, quantization, wrong load_format).

3. **`_build_abstract_model_and_load_weights()`** — extracted the existing
   `else` branch (eval_shape → load_weights → JIT) into a reusable helper.
   Added `try/finally` for `model_weights_iterator` cleanup (was bare
   `del` before — would leak on exception).

4. **`_get_nnx_model()` refactored** to mode-based dispatch:
   - `abstract_dummy` → RL path (abstract model only, no weights)
   - `abstract_load` → fast serve path (abstract + real weights + JIT)
   - `default` + `dummy` → existing concrete random-init (unchanged)
   - `default` + non-dummy → delegates to same helper

5. **No changes to `tpu_runner.py`** — `_init_sampling_rng()` checks
   `== "abstract_dummy"` by string, so `abstract_load` uses normal RNG path.

6. **Tests** — `TestTpuBootstrapConfig` (abstract_load parsing),
   `TestResolvedBootstrapMode` (8 tests for all mode/error combos),
   `TestAbstractLoadBehavior` (4 behavioral tests + 1 dispatch test)

#### Commit `5741def2` — fix: truthiness check + dispatch test

- Changed `hasattr(model_config, "model_weights")` to
  `getattr(model_config, "model_weights", "")` truthiness check (more
  robust if vLLM ever adds a default empty-string attribute)
- Added `test_get_nnx_model_dispatches_abstract_load` — patches
  `_resolved_bootstrap_mode` and `_build_abstract_model_and_load_weights`
  at module level, calls `_get_nnx_model()`, asserts the sentinel from the
  helper is returned (not an early-return bare abstract model)

### Marin-Side Changes (`vllm_load_fast` branch)

#### Commit `9324e95d0` — feat: abstract_load bootstrap + cleanup + wheel update

- Updated `pyproject.toml` wheel URL to `marin-5741def2`
- `uv lock --upgrade-package tpu-inference && uv sync`
- Deleted dead monkeypatch code, simplified evaluators and smoke test

### Verification — Iris Smoke Tests (2026-03-22 04:20 UTC)

Both jobs used Llama 3.1 8B Instruct from GCS, v6e-4 TPU, 64GB host RAM,
`max_model_len=4096`, native mode.

| Job | Mode | Zone | Time | Output |
|-----|------|------|------|--------|
| `/ahmed/vllm-smoke-abstract-load-v2` | `abstract_load` (new) | us-east1-d | 240.9s | "Accelerated code / TPUs speed through the math / Faster, leaner runs" |
| `/ahmed/vllm-smoke-default-v2` | `default` (old) | us-east5-b | 231.0s | "Accelerated code / TPUs speed through the math / Faster, leaner runs" |

Both produced identical coherent haiku output. ~10s difference is within
cold-start compilation noise.

**Result**: `abstract_load` mode works end-to-end. Existing default path
is not regressed by the refactor.

Note: first attempt at 32GB OOM'd — 8B model (~16GB weights) needs
headroom for KV cache + loading overhead. 64GB is the minimum for 8B.

### What's Deployed on the Fork (Updated)

| Commit | Branch | What |
|--------|--------|------|
| `690e8048` | `feat/fast-tpu-bootstrap` | Initial: abstract bootstrap, RNG, KV clamp |
| `03eb4529` | `feat/fast-tpu-bootstrap` | Review round 1: narrow scope, typed config, tests |
| `00f5efc1` | `feat/fast-tpu-bootstrap` | Review round 2: routing allowlist, quantization guard |
| `5620d202` | `feat/fast-tpu-bootstrap` | Style: isort/yapf formatting |
| `92ceeb0c` | `feat/fast-tpu-bootstrap-v0.13.2` | Fix: read config from additional_config |
| `c24fb86e` | `feat/fast-tpu-bootstrap-v0.13.2` | **feat: abstract_load mode** |
| `5741def2` | `feat/fast-tpu-bootstrap-v0.13.2` | Fix: truthiness check + dispatch test |
| `b3953b2c` | `marin` only | CI: wheel build workflow |

### Next Steps

1. Run compilation-cache test (`--repeat 2 --local-cache-dir`) to measure
   warm-start speedup (should skip XLA compilation on run 2)
2. Integrate with evaluators (`lm_evaluation_harness_evaluator`,
   `simple_evaluator`) — pass `abstract_load` config through engine_kwargs
3. Consider Levanter fsspec loader integration for faster GCS weight
   streaming (current path uses vLLM's default HF loader or RunAI streamer
   at ~53 MiB/s — issue #3768 targets ~154 MiB/s via parallel byte-range)
