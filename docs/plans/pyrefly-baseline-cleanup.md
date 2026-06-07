# Pyrefly baseline burn-down

Tracking: GitHub #6225 (umbrella #6242). Weaver issue #45.

## Goal

The `.pyrefly-baseline.json` holds **159 suppressed errors** that surfaced when
#6221 enabled interpreter queries. Burn down the bulk by **fixing real issues at
the source** and **clearing whole false-positive clusters cleanly**, without
littering the code with scattered `# pyrefly: ignore` comments. A few documented
suppressions per file is acceptable; dozens in one file is not.

## Baseline composition (159 entries)

| code | count | where |
|---|---|---|
| bad-override | 40 | levanter models (24, config-reuse pattern), rigging fs (5), haliax (4), marin (2), misc |
| bad-return | 30 | zephyr/dataset.py (16), haliax (6), levanter (6), marin (2) |
| unknown-name | 17 | levanter/inference/jit_scheduler.py (all) |
| bad-specialization | 13 | levanter MoE shard_map (8), marin training (4), haliax (1) |
| bad-assignment | 13 | levanter optim/axis-unions (12), marin (1) |
| inconsistent-overload | 8 | haliax (axis/scan/ops/partitioning) |
| bad-match | 8 | levanter store cache/tree_store (jax-tree `Any`) |
| not-callable | 7 | haliax (4), levanter (3) |
| bad-argument-count | 7 | levanter MoE (6), haliax (1) |
| tail | 16 | scattered (bad-function-definition, not-a-type, unexpected-keyword, …) |

## Root-cause findings (verified)

1. **jit_scheduler unknown-name (17)** — pyrefly's `tensor-shapes=true` recognizes
   jaxtyping/haxtyping shape strings, **except a single bare-identifier axis**:
   `ht.i32[NamedArray, "position"]` and `ht.i32[NamedArray, "seq"]` are misread as
   forward-references → `unknown-name`. Multi-token strings (`"seq page"`,
   `"stop_seq position"`) parse correctly. Pure pyrefly limitation, 0 real bugs.
   **Fix:** scoped `[[tool.pyrefly.sub_config]]` disabling `unknown-name` for this
   one file (verified: clears all 17, no config-parse warnings). One documented
   config block beats 17 inline ignores or a blunt file-wide `ignore-errors`.

2. **levanter model-config bad-override (24)** — `QwenConfig(LlamaConfig)`,
   `MixtralConfig(MistralConfig)`, `Gemma2Config(GemmaConfig)`, etc. reuse a parent
   config by inheritance but override `model_type` / `to_hf_config` /
   `hf_checkpoint_converter` / `config` to return their **own sibling types**
   (`type[QwenLMHeadModel]` vs `type[LlamaLMHeadModel]`, `Qwen2Config` vs
   `LlamaConfig`). That is a genuine (intentional) LSP narrowing; mypy already flags
   it and the code carries `# type: ignore[override]`. No clean single source fix
   short of restructuring upstream inheritance (out of scope). **Fix:** targeted,
   documented `# pyrefly: ignore[bad-override]` co-located with the existing mypy
   ignore — 2–4 per model file (within "a few per file").

3. **zephyr/dataset.py bad-return (16)** — every transform method builds a new
   `Dataset[T]` via the constructor (preserving the *source's* element type) but is
   annotated to return the *output* element type (`Dataset[R]`, `Dataset[str]`, …).
   Our code. **Fix:** a private `_derive(*ops) -> Dataset[Any]` builder + `cast` to
   each method's declared return type. Purely type-level; zero runtime change.

4. **levanter MoE shard_map (8 = bad-specialization + bad-argument-count)** —
   `@partial(shard_map, …)`-decorated helpers; pyrefly mis-models the jax
   `shard_map` decorator's TypeVar (`Array` vs `(...) -> Unknown` bound). Library FP.
   **Fix:** one documented `# pyrefly: ignore[bad-specialization,bad-argument-count]`
   per decorated function (≈3 per file in mixtral/qwen3_moe, 1 in grug/_core).

5. **haliax inconsistent-overload (axis.py, scan.py, ops.py)** — `@overload` sets
   whose implementation return annotation is narrower than the union of overload
   returns. Mostly real, source-fixable by widening the impl `-> …` annotation.

6. **haliax reparam bad-override (embedding.py, linear.py)** — base
   `ReparamEnabled.reparam` declared as a bare `AbstractVar`/attribute while
   subclasses use `@property`. Source-fix the base to a `@property`.

7. **Real bugs to verify (do not blanket-suppress)** — flagged by triage, must be
   confirmed against behavior before acting; fix at source if real, else minimal cast:
   - `levanter/data/text/datasets.py:212,230` — returns/calls `_create_lm_example`
     with apparent arg-count mismatch (possible real bug).
   - `levanter/adaptor/lora.py:158,345` + `inference/utils.py:64` — a value
     (`jax.random.PRNGKey`/`np.asarray`) used in a type position (`not-a-type`).
   - `haliax/partitioning.py` `almost_shmap(...)(args, kwargs)` unpacking — verify
     before "fixing"; the triage agent's claim is unconfirmed.
   - `marin/web/convert.py:90` — dict may carry `None` values vs `dict[str,str]`.
   - `marin/rl/rl_experiment_utils.py:250,251` — `replace(cfg, tokenizer=…,
     attn_backend=…)` on a too-broad `HFCompatConfig` type.

8. **Library/stub FP tail** — jax-tree `bad-match` on `Any` (store/cache,
   tree_store), haliax-axis-union `bad-assignment` (flash_attention, loss),
   `NamedArray.__eq__/__ne__` numpy-semantics `bad-override`, `MarinTokenizer`
   `unsafe-overlap`, etc. Minimal documented suppressions; a few per file.

## Workstreams (parallel sub-agents, file-disjoint)

Each agent edits a disjoint file set, verifies its own entries cleared with
`uvx pyrefly@1.0.0 check --only <code>` filtered to its files, and runs the
nearest test target for any source/behavior change. The main session owns
`pyproject.toml` and the final `--update-baseline` + full verification.

- **WS-zephyr** — `lib/zephyr/src/zephyr/dataset.py` (16). `_derive` + casts. Run zephyr dataset tests.
- **WS-marin** — `lib/marin/**` our-code (12): ExtractionConfig inheritance, convert.py return typing, rl_losses protocol, arrow_flight contract (+base/checkpoint/train_worker), training.py TypeVar bounds, rl_experiment_utils verify. Run marin transform/web/rl tests.
- **WS-rigging** — `lib/rigging/src/rigging/filesystem.py` (5). fsspec signature alignment or targeted ignores. Run rigging tests.
- **WS-haliax** — `lib/haliax/**` (26): overload-return widening (source), reparam base fix (source), einsum None-default (source), state_dict/parsing returns, + documented suppressions for `__eq__/__ne__`, not-callable, jax_utils kwarg, partitioning (verify the "real bug" first). Run haliax tests.
- **WS-levanter-models** — `lib/levanter/src/levanter/models/*.py` (≈34): bad-override config suppressions (24) + MoE shard_map suppressions (mixtral/qwen3_moe) + flash_attention/loss axis-union + whisper inconsistent-inheritance. Pure type-level.
- **WS-levanter-misc** — `lib/levanter/src/levanter/{optim,store,inference,compat,data,adaptor,analysis,callbacks,main,grug,dpo,eval_harness.py}` (≈30): real-bug fixes (datasets, lora, utils — verify) + documented suppressions for jax-tree bad-match, etc. Run affected levanter tests.
- **WS-config** (main session) — `pyproject.toml` sub_config for jit_scheduler unknown-name (17); final baseline regen.

## Verification & exit

1. Agents land edits; each confirms its targeted entries are gone.
2. Main: `uvx pyrefly@1.0.0 check` (no baseline) — confirm total dropped from 159
   to the residual; confirm **no new** error codes/files appeared.
3. `uvx pyrefly@1.0.0 check --baseline .pyrefly-baseline.json --update-baseline`;
   inspect the diff (only intended entries removed).
4. `./infra/pre-commit.py --all-files --fix` clean (ruff/black/license + pyrefly).
5. Targeted tests for every source/behavior change (zephyr, marin, haliax, levanter).
6. Residual entries (genuine upstream-noise that resists clean fixes) stay
   baselined and are summarized in the PR for the policy call in #6225/#6242.
7. Open PR with `agent-generated` label; close weaver #45.

## Outcome

Baseline went **159 → 0 entries** (file now `{"errors": []}`). Resolution split:
~100 fixed at source, the rest via 53 documented inline `# pyrefly: ignore`
(≤6 per file, avg ~2) plus one scoped `sub_config` for jit_scheduler. Real bugs
caught and fixed along the way:

- `marin/rl/weight_transfer`: `GCSCheckpointServer.get_metrics` returned a plain
  dict that `train_worker.py` feeds to `dataclasses.asdict()` → `TypeError` at
  runtime; both servers now return the metrics dataclass.
- `haliax/_src/parsing.py`: removed a dead `None` branch that would have crashed
  the consumer (`len(lhses)`) if ever reached.
- `levanter/adaptor/lora.py` + `levanter/inference/utils.py`: `jax.random.PRNGKey`
  and `jnp.array` (functions) used in type-annotation positions → fixed to
  `PRNGKeyArray` / `jnp.ndarray`.

Refuted: `haliax/partitioning.py:905` `almost_shmap(...)(args, kwargs)` is correct
(the wrapped `inner` takes the packed `(args, kwargs)` positionally) — not a bug.

`uvx pyrefly@1.0.0 check` is fully green; `./infra/pre-commit.py --all-files`
passes. Residual entries: none.

## Non-goals

- Restructuring levanter config inheritance or the upstream-libs exclude policy
  (explicit human call in #6225 — surface, don't decide).
- Relaxing tolerances or disabling high-signal error codes globally.
