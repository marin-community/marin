I read all seven listed files plus the Levanter/Marin APIs they depend on. Below are the blockers, then optional improvements, then what I could not verify.

---

## Things I checked and found correct

Stating these so the coverage is clear, since several are the experiment's load-bearing controls:

- **Total-parameter counts and TPP.** `total_parameters` (`starcoder_tpp10.py:113`) sums `ShapeDtypeStruct` leaves of the abstract tree. Independently, `vocab·d + L·(qkvo + 3·d·d_ff + 2d + 2·head_dim) + d` gives 16,587,008 / 8,395,008 (proxy) and 301,241,344 / 268,473,344 (target) — exactly the design table. The `+2·head_dim` term is Qwen3's Q/K RMSNorm over `HeadSize` only (`layers/attention.py:1312-1314`), confirming the design's stated reason for not using `LlamaConfig.total_trainable_params` (`models/llama.py:212`), which undercounts by `L·2·head_dim`. Horizons (633 and 11491 units of 128×2048), TPP (10.0040 / 9.9996), FLOPs (2.491e16 / 6.664e18 via `3 × lm_flops_per_token`), and stage run counts (8 / 3 / 57 / 183 / 187) all reproduce exactly.
- **The allocator audit uses the real mixture key.** `sequence_allocation` derives `mix_key, _ = jax.random.split(PRNGKey(DATA_SEED))` (`starcoder_tpp10.py:292`); Levanter does `data_key = jrandom.PRNGKey(config.data_seed)` then `mix_key, shuffle_key = jax.random.split(key)` (`main/train_lm.py:285`, `data/text/datasets.py:1168`). Identical. Per-block counts are key-independent (`data/mixture.py:143-160`), so multiplying block 0 by `full` is exact and only the final partial block is key-sensitive — correctly sampled at `starcoder_tpp10.py:300`.
- **Shard retry is safe.** `SerialCacheWriter` opens `TreeStore.open(..., mode="w")` (`store/cache.py:959,965`), which maps to `ts.OpenMode(create=True, delete_existing=True)` (`store/jagged_array.py:751-752`). A part that died mid-write is fully truncated on retry, and `__exit__` does not commit a ledger on exception (`store/cache.py:981`). No append/duplication path exists.
- **Zero-weight name/key invariance.** `shuffle_train_sets` consumes one key per component but overrides every named one (`data/text/datasets.py:1253-1262`), and `training_step` renames `selected.name → "starcoder"` (`launch_starcoder_tpp10.py:143-144`) before assigning keys over `weights` only. So the surviving web stream is identical across arms, subsets, and p=0/p=1 component removal. Component insertion order is also identical across arms (6 web, then selected, then validation), so `MixtureDataset.dataset_index` and its argmax-remainder rule match the audit.
- **Endpoint collection.** `StepInfo.step == state.step - 1` (`callbacks/_core.py:76`) and `train()` forces a final hook (`trainer.py:616`), so `total_steps - 1` is right. `eval_metrics.jsonl` lands at `checkpointer.expanded_path(run_id)` with `append_run_id_to_base_path=False` (`training/training.py:195-197`), i.e. `{output}/checkpoints`, matching `LevanterCheckpoint.checkpoint_dir` (`training/training.py:69`). The metric key `eval/<component-name>/bpb` follows from `tagged_eval_sets` (`data/text/datasets.py:1546`) plus `construct_log_dict` (`eval.py:479`). Note the target's final step (11490) is also a multiple of `steps_per_eval` (2298), so that record is written twice — `collect_results`' `v == values[0]` check (`launch_starcoder_tpp10.py:231`) tolerates this correctly rather than rejecting it.
- **Per-subset, not pooled, regret.** `analyze` averages only within a subset (`analyze_starcoder_tpp10.py:99`), aliases matched-at-p=0 to the unmatched observation (`:82`), and keeps the pooled optimum strictly secondary (`:125`). The test's expected `(0.36+0.16+0.04)/3 − 0.09` follows from the code.
- Cross-part sequence reads are handled (`store/cache.py:430-459`), so 2048-token sequences straddling the 49 quota-sliced parts are correct; `gcsfs` supports `"xb"` and maps HTTP 412 to `FileExistsError` (`gcsfs/retry.py:115-116`), so `persist_submission_plan` works on GCS; `StrEnum` arms round-trip through JSON identically, so `plan_sha256` survives the file round-trip.

---

## Blockers

### B1. `--write-design` silently overwrites the frozen manifest

`experiments/domain_phase_mix/starcoder_tpp10.py:341-342`

```python
if args.write_design:
    DESIGN_PATH.write_text(json.dumps(build_design(), indent=2) + "\n")
```

There is no guard. The sibling module has exactly the guard that is missing here (`starcoder_epoch_matching.py:264-266`: *"Existing frozen manifest differs; review and version the experiment instead of overwriting"*). This directly contradicts `spec.md:14` — *"New source identities require a new reviewed design; `load_design` rejects drift rather than silently regenerating it."* `load_design` rejects it; `main` regenerates it.

**Scenario.** `build_design` embeds `source_code_sha256` of `starcoder_tpp10.py` itself (`:220`). Any edit to that file — including the one-line fix for B3 below — makes `load_design()` raise for every entry point. The CLI's own affordance for that is `--write-design`, which rewrites `design.json` with a new `design_sha256`. Because `code_pins()` hashes every file in `ASSETS` (`launch_starcoder_tpp10.py:59`), `design.json` included, every training step's fingerprint changes, and completed artifacts become unusable (see B2). The operator gets no warning that they just re-froze a reviewed manifest.

**Minimal fix.** Mirror `starcoder_epoch_matching.py:264-266`:

```python
encoded = json.dumps(build_design(), indent=2) + "\n"
if args.write_design:
    if DESIGN_PATH.exists() and DESIGN_PATH.read_text() != encoded:
        raise ValueError("Existing frozen design differs; version the experiment instead of overwriting")
    DESIGN_PATH.write_text(encoded)
```

### B2. The training-identity pin is broad enough that no completed stage survives ordinary maintenance, and the failure has no recovery path

`experiments/domain_phase_mix/launch_starcoder_tpp10.py:42-60`, `:170`, `:172-173`, `:213-214`, `:281-284`; `prepare_starcoder_tpp10.py:294`, `:366-367`

`code_pins()` hashes 14 source/lock files plus all 7 asset files, and that whole dict goes into `TrainingRecipe` (`:170`), hence into `step.fingerprint()`, which is then pinned via `expected_fingerprint` (`:173`). The pinned set includes `launch_starcoder_tpp10.py` itself, `prepare_starcoder_tpp10.py`, `starcoder_tpp10.py`, five vendored Levanter modules, two Marin modules, and `uv.lock`.

Changing any byte in that set makes `pending_training_steps` **raise** rather than report:

```python
record = read_record(path)
if record is not None and record.fingerprint != step.fingerprint():
    raise FingerprintMismatchError(...)     # launch_starcoder_epoch_matching.py:164-165
```

That raise happens before the status check, so `collect_results` (`launch_starcoder_tpp10.py:213`) — which is also the stage gate at `:283-284` — cannot even enumerate which runs completed. `check_drift` raises for the same reason (`marin/execution/artifact.py:400-405`).

**Scenario.** The canary completes: three p=1 runs including one target run of 6.66e18 FLOPs (~2.5 h on a v5p-8). Review then calls for any fix in the launcher — the blank-line guard of O1, the `--write-design` guard of B1, a `uv.lock` bump for a security patch. Running `--stage pilot --submit` now raises `FingerprintMismatchError` on all three canary outputs instead of reusing them. This defeats `spec.md:30` (*"They remain part of later stages"*) and `spec.md:36` (*"Stage expansion preserves identities for previously completed primary points"*). The circularity is the sharp part: because `launch_starcoder_tpp10.py` hashes itself, **no edit to the launcher can repair a fingerprint break without deepening it**. The same applies to the CPU caches: `implementation = file_sha256(Path(__file__))` (`prepare_starcoder_tpp10.py:294`) means a docstring change in `prepare_starcoder_tpp10.py` invalidates ~3.8 B tokens of already-materialized cache, and `verify_caches` (`:366`) raises rather than rebuilding.

**Minimal fix.** Keep the *assertion* but drop it from *identity*. `ArtifactStep.runtime_args` exists precisely for this (`marin/execution/lazy.py:188-190`, `:125-134`): values pulled via `ctx.runtime_arg(key)` reach the step function but render as placeholders at fingerprint time. Pass `code_sha256` (and `runtime_versions`) through `runtime_args` so `verified_training` still refuses to run under drifted code and still records it in `verified_runtime.json`, while the artifact's identity stays `(design_sha256, asset digests, locked dep versions, pod config)`. Then relax the `verified_runtime.json` equality check at `:218-222` to compare only the identity-bearing fields.

### B3. The TPU child never checks that the bundled tokenizer resolves, and `VOCAB_SIZE` is never checked against it

`starcoder_tpp10.py:30`, `:40`, `:113-116`, `:159`; `launch_starcoder_tpp10.py:74-89`

Two gaps in the same place.

**(a) The tokenizer path is CWD-relative and unverified.** `TOKENIZER = "experiments/domain_phase_mix/starcoder_tpp10_assets"` is a repo-relative string, and it is the only relative-path tokenizer anywhere under `experiments/` (every other experiment uses an HF id). `load_tokenizer` branches on `os.path.isdir(name_or_path)` (`levanter/tokenizers.py:759`), which is resolved against the process CWD. If the worker's CWD is not the workspace root, it falls through to `_stage_tokenizer` → `snapshot_download("experiments/domain_phase_mix/starcoder_tpp10_assets")` and dies. `verified_training` (`:74-89`) checks host zone, `MARIN_PREFIX`, code hashes, asset hashes and dependency versions — but not the one runtime precondition the whole `pins.json` mechanism exists to guarantee. The failure therefore lands inside Levanter at `config.data.the_tokenizer` (`main/train_lm.py:215`), *after* the TPU is acquired.

Note the relative path is the *right* choice for identity (it is embedded in `preprocessor_metadata["tokenizer"]` via `BatchTokenizer.metadata` at `_batch_tokenizer.py:171`, so an absolute path would make cache metadata host-dependent). The gap is only the missing precondition check.

**(b) `VOCAB_SIZE = 32000` is asserted nowhere.** It drives `total_parameters` (`:115`) and `flops_per_token` (`:159`) — i.e. the entire total-parameter TPP-10 control — but the model actually trained uses `round_axis_for_partitioning(Axis("vocab", len(tokenizer)), ...)` (`main/train_lm.py:313-314`). `pins.json` records `"vocabulary_size": 32000` as a claim that nothing compares against `len(load_tokenizer(TOKENIZER))`. If the bundled `tokenizer.json` ever reports a different size, every quoted parameter count, TPP and FLOP figure is wrong and nothing detects it.

**Minimal fix.** Add to `verified_training`, before `run_levanter_train_lm` (loading a tokenizer touches no JAX backend, so this respects the "no premature initialization" constraint):

```python
from levanter.tokenizers import load_tokenizer
if Path(experiment.TOKENIZER).resolve() != experiment.ASSETS:
    raise ValueError(f"Bundled tokenizer path does not resolve from CWD: {experiment.TOKENIZER}")
if len(load_tokenizer(experiment.TOKENIZER)) != experiment.VOCAB_SIZE:
    raise ValueError("Bundled tokenizer vocabulary differs from the counted model vocabulary")
```

The same two lines belong in `prepare_raw` (`prepare_starcoder_tpp10.py:187-190`).

### B4. `MARIN_PREFIX` is required on the child but never set by this package

`launch_starcoder_tpp10.py:92-93`; `starcoder_tpp10.py:60-71`; `prepare_starcoder_tpp10.py:38`, `:322`

`require_central1` demands exact equality:

```python
if os.environ.get("MARIN_PREFIX") != PREFIX:      # starcoder_tpp10.py:70
    raise ValueError("MARIN_PREFIX must be explicitly set to gs://marin-us-central1")
```

`dispatch_training` submits with no `env_vars` (`:93`), and `create_environment` seeds only `HF_DATASETS_TRUST_REMOTE_CODE`, `TOKENIZERS_PARALLELISM`, `HF_TOKEN`, `WANDB_API_KEY`, `MARIN_CI_DISABLE_RUNTIME_ENVS` (`fray/types.py:632-640`). So the child's `MARIN_PREFIX` comes entirely from Iris cluster configuration this package does not control or pin. A cluster that pins a prefix with any path suffix — the repo's own tests use `gs://marin-us-central1/scratch` (`lib/rigging/tests/test_region_routing.py:131`) — makes every TPU child and every CPU prep job fail immediately after resource acquisition. Note `_train_job`, the normal Marin path, performs no such check, so there is no precedent establishing that the equality holds on this cluster.

This is the one item on this list that the design's own gate is built to catch (`design.md:53`, *"require regional source/cache receipts and a child-runtime canary"*), and it is an empirical question about the cluster, not a logic error. But it costs a TPU acquisition per attempt to discover, and it is a one-line pre-emption.

**Minimal fix.** `remote(verified_training, resources=recipe.pod.resources, env_vars={"MARIN_PREFIX": experiment.PREFIX})`, and the same for `remote(prepare_raw/parent/subset, resources=CPU, env_vars=...)`. If Iris's pinned `task_env` wins over job-level `env_vars`, relax the check instead to compare the bucket rather than the full string.

---

## Optional improvements

1. **`collect_results` will crash on a blank line in `eval_metrics.jsonl`** — `launch_starcoder_tpp10.py:227-229` iterates lines and calls `json.loads(line)` with no guard; `launch_starcoder_epoch_matching.py:333-334` has `if not line.strip(): continue`. The file is rewritten whole on each eval (`eval.py:368-382`), so a partial write is possible. Add the guard.
2. **`verify_caches` ignores the membership digests it was given.** `materialize_indices` writes `indices_sha256` into the step receipt (`prepare_starcoder_tpp10.py:284`), but `verify_caches` (`:364-400`) checks only `design_sha256`, token counts and receipt hash. Comparing `receipt["indices_sha256"]` against `design["subset_indices_sha256"][seed]` / `design["parent_permutation_sha256"]` would make subset membership independently auditable from the receipts rather than only implied by the fingerprint chain.
3. **`audit_allocations` checks web wrap only for the target** (`starcoder_tpp10.py:311-313`). The proxies cannot wrap because `n` is ~18× smaller, but `design.md:30` claims *"no web component wraps"* unconditionally; adding the proxy allocation to the same loop makes the claim self-evidencing at zero cost.
4. **`web_sequences()` hard-codes `11491`** (`starcoder_tpp10.py:139`) instead of deriving it from the computed target steps. Frozen by the design hash so it can't silently drift, but it duplicates a derived quantity.
5. **Artifact names are unnamespaced at the bucket root** — `hq_actual`, `medium`, `low_actual`, `starcoder`, `starcoder_parent` (`prepare_starcoder_tpp10.py:317-323`) resolve to `gs://marin-us-central1/<name>/2026.09.09`. No current collision (existing catalogs use `nemotron_cc/…-llama3`, `dolma/starcoder`), and `verify_caches` would detect one, but a `starcoder_tpp10/` prefix would remove the hazard.
6. **`max_concurrent=len(pending)`** (`launch_starcoder_tpp10.py:289`, `prepare_starcoder_tpp10.py:417`) submits up to 183 v5p-8 jobs in one burst at the dense stage.
7. **Preemptible with zero retries.** `ResourceConfig.preemptible` defaults `True` (`fray/types.py:438`) and `RemoteCallable` defaults `max_retries_failure=0, max_task_failures=0` (`remote.py:60-61`). A preempted ~2.5 h target run fails its Fray job and aborts the whole `run(...)` batch; the next `--submit` resumes from the 10-minute rolling checkpoint, so this is recoverable but noisy.
8. **The read budget is a heuristic.** `limit = min(size_bytes, max(8 MiB, 4 * quota))` (`prepare_starcoder_tpp10.py:110`) bounds *compressed* bytes by 4× the *token* quota. It fails loudly (`:96`), and my estimate leaves 3–5× headroom for both Dolma StarCoder (15.5 MB/shard for 3.88 M tokens) and Nemotron, but a measured per-corpus bytes-per-token from a small probe would remove the guesswork.
9. **`prepare_raw` discards `load_design()`** (`:189`) without comparing `recipe.design_sha256` to the loaded digest. Covered downstream by `verify_caches`, but a one-line comparison fails earlier and cheaper.
10. **`code_pins()` and `runtime_versions()` are recomputed per run** (`launch_starcoder_tpp10.py:97-98`), so `build_plan("dense")` parses `uv.lock` and hashes 21 files 183 times, twice per fingerprint. Hoist them out of `training_step`.

---

## What I did not verify

- **I ran nothing.** No code, tests, shell commands, cloud reads, or job submissions. Specifically, I did **not** execute `load_design()`, so I cannot confirm that `design.json`'s `source_code_sha256` (`e80cb8c4…`, `29dac8e4…`) still matches the current bytes of `starcoder_tpp10.py` / `starcoder_epoch_matching.py`, nor that `pins.json`'s five digests match the bundled assets. If any differ, every entry point raises at `load_design()` before doing anything else.
- **The bundled tokenizer's actual vocabulary size.** I did not parse the 32k-entry `tokenizer.json`. I confirmed only that nothing in the package checks it (B3b). `tokenizer_config.json` shows `bos=<s>`, `eos=</s>`, `pad=null` and no added tokens beyond the three specials, which is consistent with 32000, but that is inference, not verification.
- **The realized epoch discrepancy.** `design.md:26` claims ≤0.0504% across the grid and the test asserts <0.1%. I verified that the audit reproduces the real allocator (matching `mix_key`, key-independent per-block counts) and that only the single trailing partial block is key-sensitive, but I did not compute the realized value; at small p its standard deviation is roughly 0.2%, so the stated maximum is a property of this frozen seed, not a margin.
- **Cluster and deployment facts**: Iris `task_env`/`inject_env` contents (B4), worker CWD (B3a), TPU quota for the burst in O6, whether a nested `ZephyrContext` (`store/cache.py:1337-1344`) runs cleanly inside a Fray CPU job, and whether `lib/levanter/**` is present on workers for `code_pins()`. `create_environment` uses `workspace = os.getcwd()` (`fray/types.py:630`), so all of these hold if and only if the launcher is invoked from the repo root.
- **GCS state**: existence and current generations of the 245 objects in `sources.json`, and whether the 49 StarCoder / 16-per-component selections can actually supply their quotas inside the `4*quota` compressed-byte budget (O8).
- **One upstream ordering question on this package's critical path.** `consolidate_shard_cache_ledgers` zips `probe_results` straight against `shard_cache_paths` (`store/cache.py:1342-1348`), whereas `_distributed_build_cache` explicitly re-sorts Zephyr results by an embedded index first (`store/cache.py:1122`). If `ZephyrExecutionResult.results` is not order-preserving, `field_counts_by_shard` would be attached to the wrong shard names, shifting `_ensure_shard_field_offsets`. I traced the impact — the parts differ in length by at most 1 token, so any misalignment shifts sequence boundaries by ≤5 tokens and leaves the total exact — but I did not read the Zephyr coordinator's result assembly to settle whether the reordering happens at all.