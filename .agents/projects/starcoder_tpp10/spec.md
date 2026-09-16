# TPP10 experiment contract

## Sources and identities

The Python entry points live under `experiments/domain_phase_mix/`:

| Module | Contract |
|---|---|
| `starcoder_tpp10.py` | `build_design() -> dict`, `load_design(path: Path) -> dict`, `select_runs(design: dict, stage: str) -> tuple[RunSpec, ...]`, `audit_allocations(design: dict) -> dict` (async), `calibration_summary(plan, values) -> dict`. |
| `prepare_starcoder_tpp10.py` | `data_steps(design: dict) -> dict[str, ArtifactStep[TokenizedCache]]`; `verify_caches(design: dict, steps: dict, prefix: str) -> dict`. Preparation has explicit finite token quotas and independent durable shard receipts. |
| `launch_starcoder_tpp10.py` | `build_plan(design: dict, stage: str) -> (plan, steps)`; `collect_results(plan, output) -> dict[str, float]`; `validate_release(plan, release, cache_audit)`. Collection reads an archived plan without rebuilding current identities. |
| `analyze_starcoder_tpp10.py` | `verified_measurements(plan, csv_path) -> dict[str, float]`; `analyze(plan, values) -> dict`. |

`RunSpec` fixes name, arm, StarCoder percentage, trainer seed, optional subset seed, step count and batch size. Derived tokens equal steps × batch × 2,048. The model's total trainable parameter count includes tied embeddings once and Q/K normalization weights. New source identities require a new reviewed design; `load_design` rejects drift rather than silently regenerating it.

`starcoder_tpp10_assets/design.json` is canonical JSON plus `design_sha256`, covering model counts, token horizons, source/tokenizer pins, parent permutation and subset-index hashes, grids and all run rows. `pins.json` pins the four tokenizer files and selected source inventory byte-for-byte. `sources.json` contains generation-bound regional inputs. The runtime uses the bundled tokenizer path and never fetches an unpinned tokenizer by model name.

## Cache contracts

Each raw component writes exact token quotas across its explicit source objects. The evaluation component consumes the complete finite list of validation objects. Prefix sampling is conditional on these source shards. A part becomes reusable only after a finished Levanter ledger and `receipt.json` agree on its identity and token count. A failed partial part can be reconstructed; completed parts are preserved. Receipts contain the SHA-256 of the written little-endian int32 token stream. Consolidation merges ledgers under the split directory without copying web arrays.

StarCoder first has a finite raw cache, then a physically reordered parent cache with a frozen global sequence permutation. Target and unmatched use that same parent handle. Three materialized caches draw distinct uniform samples without replacement from the parent's sequence indices. Their sorted selected indices, fixed seeds and hashes record exact membership. Training-time shuffling does not select support. Web caches have fixed internal token proportions and 20% headroom plus one allocator block; no arm repeats them.

`verify_caches` requires successful artifact records, matching config fingerprints, finished ledgers, matching tokenizer metadata, and exact source/support lengths. Read quotas, missing pinned generations, missing text, records over 64 MiB, token deficits and changed successful recipes raise errors. There is no whole-corpus builder or remote-source fallback. Cache paths, executor state and checkpoints use `gs://marin-us-central1`; real preparation and training require a us-central1-a host and that explicit `MARIN_PREFIX`.

## Release and reuse

1. Preparation: the CLI writes paths and fingerprints to the requested local JSON by default; `--run` materializes regional CPU caches. `--audit` checks their receipts and ledgers on a regional host. Both live modes publish the completed audit at an immutable GCS URI and print its digest and URI.
2. `calibration`: eight proxy runs at p=0.5, batches 32/128, two trainer seeds, both arms, one matched subset. Four batch-32 artifacts also belong to the pilot. Batch-128 artifacts remain calibration-only.
3. `canary`: three full p=1 runs (target and both proxies), one trainer seed and one matched subset. They remain part of later stages.
4. `pilot`: seven common coordinates, 57 distinct primary artifacts.
5. `dense`: 21 common coordinates, 183 distinct primary artifacts; the four additional calibration artifacts bring the whole program to 187 training artifacts.

Each training command is a dry run unless `--submit` is supplied. Submission additionally requires a release JSON containing `approved: true`, `reviewer`, `stage`, the exact `plan_sha256` and `cache_audit_sha256`. Release JSON is written only after review of concrete data and plans; this package does not generate an approved release. Completion of the preceding stage, including verified final metrics, is required. Before the canary, the launcher also enforces the calibration rule: mean unmatched batch-32 BPB may not exceed batch-128 by more than 0.01. The release check does not require matched selection to improve.

A `TrainingRecipe` wraps the standard pod configuration and fixes design digest, source/asset/lock hashes and expected dependency versions. The TPU process verifies those before entering Levanter, without initializing a JAX backend prematurely, and writes `verified_runtime.json`. Training outputs are reused only after successful status and fingerprint checks. Stage expansion preserves identities for previously completed primary points.

That reuse contract assumes the frozen recipe and source snapshot are unchanged. Drift deliberately blocks new training; source-code and dependency changes are not silently declared equivalent. Archived-plan collection remains available after maintenance and compares outputs with the archived fingerprints and runtime receipts. Restore the reviewed source snapshot in an isolated checkout before extending the same experiment, or explicitly review a new version. Default active concurrency is four preparation artifacts or eight training artifacts, with an explicit CLI override.

## Measurements and analysis

The persisted submission plan contains the complete selected run set, fingerprints, output paths, code/runtime pins, stage and metric. Collection requires one unique finite native primary value at `total_steps - 1` and an exact child-runtime receipt. A measurement CSV has `run_name,step,value,metric,config_fingerprint,plan_sha256,status`. Analysis rejects missing, duplicate, extra, stale or wrong-plan rows.

Primary selection averages the two trainer seeds within each subset, chooses the smallest p among exact tied minima, and evaluates that p on the single target curve. At p=0, the matched arm uses the corresponding unmatched observation. Report each subset's target regret, its difference from unmatched regret, their descriptive mean, all six single-seed matched selections and the two unmatched selections. The pooled-subset optimum is secondary. Raw-loss and excess-over-own-minimum plots retain every measured grid point; interpolating lines do not choose a mixture. The primary endpoint is programming-language BPB, so the result is scoped to that evaluation.

## Exclusions

No training or CPU preparation is submitted by this build/review task. Historical curves, paper results and old run identities remain untouched. The package does not fit response curves, choose a favorable experimental setting, add curricula/cap sweeps, or claim seed-based uncertainty for the target curve.
