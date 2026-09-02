Read the round-3 PASS review, the generator, launcher, test, and surrounding tree. No shell in this session, so executable checks rest on your reported run.

**Path agreement.** `OUTPUT_PATH` (`design_...20260825.py:36`, `DOMAIN_PHASE_MIX_DIR / "starcoder_wsd80_lr_onset_dense_surface_design_20260825.json.gz"`) and `DESIGN_PATH` (`launch_...:45`, `Path(__file__).with_name(...)`) resolve to the same file — the launcher sits in `experiments/domain_phase_mix/`, and `SCRIPT_DIR.parents[1]` from `exploratory/two_phase_many/` lands there. A repo-wide grep for the design filename returns exactly three code references (generator `OUTPUT_PATH`, generator `ARTIFACT_DIR`, launcher `DESIGN_PATH`), all `.json.gz`; the only other hit is round-1 review prose.

**Gzip and load.** `gzip.compress(serialized, mtime=0)` (`:571`) — CPython routes `mtime == 0` to `zlib.compress(..., wbits=31)`, which emits a zeroed mtime header, so the bytes are timestamp-independent. Reads are `json.loads(gzip.decompress(path.read_bytes()))` in both the launcher (`:143`) and the test (`test_...:26`), matching the already-working `_load_historical_confirmation` (`:109`) against the committed `..._20260811.json.gz`.

**Hash invariance.** `EXPECTED_DESIGN_SHA256` is `99d3af5aff1f62e679c200219e92914df1f12a407e9afe08cd365c0a8079d889` (`launch_...:46`), unchanged from round 3. `design_sha256` is `canonical_sha256(payload)` over the sorted-compact JSON of the dict, and `build_payload()` never reads `OUTPUT_PATH` — it touches only `SOURCE_DESIGN_PATH`, `HISTORICAL_CONFIRMATION_PATH`, and `REPO_ROOT/uv.lock`. The container swap therefore cannot move the digest, which is consistent with `test_design_reproduces_frozen_manifest` asserting manifest-embedded == recomputed == launcher constant. Both upstream inputs still exist in their pinned forms (`..._20260808.json` plain, `..._20260811.json.gz` compressed), so their file digests are undisturbed.

**No stale artifacts.** Only the `.json.gz` exists on disk — no uncompressed twin under `experiments/domain_phase_mix/`. No `manifests/starcoder_wsd80_lr_onset_dense_surfaces_v1_20260825/` directory and no `release.json` for this panel, so nothing was frozen against the pre-gzip file digest. `report.md` is still present in `reference_outputs/`. `.gitignore` has no `*.gz` rule, so the artifact is committable; the gate is 500 KB (`infra/pre-commit.py:319`) with no extension exemptions, and 60,549 bytes clears it.

Two carried-over notes, neither new and neither blocking: `CC_REVIEW_PATH` is still empty and must be written before `--freeze` (`:507`), and `mtime=0` fixes the timestamp but not the zlib OS byte, so a regeneration on a different platform could change the `.gz` file digest — that only matters after `release.json` seals `design_file_sha256`, which is the intended drift alarm anyway. Nothing else changed.

VERDICT: PASS

## Parent CPU audit repair re-review (2026-08-25)

The first live parents failed before spawning children because `_schedule_vector` initialized JAX's advertised TPU backend on a CPU coordinator. The repair calls `_configure_parent_jax()` before any backend-touching audit work, sets the process-local platform to CPU, and fails closed unless `jax.default_backend()` confirms `cpu`. This does not write an environment variable or alter the separately launched `v5p-8` child tasks.

The regression test now exercises both the production auto-discovery path with `JAX_PLATFORMS` unset and an explicit TPU-advertised path in isolated subprocesses. It checks the return code with captured diagnostics and verifies both the CPU backend and frozen schedule length. Focused tests passed 8/8, Pyrefly reported zero errors, targeted pre-commit checks passed, and the complete 500-row central1 audit passed with `JAX_PLATFORMS=tpu`.

The first Opus review blocked resubmission because the old frozen release necessarily had a stale launcher hash and requested stronger regression coverage. After those repairs, a resumed `claude-opus-5` max-effort read-only OAuth review found no code or test blocker and directed us to refreeze and verify `_load_release()` before resubmission. Account preflight was `plambdafour@proton.me` with `stripe_subscription`; `ANTHROPIC_API_KEY` was removed from the review environment.

VERDICT: PASS
