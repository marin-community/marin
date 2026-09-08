Read-only review; nothing in the repository was modified. No shell in this session, so nothing was executed — see "Not verifiable here."

## 1. The fix is sufficient for the observed failure

The limit is client-side and pre-network: `Settings.validate_run_tags` is a pydantic `field_validator` that rejects any tag outside 1–64 characters before `wandb.init()` touches the server (`.venv/lib/python3.12/site-packages/wandb/sdk/wandb_settings.py:1445-1496`). That is exactly the reported error class, and it explains why all 96 children died at the same point with no artifacts written.

New `panel_tag="wsd80_coupled_onset_successor_c2v4"` is 34 characters. I checked every other element of the tag tuple at `experiments/domain_phase_mix/launch_starcoder_wsd80_coupled_onset_refinement_confirmation.py:271-282`, since a sibling tag over the limit would reproduce the failure verbatim:

| tag source | value | len |
|---|---|---|
| `deployment_id` | `central2-v4-coupled-successor` | 29 |
| `stage` | `bayesian_refinement_discovery` | 29 |
| `support_id` | `m100` (`launch_starcoder_wsd80_coupled_onset_dense_surfaces.py:46`) | 4 |
| `arm_id` | `coupled_0p60/0p80/0p90` | 12 |
| `coordinate_id` | `bo_0p60_01` / `c096` (`design_...20260901.py:238,254`) | ≤10 |
| literals | `coupled_phase_lr_onset` longest | 22 |

Max tag width across all 96 rows is now 34. No other W&B field is length-validated: only `run_id` and `run_tags` have validators, and `validate_run_id` (`wandb_settings.py:1291-1312`) checks only emptiness/whitespace/reserved chars — `c2v4r_lrcd_*` (~40 chars, no reserved chars) passes. Levanter forwards `tags` unchanged and appends none (`lib/levanter/src/levanter/tracker/wandb.py:377,451`). `tests/test_starcoder_wsd80_coupled_onset_refinement_confirmation.py:31-33` pins the invariant.

## 2. Nothing but observability metadata moved

Identity is `{prefix}/{name}/{version}` with **no content hash** (`lib/marin/src/marin/execution/lazy.py:21,236-237,312-314`), so a tag cannot fork an output path. The scientific payload is double-hash-checked on load against `79943e36…` (`launcher:80-89`), which matches `release.json:22`, so seeds, mixtures, schedules, 96 run names, and the coupled-onset invariant cannot have shifted without an abort. `_load_release` (`launcher:384-400`) re-verifies the release hash, the full deployment record, and five file digests at launch — any undetected drift fails closed before a single child is emitted.

Independent corroboration that `panel_tag` was the only edit: the failed 65-char tag is precisely `wandb_group` minus its `_20260901` suffix, matching the predecessor's convention (`launch_starcoder_wsd80_lr_onset_dense_surfaces.py:142-143`). `name`, `version`, `wandb_group`, and `run_id_prefix` are untouched, which is what fixes the output root.

## 3. The retry is idempotent over the same 96 identities

- Same paths: `checkpoints/pinlin_calvin_xu/data_mixture/…_central2_v4_20260901/{run_name}/2026.09.01.1`, tag-independent.
- The changed fingerprint is inert. `train_lm` sets no `expected_fingerprint`, so `check_drift` warns at worst and never raises (`execution/artifact.py:380-415`). In fact there is nothing to drift against: `write_record` only runs *after* `_handle.run(config)` returns (`lazy.py:385-400`), and every child raised, so no record exists at those paths.
- `force_run_failed=True` (`launcher:462`) makes `_launch_step` skip the `PreviousTaskFailedError` branch on `STATUS_FAILED` (`execution/step_runner.py:379-380`), and `run_step` short-circuits only on `STATUS_SUCCESS`.
- W&B side is clean too: validation aborts before run creation, so there are no orphan runs, and levanter defaults `resume="allow"` (`wandb.py:381`) if any did exist.

## 4. The command is region-local and stages every required file

Region: `--region us-central2 --zone us-central2-b` and `MARIN_PREFIX=gs://marin-us-central2` line up with `ResourceConfig.with_tpu("v4-8", regions=("us-central2",), zone="us-central2-b")`, and `launcher:458-460` hard-asserts the prefix.

Staging: the bundler runs `git ls-files --cached --others --exclude-standard` (`lib/iris/src/iris/cluster/client/bundle.py:161`), so untracked-but-unignored files ship by default — the design `.json.gz` and `uv.lock` are neither gitignored nor pattern-excluded. The three files that *are* excluded and needed are each re-included, and `_should_stage` treats includes as overrides, not an allowlist (`lib/marin/src/marin/run/iris_run.py:95-98`): the design generator past `exploratory/`, `release.json` past `manifests/`, and the FINAL_RESPONSE past the default `.agents` exclude (`iris_run.py:34`). That is exactly the set `_load_release` digests. `--max-concurrent 96` equals the frozen maximum, the confirmation string is byte-identical to `release.json:4`, the `-retry1-` job name avoids colliding with the failed parent, and excluding `tests/` is safe because no runtime import reaches it.

One housekeeping item, not a blocker: `.agents/handoffs/starcoder_coupled_onset_wandb_tag_retry_cc_review_20260901_RESPONSE.md` exists but is **0 bytes**. It is not hashed into the release and a missing extra-include is silently skipped (`bundle.py:182-186`), so it cannot fail the launch — but write this verdict into it before you submit, or you ship an empty provenance file.

## 5. No canary is necessary

Agreed. Every row failed at the same client-side pydantic check before any device step, so there is no partial state, no split population, and no per-row variance to sample. The fix is a metadata string with a regression test, the widest tag is now 34 of 64 characters, and the 375-row predecessor already exercised this exact launch path on the same cluster, prefix, and TPU type. A canary would only re-prove `wandb.init()`, and a `--runs` single-row canary burns a v4-8 for the full 28,260 steps unless killed — the cost/benefit does not favor it.

## Not verifiable here — no shell

I could not recompute `launcher_sha256` (`2dd0aff…`), `design_file_sha256`, `release_sha256` (`561f2ab…`), or the `uv.lock` digest, nor rerun pytest, pre-commit, `--audit`, or `--dry-run`. Those claims rest on your evidence, not mine. All five digests plus the design double-hash are re-verified by `_load_release`/`_load_payload` at launch and fail closed before children are emitted, so this is a gap in my review's provenance, not a launch-safety hole.

## Record, not blocking

- The quoted error prints the full 65-char tag, but the installed wandb elides tags longer than 43 chars in that message (`f"{tag_str[:20]}...{tag_str[-20:]}"`, `wandb_settings.py:1485-1487`). Either the handoff reconstructed the text or the cluster image's wandb differs from `.venv`. The substance is unaffected.
- `wandb_group` is 74 characters. It has no validator and groups are not tags, so it is not a latent failure — but never reuse that string as a tag in a successor deployment.
- The successor's `load_design` drops the `support_id`/`cell_id` inventory assertion that `lr_only.load_design:153-154` carries. `source._validate_model` covers cell geometry and `support_id` only reaches a tag, so nothing is at risk; re-add for parity when convenient.

No blocker.

APPROVE
