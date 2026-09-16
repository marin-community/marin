No shell in this session, so I verified by reading the artifacts and relied on your reported run for the executable checks (7 passed / pyrefly clean / pre-commit clean). Everything below is from direct reads.

## The two substantive changes

**C4 key correction — confirmed correct, not merely changed.** `SECONDARY_BROAD_METRIC` is `eval/paloma/c4_en-llama3/bpb` in the generator (`design_...20260825.py:56`) and in the frozen payload (`...design_20260825.json:1880`). The decisive evidence isn't just the `paloma/{subset}-llama3` registration at `experiments/datasets/paloma.py:46` — it's `starcoder_wsd80_atomic_metrics.py:21`, the extractor this family uses against real eval JSONL, which raises on any missing key and lists `eval/paloma/c4_en-llama3/bpb` at line 21 next to this panel's primary at line 11. `c4_en` is a real subset (`paloma.py:26`). The unsuffixed `eval/paloma/c4_en/bpb` still appearing in ~40 repo files is pre-`-llama3` legacy and mocked test data; none of it is in this package.

**`p1_primary.scale_disagreement_rule` — coherent with the surrounding contract.** Text matches in generator (`:429-432`) and manifest (`:32`). It keeps the additive contrast primary and uses the log gain's *direction* only as a veto on the "scale-robust increased two-phaseness" wording. That does not collide with `multiple_testing`, which withholds *significance* claims from the log-gain secondary — a sensitivity analysis constraining interpretation without carrying a confirmatory test is consistent. Sign conventions still align: `metrics.scale_invariant_secondary` ("positive favors the untied policy"), the additive estimand, and `p3.gain` ("positive favors two-phase") all point the same way, so "disagreement" is well-defined.

**`release["submitted"]` — gone.** No `submitted` in `_freeze_release()` (`launch_...:509-524`) or anywhere under `*lr_onset*`. The write-only field is removed without disturbing the hash chain: `release_sha256` is still computed over the record with that field blanked, and `_load_release` re-verifies design/launcher/generator/review/uv.lock digests.

## Mutual consistency

Manifest `design_sha256` (`:1809`) and launcher `EXPECTED_DESIGN_SHA256` (`:45`) are byte-identical to the hash you gave. I could not recompute the digest or re-derive `build_payload() == manifest` without a shell — those rest on `test_design_reproduces_frozen_manifest`, which asserts both plus equality with the launcher constant. `DESIGN_VERSION` is `2026-08-25-v2` in both generator and manifest; since it feeds each row's identity hash, every `row_id` changed with the regeneration, and I confirmed no `row_id` or design-hash pin exists outside the four package files. The generated `report.md` is in sync (arm integrals 22466.998 / 25293.0 / 26706.0 / 28118.5, area multiplier 0.88826940 matching `decay_0p60`'s integral). Test count matches: 7 functions in the file. No `release.json` exists under `manifests/starcoder_wsd80_lr_onset_dense_surfaces_v1_20260825/`, so nothing was frozen against the pre-edit hash — the regeneration was free, exactly as round 2 predicted.

## Non-blocking notes

- **One mechanical step remains before `--freeze` will run.** `CC_REVIEW_PATH` (`.agents/handoffs/starcoder_wsd80_lr_onset_dense_surfaces_cc_review_20260825.md`) is currently empty, and `_freeze_release()` requires it to end in `VERDICT: PASS` (`:506`). It must be written before `--freeze` and left unchanged after, since its digest is sealed into `release.json`. Tests are unaffected — the gate test monkeypatches the path.
- The round-2 review's parenthetical that the discovery-BPB difference "sits inside the frozen CI" is marginally off: 0.7880429029 − 0.7819650173 = 0.0060779, just below the 0.006108 lower bound. This changes nothing — that CI is a 5-seed confirmation interval used to judge the *new* `decay_0p80` estimate, while the provenance values are single draws at discovery seed 20260711. Nothing in the generator, launcher, or tests asserts a relationship between them.
- The disagreement rule is scoped to `p1_primary` only, not the m200 replication block. Reasonable, since the headline claim lives on the primary, but it's a scope choice rather than an oversight.
- Carried over unaddressed from round 2: `_validate_runtime_environment()` still hard-fails on any `uv.lock` byte change, a resumption risk across a multi-day launch.

Neither edit introduced a launch-fatal or science-invalidating defect.

VERDICT: PASS
