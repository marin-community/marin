# Executable MT-MBPP accuracy (started 2026-09-26)

Accuracy counterparts for the 17 MT-MBPP components of OlmoBaseEval Easy, which have BPB only because
`allenai/multilingual_mbpp` (MBPP's 500 test problems translated by o4-mini) has no tests and its prompts never name
the function a test would call. Calvin's inputs (2026-09-26): start TPU work as soon as possible; translate tests with
his DeepSeek key and DeepSeek-V4.1-Flash (the API exposes it as `deepseek-flash`); push generated data to Hugging Face,
private for now, organized for the paper's code and data release.

## Design

- Prompts: the native OLMo 3 prompt plus one line after each task description, `Function signature: `...``, for
  the three examples and the target (`experiments/domain_phase_mix/mt_mbpp_exec/prompts.py`). Removing the lines
  recovers the native prompt exactly; the runner checks this against the native request set before any TPU work.
- Signatures (`signatures.py`, `build_requests.py`): the declaration of the function MBPP's asserts call, copied
  verbatim from the o4-mini reference (8,352 by name, 145 by closest name or as the only function nothing else calls,
  3 by hand: C task 31, C++ task 143's tuple overload, bash task 384 which reads stdin); bash uses usage lines derived
  from how the reference reads `$1`, `$2`, `"$@"` and stdin.
- Generation: greedy, at most 1,024 tokens, stop at the closing fence, the four 1e21 checkpoints, v6e-4.
- Tests (`translate_tests.py`): DeepSeek translates MBPP's asserts into each language against the reference's
  signature (JSON: `imports` above the solution, `main` after it, and a `stub` returning a fixed wrong value). Python
  uses MBPP's asserts directly. A test is kept only if the reference passes it and the stub fails it in the sandbox.
- Sandbox: `experiments/domain_phase_mix/mt_mbpp_exec/sandbox.py` and `sandbox/Dockerfile` (17 toolchains, local
  OrbStack image; built by a background agent).

## Frozen artifacts

- Requests: `requests/` (manifest digest `77603128a951f725aa1a31be62236d0a21dd7aac7c7a7ed530df838d134f9007`,
  recorded in `NATIVE_PARITY_VERIFIED` after a local check of all 8,500 rows), uploaded to both regions by `--prepare`.
- Plans: `plan_east5.json` (`92388eee...`; Proportional, UniMax-8, MARINER) and `plan_east1.json` (`eede76ed...`;
  Olmix, whose checkpoint copy is in us-east1). Do not edit `evaluate_table9_accuracy.py`, `mt_mbpp_exec/prompts.py` or
  the other pinned files until both full runs finish: the plans pin their hashes and the full release revalidates.
- Hugging Face: `Calvin-Xu/mt-mbpp-exec` (private), from `export_hf.py`; first push d6aa95f (prompts, signatures).

## Jobs

- Canaries submitted 13:01 PDT: `/calvinxu/mt-mbpp-accuracy-v6e4-canary-{east5,east1}-20260926` (Fieldbook
  experiment `exp_01kvvvv6zxrf0j7tkp4f7k6y66`). The east5 parent confirmed parity for all 8,500 documents in region.
- `auto_release.sh` (detached, `auto_release.log`) releases `/calvinxu/mt-mbpp-accuracy-v6e4-full-{east5,east1}-20260926`
  when each region's canary parent succeeds, then waits for both.
- Translation: `tests/translations.jsonl` (resumable; log `tests/translate.log`), started 13:05 PDT with 96 workers.

## Open decisions for Calvin

- Whether the paper's accuracy mean covers all 51 components once MT-MBPP is scored (it is 34 now).
- Python appears twice with different protocols: `mbpp` keeps its native prompt and binds the tested name at grading;
  `mt_mbpp_python` discloses the signature like the other 16 languages.
- `allenai/multilingual_mbpp` states no license; confirm before a public release (MBPP is CC-BY-4.0).
- The HF token has no organization access; a release namespace other than `Calvin-Xu` needs one.

## Progress (2026-09-26 afternoon)

- Tests: 8,000 DeepSeek translations (`deepseek-flash`, thinking effort `high`, 31.2M tokens; 11 redone with a
  65,536-token cap) plus MBPP's own asserts for Python. Validation in `mt-mbpp-sandbox:2277a08ead83` kept 8,076; one
  repair round at effort `max` (4.0M tokens) flagged 296 references as wrong and fixed 90 tests: 8,166 valid (Python
  500, lowest Haskell 453). `tests/validated/` holds every test with its verdict; pushed to `Calvin-Xu/mt-mbpp-exec`
  (cf8b1f2).
- Harness fixes found by validation: trailing success exits are stripped from script-language tests (the sentinel
  prints after them), PHP test code goes inside the solution's `<?php` tag, and a bare Java or C# method is wrapped
  in a class.
- TPU: Olmix's full run in us-east1 started 13:23 PDT. The us-east5 full run started 13:48 PDT but got one v6e-4 slice
  (the zone's 7 preemptible slices were taken and scale-ups failed), so Calvin approved copying the three us-east5
  checkpoints (40.7 GB) to `gs://marin-eu-west4` once (`euw4/copy_checkpoints.py`) and running there.
  `evaluate_table9_accuracy.py` gained `europe-west4` (bucket `gs://marin-eu-west4`, v6e-4 in europe-west4-a); this
  changed the runner's source pin, so `plan_east5.json` and `plan_east1.json` can no longer be resubmitted as frozen
  (their running parents are unaffected). `euw4/auto_release.sh` relocates `plan_east5.json` to `plan_euw4.json` after
  the copy verifies, runs the canary and releases the full run.
- Grading: `grade_loop.sh` grades completed tasks every 15 minutes into `grading/<region>/<checkpoint>/`.

## Results (2026-09-26 20:07 PDT)

All 68 checkpoint-language pairs generated and graded. Olmix ran in us-east1 (plan_east1); Proportional, UniMax-8 and
MARINER finished almost entirely in europe-west4 (plan_euw4: 50 of 51 pairs; one from the cancelled us-east5 run).
`race_watch.py` cancelled the second us-east5 run (plan_east5b) once every pair had completed in some region. The 8
pairs generated in two regions agree problem by problem. `results/` (from `summarize_results.py`) holds pass@1 per
language and the 17-language mean with a bootstrap over MBPP problems:

| Mixture | Mean pass@1 | 95% interval |
|---|---|---|
| Proportional | 4.4% | 4.0-4.9 |
| UniMax-8 | 8.9% | 8.3-9.6 |
| Olmix | 12.5% | 11.8-13.2 |
| MARINER | 15.7% | 15.0-16.5 |

Every pairwise difference excludes zero (MARINER - Olmix +3.2 pp, 2.5 to 3.9); MARINER is highest in 16 of 17 languages
(Go: Olmix 4.8%, MARINER 4.6%). The ordering matches MT-MBPP BPB (0.516 / 0.453 / 0.428 / 0.386). Equal-weight
51-component accuracy means: 30.09 / 32.66 / 35.11 / 36.41 with MBPP's name binding, 29.98 / 32.42 / 34.87 / 36.14
with the original MBPP grader. Calvin decides the paper's accuracy computation. Released to `Calvin-Xu/mt-mbpp-exec`
(4d4f9c4) with every completion and its grade. Fieldbook statuses updated (canaries and the east1 and euw4 full runs
succeeded; the two cancelled us-east5 full runs killed).
