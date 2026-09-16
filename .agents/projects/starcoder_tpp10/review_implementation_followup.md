## Verdict

No concrete offline blocker remains for preparation or calibration. B1, B3 and B4 are closed in code; B2's actionable defect (unreadable history) is closed, and I accept the retained-provenance design. Below is what I checked, then the live checks that stay open.

## B1 — closed

`write_design` (`starcoder_tpp10.py:259-266`) creates via `path.open("x")`, returns on an identical existing manifest, and raises otherwise; `main:392-393` uses it. `test_frozen_design_cannot_be_overwritten_with_changed_budgets` asserts `path.read_bytes()` is unchanged after the rejected call. The comparison is semantic (`json.loads(...) != design`), so a reformatted-but-equal file is accepted without rewriting — correct for the stated intent.

`design.json:36` now carries `1c7cc500…` for `starcoder_tpp10.py` (was `e80cb8c4…`), so the manifest was regenerated against the revised module. `web_sequences` is now derived from `models["target"]["tokens"]` (`:149,237`) and reproduces the same six values, so the derivation is a refactor, not a budget change.

## B2 — architecture accepted; recovery defect verified fixed

The concrete part of my objection was that a completed stage became *unreadable*, not merely unrepeatable: `collect_results` rebuilt identities from today's checkout and `pending_training_steps` raised before it could report anything. That is fixed. `collect_results(plan, output)` (`launch_starcoder_tpp10.py:217-265`) reads only `request["output_path"]`, `request["fingerprint"]`, `plan["design_sha256"]`, `plan["code_sha256"]`, `plan["runtime_versions"]` — no `build_plan`, no `step.fingerprint()`, no `load_design`. `main:289-291` returns before `load_design`/`build_plan` and before `args.plan_path.write_text`, so the archived plan is never clobbered. `test_collection_requires_success_runtime_receipt_and_final_endpoint:249-281` supplies `design_sha256="historical-design"` and a `code_sha256` naming a file that does not exist in this checkout, drives the real CLI, and asserts `archive.read_bytes() == original`. That is the right test.

Given that, retaining code/data/lock pins in training identity is a defensible position rather than a defect: drift now blocks *new* training while leaving *recorded* results fully recoverable, and `spec.md:38` states the contract and the isolated-checkout restoration path explicitly. Two conditions make it sound, and both hold today: nothing is prepared or trained yet, so the pre-submission revisions cost nothing; and the retained Iris workspace bundle is what makes same-recipe continuation possible. I would keep the bundle-retention requirement in the release checklist rather than only in prose — it is now the sole recovery mechanism for stage expansion, and `code_pins()` (`:44-68`) spans 20 repo files including `uv.lock` and seven vendored Levanter modules. That is an operational obligation, not a correctness blocker.

One consequence worth stating plainly: `--submit` still rebuilds the previous plan (`:307`) and so requires today's checkout to reproduce it. That is the intended policy, not a bug, but it means the "restore the snapshot" instruction is load-bearing for every stage boundary, not just after an incident.

## B3 — closed, and my open vocab question is now answered

`verified_tokenizer` (`starcoder_tpp10.py:139-146`) rejects a CWD that does not resolve `TOKENIZER` to `ASSETS`, then compares `len(load_tokenizer(...))` to `VOCAB_SIZE`. `load_design:255` calls it on the parent/preparation path; `verified_training:90` calls it on the child before `run_levanter_train_lm`. Loading goes `HfBaseTokenizer.from_file` + JSON only, so no JAX backend is initialized — the constraint at `launch_starcoder_tpp10.py:83` holds.

I can now close the vocabulary item I previously left unverified: the bundled `tokenizer.json` has maximum id `31999` and its three `added_tokens` are ids 0/1/2 already inside the vocabulary, so `get_vocab_size()` is 32000. The new assertion will pass, and it now guards the total-parameter TPP control at runtime rather than by inference.

## B4 — closed; and I accept the correction

You are right that I asserted a failure I had not established. `create_environment`'s defaults were the wrong place to look; parent-job env inheritance is the operative mechanism. The added `env_vars={"MARIN_PREFIX": experiment.PREFIX}` on all four wrappers (`launch:102`; `prepare:330,345,366`) makes it explicit without weakening `require_central1` — the zone check and exact-prefix equality at `starcoder_tpp10.py:69-72` are unchanged.

## Spot checks on the other changes

- `PRIMARY_METRIC` still matches: namespacing moved training data to `starcoder_tpp10/<name>` (`prepare:327,342,363`) but left the eval artifact at `paloma/dolma_100_programing_languages-tpp10`, and `source_names` (`launch:152-154`) maps the namespaced handles back to `hq_actual…low_actual`/`starcoder`, so runtime component keys, `train_weights`, and `train_component_shuffle_keys` are unchanged. Slashes in `ArtifactStep.name` are permitted (`validate_path_segment` rejects only `://`, `..`, and edge slashes).
- `verify_caches:385` compares `receipt["recipe_sha256"]` to `canonical_sha256(asdict(materialized_config(step, prefix)))`. This matches only because the recipes read `ctx.output_path`/`ctx.artifact_path` and never `ctx.prefix`/`ctx.region`, and because the worker's `marin_prefix()` equals `PREFIX` — which `require_central1` pins. Sound, but coupled to that pin.
- Blank-line skipping (`launch:241-242`) skips only whitespace; malformed JSON still raises. Index-hash checks (`prepare:387-392`), recipe/design equality (`prepare:189-192,242,254,267`), the proxy web-wrap check (`starcoder_tpp10.py:361-364`), the calibration gate wired into `--submit` (`launch:309-313`), and `--max-concurrent` defaults (4/8) all read as intended. `calibration_summary` now also requires both finite trainer-seed endpoints per cell (`:305-306`).

## Live checks, not offline blockers

Cache yield under the unchanged `4 * quota` compressed-byte bound; regional runtime facts (worker CWD, MARIN_PREFIX precedence, nested Zephyr consolidation inside the Fray CPU job, `lib/levanter/**` presence for `code_pins`); MuonH feasibility at the 16.6M geometry. The first two are exactly what `prepare --run/--audit` and the three-run canary exist to settle; the third is the calibration screen's job.

## Not verified

I ran nothing. I did not hash any file, so I cannot confirm `design.json:36-38` and `pins.json` match the current bytes — if they do not, `load_design()` blocks every entry point first. I accept your Zephyr ordering finding from source; I did not read `coordinator.py`. I did not read the scientific review, and did not re-derive the realized epoch discrepancy. Test, lint and type results are still refreshing; nothing above is a claim that they pass.