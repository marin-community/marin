I've inspected every file in the brief plus the upstream modules they depend on (`launch_delphi_augmented_swarm_tpp40.py`, `launch_delphi_augmented_swarm_3e18.py`, `delphi_tpp40_evaluation_identity.py`, `execution/executor.py`, `execution/step_status.py`, `olmo_base_eval/{run,components}.py`, `datasets/uncheatable.py`), and the v1 manifest/report for the before/after diff.

One caveat up front: this session has no shell, so I could not re-run the 68 tests or recompute the digests. Everything below is from reading the code; I flag one place where that distinction matters.

---

# NO-GO

Two production blockers. The scope change to `run_order=2` is faithfully implemented — I found no stale four-row count check that would let a bad pair through — but one pre-existing defect makes the Uncheatable half of the gate unreachable, and one makes the polling command stop failing closed.

---

## Blocker 1 — The Uncheatable component-order check can never pass on a real result file

`analyze_delphi_tpp40_bridge_acceptance.py:488-489`
```python
component_bpb = payload.get("component_bpb")
if not isinstance(component_bpb, dict) or tuple(component_bpb) != bridge_eval.EXPECTED_UNCHEATABLE_NAMES:
    raise ValueError("Uncheatable component inventory or order changed")
```

The producer writes that dict sorted: `launch_delphi_tpp40_bridge_uncheatable_eval.py:370`
```python
encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
```

`sort_keys=True` applies recursively, so `component_bpb` lands on disk alphabetically. `EXPECTED_UNCHEATABLE_NAMES` (`launch_...:64`) derives from `UNCHEATABLE_SUBSETS` (`experiments/datasets/uncheatable.py:23-31`), which is deliberately **not** alphabetical:

| on disk (`sort_keys=True`) | `EXPECTED_UNCHEATABLE_NAMES` |
|---|---|
| ao3_english | wikipedia_english |
| arxiv_computer_science | github_python |
| arxiv_physics | github_cpp |
| bbc_news | bbc_news |
| github_cpp | arxiv_physics |
| github_python | arxiv_computer_science |
| wikipedia_english | ao3_english |

Exactly reversed. `_read_json` → `json.loads` preserves file order, so `tuple(component_bpb)` never equals the expected tuple.

**Failure scenario:** East5 and Europe both finish all four Uncheatable cells with perfectly matched BPB. `analyze_payloads` raises inside `_validate_uncheatable_payload` for all four, catches at line 794 into `errors`, and `normalized_uncheatable` stays empty. Both `phase_0` and `endpoint` report `observed_pair_count: 0`, `passed: false`. The gate blocks permanently, with the message *"Uncheatable component inventory or order changed"* — which points the operator at a phantom data-integrity problem rather than JSON key sorting.

It fails closed, so nothing bad gets approved. But per Q1 this is the "silently block" case, and it will burn a debugging cycle on live results.

Table-9 is unaffected and shows the correct pattern: `olmo_base_eval/run.py:310` writes `json.dumps(results, indent=2)` with no `sort_keys`, so `tuple(components) != table9_components()` at line 534 holds.

**Fix:** compare as sets at `analyze_...:488`. Order is already re-imposed canonically by `_uncheatable_metrics` (`launch_...:330-341`), which looks components up by name, so the ordered comparison buys nothing:
```python
if not isinstance(component_bpb, dict) or set(component_bpb) != set(bridge_eval.EXPECTED_UNCHEATABLE_NAMES):
```

---

## Blocker 2 — `--allow-incomplete` suppresses genuine numerical failure

`analyze_delphi_tpp40_bridge_acceptance.py:978-986`
```python
loading_is_only_incomplete = all(
    "executor status is" in error or "result is missing" in error for error in loading_errors
)
...
allowed_incomplete = args.allow_incomplete and (loading_is_only_incomplete or missing_idempotence_only)
if not report["production_launch_authorized"] and not allowed_incomplete:
    raise RuntimeError(f"Bridge acceptance failed closed; see {args.output}")
```

When every result is present and valid, `loading_errors == []`, so `all([])` is vacuously `True`. With `--allow-incomplete`, `allowed_incomplete` is `True` unconditionally and the script never raises.

**Failure scenario:** both sides complete; Europe endpoint macro BPB is 0.004 above East5, failing the binding 0.002 mean threshold. An operator polling with the same `--allow-incomplete` command they've been using since Europe was mid-flight gets exit 0. The report JSON does record `numerical_acceptance_passed: false`, but the exit code — the mechanical fail-closed signal, per the module docstring at line 11 and the contract's `decision` clause (`bridge_acceptance_contract_v3.json:81`) — says pass.

That `missing_idempotence_only` exists at all is the tell: its `not loading_errors and ...` guard is dead code, because `loading_is_only_incomplete` already covers that case. The missing non-empty guard is clearly unintended.

**Fix:**
```python
loading_is_only_incomplete = bool(loading_errors) and all(
    "executor status is" in error or "result is missing" in error for error in loading_errors
)
```
`missing_idempotence_only` then correctly carries the all-complete-but-unaudited case, and a numerical failure raises again.

---

## Answers to the six questions

**1. Stale four-row assumptions?** No count check would let a bad one-pair gate through. `_paired_threshold(expected_count=len(BRIDGE_RUN_ORDERS))`, the manifest cell counts (`analyze_...:298-318`), `EXPECTED_UNITS` (`collect_...:27-31`), and `_validate_idempotence`'s `expected_units` (`analyze_...:677-681`) all derive from `BRIDGE_RUN_ORDERS`. No literal `120`/`240`/`260` survives in the three scripts. One stale artifact remains — item A below.

**2. Does East5 reconstruct the canonical trajectory?** Yes, and the v1→v2 diff is exactly this fix. v1 pointed East5 at a fresh experiment (`bridge_acceptance_paths_v1.json:21`, `..._east5_v5p_bridge_v2_20260830/fit_002_run_00002-ca94d0`) — a scientifically different replacement. v2 repoints at `tpp40.EXPERIMENT_NAME` with `DEFAULT_TRAINING_WANDB_GROUP` and `DEFAULT_TABLE9_WANDB_GROUP` (`launch_...:92-94`), resolving to the handoff-verified `fit_002_run_00002-29ef42`.

Table-9 identity also reconstructs exactly, which is non-obvious and worth recording: `olmo_base_eval_step` puts `resources=resource_config` **inside** the step config (`olmo_base_eval/run.py:373`), and `Executor.compute_version` hashes `config` (`executor.py:1313-1326`), so the Table-9 output path *is* resource-sensitive. The bridge builds `ResourceConfig.with_tpu("v6e-8", regions=["us-east5"], zone="us-east5-b", disk="80g")` (`analyze_...:116-121`), byte-identical to `base.TABLE9_EVAL_RESOURCES` (`launch_delphi_augmented_swarm_3e18.py:110`) and to what `--table9-tpu-type v6e-8 --table9-tpu-zone us-east5-b` builds in the frozen command. So `t9_fit_002_run_00002-f412ee` is the canonical historical address, not a variant.

**3. Is 0.002 binding with one pair?** Yes, at phase 0, endpoint, and Table-9. With `len(values) == 1`, `mean_absolute == maximum_absolute == |delta|` (`analyze_...:578-587`), so `min(0.002, 0.005)` governs. A missing pair yields `values == []` and fails on `len(values) == expected_count` before the `None` comparisons — correctly closed. `test_one_pair_mean_threshold_is_binding` covers this. (Unreachable in practice until Blocker 1 is fixed.)

**4. Does the collector prove zero children and an unchanged byte inventory?** Both, genuinely. Zero children: `collect_...:136-138` rejects any row with `parent_job_id == job_id`. Byte inventory: `tree_payload_identity` (`delphi_tpp40_evaluation_identity.py:91-112`) hashes `{relative_path, size, crc32c}` for every object in the tree, and `after_evidence` compares before/after per side (`collect_...:217-218`). The `should_run` short-circuit on `STATUS_SUCCESS` (`step_status.py:190-192`) returns before acquiring the lock or writing status, so an unchanged rerun genuinely leaves `.executor_status` and the lock file untouched — the inventory comparison is not self-defeating. See item B for what is *not* measured.

**5. Do the commands and manifest support the intended sequence?** Yes, with two things that must be in the runbook — see items D and E.

**6. Blockers vs. polish** — below.

---

## High-priority, not blocking

**A. `--max-concurrent` default is out of range** — `launch_delphi_tpp40_bridge_uncheatable_eval.py:667` defaults to `8`; line 678 computes `cell_count = 1 × 2 = 2`; line 679 rejects anything outside `[1, 2]`. `8` was exactly `4 rows × 2 checkpoints` — a direct four-row leftover. Any hand invocation without the flag dies with `--max-concurrent must be in [1, 2]`. Both frozen commands pass `--max-concurrent 2`, so the frozen sequence is unaffected, and it fails loudly. Set the default to `cell_count` (or `2`).

**B. Idempotence unit counts are hardcoded, not measured** — `collect_...:226-229`:
```python
**{
    unit: {"expected_units": count, "skipped_completed_units": count, "executed_units": 0}
    for unit, count in EXPECTED_UNITS.items()
},
```
`skipped_completed_units` is *defined* as `expected_units` and `executed_units` is *defined* as `0`. The analyzer then checks those three fields against the same constants (`analyze_...:697-710`), so that block is tautological — it can only fail if someone hand-edits the evidence file, which is precisely what `test_corrupted_idempotence_count_blocks_authorization` does. The substantive proof (zero children, unchanged inventory) is sound and independent, so the gate's correctness doesn't rest on this. But `bridge_idempotence_evidence_v2.json` will read to an auditor as if the collector counted skipped units. Either derive them from the Iris/executor record or drop them and let the child-count and inventory checks stand on their own.

**C. The tests bypass the boundary where Blocker 1 lives** — `tests/test_analyze_delphi_tpp40_bridge_acceptance.py:194-213` passes in-memory dicts straight to `analyze_payloads`, and `_uncheatable_payload` (line 79-82) builds `component_bpb` by zipping `EXPECTED_UNCHEATABLE_NAMES`, so insertion order is the expected order. Nothing round-trips through `_write_json`/`_read_json`. That's why 68 tests pass over a defect that blocks every real result. Add a regression test that writes a payload with `bridge_eval._write_json` and reads it back before validating.

---

## Polish

- **`EXPECTED_IDEMPOTENCE_EVIDENCE_SHA256 = "UNFROZEN"`** (`analyze_...:53`) means `production_launch_authorized` can never be `True` until a human edits this constant with the collector's printed digest. This is clearly the intended two-step freeze (line 656 names it explicitly), but it is a mandatory manual step absent from the handoff's Q5 sequence. Also note the final gate run must omit `--allow-incomplete`, or Blocker 2 masks it.
- **East5 Table-9 must be produced by the *training* command**, not the Uncheatable one — the Table-9 step lives in `build_launch_artifacts`' `eval_steps`. I could not confirm from the repo whether `t9_fit_002_run_00002-f412ee` already exists; the handoff verifies the training output and both Orbax checkpoints but is silent on Table-9. If it's absent, `east5_bridge_reference_run2_command_v2.txt` will execute a real v6e-8 job on its first invocation and only be a no-op on the second. `before_snapshot` fails closed if it's missing (`collect_...:50-53`), so the sequence is self-protecting — just don't treat the first East5 training invocation as guaranteed idempotent.
- **`_load_frozen_path_manifest` never validates `training_output_paths` length** (`analyze_...:293-322`) though it validates both cell lists. A manifest with the wrong count reaches `result_inventory`'s `zip(..., strict=True)` (`analyze_...:605-609`) and dies with a bare `ValueError` outside any handler. Fails closed but unreadably.
- **East5 path identities aren't pinned to the contract** — `analyze_...:359` and `:435` only check `europe`, because the contract only carries `evaluation_audit_named_europe_paths_sha256` and `runtime_cache_paths_sha256` for Europe. Defensible: the scientifically meaningful check (`validation_payload_identity`, `table9_request_set_identity`) is recomputed live for *both* sides against frozen digests at `:388` and `:396`. Worth a comment so it doesn't read as an omission.
- **Table-9 results aren't re-bound to the current checkpoint** the way Uncheatable is (`analyze_...:746-753` runs `_checkpoint_metadata` only for `kind == "uncheatable"`). `_validate_table9_payload` only compares the path string. The `hf/step-27335` export is covered by `tree_payload_identity` of the training tree during idempotence collection, so the exposure is narrow.
- **`int(parent["exit_code"])`** (`collect_...:140`) raises an uninformative `ValueError` if Iris returns an empty CSV field for a job without a recorded exit code.
- **`_require_uncheatable_caches`** hardcodes "seven" in its message (`launch_...:256`) while checking `len(EXPECTED_UNCHEATABLE_NAMES)`. Correct today; interpolate the length.

---

**Bottom line:** fix Blocker 1 (`set` comparison at `analyze_...:488`) and Blocker 2 (`bool(loading_errors) and ...` at `analyze_...:978`), add the round-trip regression test from item C, and re-run the focused suite. The scope narrowing to `run_order=2` itself is clean — the frozen numerical, provenance, completeness, and idempotence checks are all intact, and the East5 side correctly reconstructs the canonical trajectory rather than replacing it.
