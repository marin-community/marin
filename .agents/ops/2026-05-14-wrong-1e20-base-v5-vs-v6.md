# 2026-05-14 — Delphi midtraining used the wrong 1e20 base (v5 isoflop ablation, not v6 Delphi)

**Severity:** High. Every "1e20" result published in GitHub issue #4547 was trained on a non-Delphi base.
**Status:** Discovered 2026-05-14 in a Discord exchange between Ahmed (ahmeda14960) and Will Held (Delphi lead). Not yet remediated.
**Owner:** Ahmed. Re-run options laid out in §6.

---

## 1. What happened

For every "1e20" cell across three midtraining sweeps (April 10 B, April 20 B, May K=0.20 36-cell), the base checkpoint used was:

    isoflop-3e+20-d2048-L21-B128-adamh_scaling_v5
    gs://marin-us-central2/checkpoints/isoflop/isoflop-3e+20-d2048-L21-B128-adamh_scaling_v5/

This is **NOT** a Delphi compute-optimal model. It is a point from an **older, deprecated isoflop sweep generation (`adamh_scaling_v5`)** using a different optimizer recipe. Will's words: *"A checkpoint from a totally different scaling recipe."*

The canonical Delphi 3e20 ISOFlop-bucket winner is:

    isoflop-3e+20-d2304-L23-B128-adamh_scaling_v6
    gs://marin-us-central2/checkpoints/isoflop/isoflop-3e+20-d2304-L23-B128-adamh_scaling_v6/
    Registered: experiments/exp1337_eval_suite.py:180

1e21 and 1e22 anchors **are correct** — they're the canonical `adamh-scaling-ladder-nemotron-optimal-{1e+21,1e+22}-v5-...` runs which internally use `LABEL = "adamh_scaling_v6"` (`exp1337_delphi_suite.py:62`).

## 2. Side-by-side

| | Used (wrong) | Canonical Delphi 3e20 (right) |
|---|---|---|
| Run name | `isoflop-3e+20-d2048-L21-B128-adamh_scaling_v5` | `isoflop-3e+20-d2304-L23-B128-adamh_scaling_v6` |
| Hidden dim | 2048 | **2304** (+12.5%) |
| Layers | 21 | **23** (+9.5%) |
| Heuristic | v5 (old) | **v6** (sqrt-batch LR, no `/H` on adam_lr) |
| Final step | 46,915 | 35,408 |
| Tokens | 24.62 B | ~18.6 B (different tokens/params ratio at fixed 3e20 FLOPs) |

The v5→v6 change moved compute-optimal at fixed FLOPs to a **wider, fewer-tokens-per-param** point. Different architecture AND different optimizer hyperparameters.

## 3. Root cause — the `v5` triple-meaning trap

Three different things in the repo all surface as the string "v5":

1. **`adamh_scaling_v5`** as an `experiment_name` argument to `create_isoflop_sweep_steps` → deprecated isoflop generation. Bakes into `isoflop-...-adamh_scaling_v5` run names. NOT Delphi.
2. **`adamh_scaling_v6`** as the current `experiment_name` (`isoflop_sweep.py:229`) → canonical Delphi isoflop. Bakes into `isoflop-...-adamh_scaling_v6`. IS Delphi.
3. **`-v5-`** as a hardcoded experiment-iteration tag (`exp1337_delphi_suite.py:232`: literal `f"-v5{suffix}"`) → appears in 1e21/1e22/1e23 optimal-training run names. **Has nothing to do with the heuristic.** Those runs use `LABEL = "adamh_scaling_v6"` (line 62).

So `adamh-scaling-ladder-nemotron-optimal-1e+21-v5-019021` reads as "v5-something" but IS a v6-heuristic run. The 1e20 we used reads as "v5-something" AND IS a v5-heuristic run. Same surface form, different semantics.

## 4. How it got past us — five contributing factors

1. **GCS-grep instead of registry-grep.** A prior session grepped `gs://marin-us-central2/checkpoints/isoflop/` for "3e+20" and picked the v5 entry without consulting the registry. GCS preserves v1 → v8 generations side-by-side; the registry (`exp1337_eval_suite.py:174-180`) explicitly maps each budget → canonical v6 path.
2. **`v5` name overlap masked the mismatch.** The 1e21 and 1e22 runs end in `-v5-019021` / `-v5-025b0e`. Combined with the 1e20's `-adamh_scaling_v5`, the "v5"s looked consistent across scales.
3. **Self-confirming footnote.** The logbook recorded *"User-confirmed substitution"* — but the confirmation was that A substitution was acceptable, not that THIS substitute was right.
4. **No HF-collection check.** https://huggingface.co/collections/marin-community/delphi lists the 7 ISOFlop-bucket winners (3e18 → 3e20) plus headline 1e21/1e22/1e23. A 30-second skim would have surfaced "3e20 winner is d=2304, L=23."
5. **Results self-validated.** Loss curves looked sensible, cross-scale "transfer" looked clean (mix-gap stable at 0.103 / 0.106 / 0.106 across 1e20/1e21/1e22). Nothing in eval space flagged "different scaling family at 1e20."

## 5. Blast radius

**Contaminated:**
- All 1e20 cells across all three sweeps (April 10 B, April 20 B, May K=0.20 36-cell).
- The cross-scale transfer claim: weakened from "1e20 → 1e22 transfers cleanly" to "1e21 → 1e22 transfers cleanly (1e20 evidence is from a different scaling family)."
- The mix-gap stability claim (0.103 / 0.106 / 0.106): the v6-Delphi numbers (1e21, 1e22) still agree at ~0.106; the v5-isoflop 1e20 happening to land at 0.103 is now unexplained, not a confirmation.
- The "sweep cheap at 1e20, deploy at 1e22" recommendation: weakened by the same logic.

**Survives:**
- All 1e21 and 1e22 results.
- The 1e21 → 1e22 scaling claims (math gains compounding +0.03–0.04, retention damage +60% from 1e21→1e22).
- Optimizer-heuristic code for 1e21 and 1e22 in `exp_delphi_true_midtrain.py`.

**Caught before launch:**
- The TRUE-midtrain plan (`exp_delphi_true_midtrain.py:184`) would have re-used the wrong v5 1e20 base. Catching this saved ~$10 in cross-region egress and ~500 chip-h that would have been mislabeled.

## 6. Remediation options

| Option | Effort | Lands what |
|---|---|---|
| **A. Re-label, don't re-run.** Edit logbook §3, eval suite, GitHub issue #4547 comment: "1e20" → "v5-isoflop-3e20 (NOT Delphi)". Add caveat to cross-scale claim. Leave data as-is. | ~1 h | Honest record, weak transfer claim, $0 compute |
| **B. Re-run 1e20 cells on v6.** Stage `isoflop-3e+20-d2304-L23-B128-adamh_scaling_v6/checkpoints/step-30000` (or step-35408) into per-cell output paths. Re-run 3 p33m67 cells (or all 12 for full LR × mix). | 3 cells × ~10 h on v5p-32 ≈ ~500 chip-h; 12 cells ≈ ~2,000 chip-h | Clean Delphi cross-scale signal |
| **C. Drop 1e20 entirely.** Restate sweep as 1e21 + 1e22 only. Update issue + logbooks. | ~1 h | Two-scale signal, cheapest |
| **D. A+B together.** Keep v5 cells AS labeled comparison; also re-run on v6. | Cost of B | Maximum signal: shows the family-cross effect |

Decision pending Ahmed.

## 7. Rules going forward — never re-make this mistake

**For Marin scaling-law / Delphi / isoflop / midtraining work, the source of truth for base checkpoints is ALWAYS one of these — in this order:**

1. `experiments/exp1337_eval_suite.py` EVAL_BASES dict (lines 174-186) — explicit canonical `{scale → GCS path}` mapping.
2. `MARIN_SCALING_SUITES["nemotron-completed-adamh"]` in `experiments/isoflop_sweep.py:227-231` — executor steps that generated the canonical runs.
3. `experiments/exp1337_delphi_suite.py` — produces the `adamh-scaling-ladder-nemotron-optimal-*-v5-*` headline runs.
4. HF collection: https://huggingface.co/collections/marin-community/delphi

**Forbidden:** picking a base by `gsutil ls`-then-grep on `gs://marin-us-central2/checkpoints/isoflop/`. The GCS bucket preserves every deprecated experiment generation; filenames are not safe to read as canonical.

**If the registry doesn't include the scale you want:** that means the canonical experiment doesn't go that low. STOP. Ping Will Held (Delphi lead) before substituting. Do not silently substitute a similar-looking checkpoint.

**The `v5` trap, restated for the next reader:**
- `adamh_scaling_vN` (suffix on `isoflop-...` step names) — the HEURISTIC generation tag. v6 is current; v5 is deprecated.
- `vN-XXXXXX` (suffix on `adamh-scaling-ladder-nemotron-optimal-...` step names) — an unrelated experiment-iteration tag hardcoded in `exp1337_delphi_suite.py:232`. The 1e21/1e22/1e23 optimal-training runs use the **v6** heuristic despite the `-v5-` in their names.

## 8. Code state — what's still wrong

These files still hardcode the wrong v5 1e20 base. They need to be edited before any further 1e20 work:

- `experiments/exp_delphi_math_10b_midtrain.py:193` — the K=0.20 / 10B sweep base
- `experiments/exp_delphi_true_midtrain.py:184, 192-193` — the TRUE-midtrain plan
- `.agents/logbooks/true_midtraining.md` §3 line 72, §4 line 86, §6.2 line 203, §7.3 line 328 — multiple hardcoded references
- `.agents/logbooks/midtraining_delphi.md` — multiple references (use grep `isoflop-3e+20-d2048`)

Each of these needs either (a) replacement with the v6 path, or (b) a clear `# WRONG BASE — see ops/2026-05-14-wrong-1e20-base-v5-vs-v6.md` comment.

## 9. References

- GitHub issue: https://github.com/marin-community/marin/issues/4547 (comments need a follow-up correction)
- Discord exchange: Ahmed ↔ Will Held, 2026-05-13 18:13 PT (transcript captured in conversation log)
- Memory: `~/.claude/projects/-Users-ahmed-code-marin/memory/project_delphi_canonical_bases.md`
- Memory: `~/.claude/projects/-Users-ahmed-code-marin/memory/feedback_verify_base_via_registry.md`
