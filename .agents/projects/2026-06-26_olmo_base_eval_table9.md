# Marin-native OLMoBaseEval Easy — Table 9 BPB evaluator

Status: in progress (2026-06-26). Author: agent, autonomously per user instruction
("proceed to implement; use best judgment; show live-compute parity evidence").

## 1. Goal & scope

Make Marin/Iris able to evaluate the **OLMoBaseEval Easy Table 9 BPB** suite (51
components) directly, region-locally (us-east5), with parity against (a) official
OLMo-Eval semantics and (b) our existing Stanford-SC (SC) oracle outputs, so we
can trust it for data-mixing scaling validation and retire the SC path.

Non-goals: no dashboard, optimizer, new modeling code, or broad refactor. Reuse
existing Marin eval/checkpoint infra. No runtime dependency on SC.

## 2. Exact BPB metric definition (the parity reference)

Source of truth: official OLMo-Eval (`/Users/calvinxu/Projects/Work/Marin/data-mixture/OLMo-Eval`),
`src/olmo_eval/common/scorers/base.py:221` (`BitsPerByteScorer`) and
`src/olmo_eval/common/metrics/base.py:239` (`BPBMetricInstanceAvg`).

For one instance with gold continuation string `cont` conditioned on context `ctx`:

```
continuation_tokens = tok.encode(ctx + cont, add_special_tokens=False)[len(tok.encode(ctx, add_special_tokens=False)):]
sum_logprob        = sum of natural-log p(token | left context) over continuation_tokens   (teacher forced)
num_bytes          = len(cont.encode("utf-8"))            # gold continuation string, incl. its leading space
bpb_instance       = -sum_logprob / (num_bytes * ln(2))
```

Per-task score = **unweighted arithmetic mean of `bpb_instance` over the task's
instances** (`BPBMetricInstanceAvg` — NOT byte/token-weighted `BPBMetricByteAvg`).

Load-bearing details (all verified against OLMo-Eval source):
- **Gold-only.** For multiple-choice tasks BPB scores only the gold continuation
  (`_select_gold_output` picks `gold_idx`), not all candidates.
- **Continuation boundary** via re-tokenize-and-slice (robust to non-additive
  tokenization), `tokenizer_utils.py:101`. Trailing spaces on `ctx` are moved into
  the continuation *before tokenization* (`tokenizer_utils.py:90`), but the stored
  `output.text` (and thus `num_bytes`) is the **original** continuation, so the byte
  denominator is unaffected by that move.
- **Whitespace join:** continuations carry a single leading space and context has no
  trailing space (`format_rc` → `"Question: {q}\nAnswer:"`, continuation `" {answer}"`).
  The leading space is in both the scored tokens and the byte denominator (consistent).
- **BOS:** prepended to the **context** iff `tokenizer.add_bos_token` is truthy
  (`tokenizer_utils.py:96`). BOS is therefore never scored and never in `num_bytes`.
  **No EOS** is appended in any logprob path.
- **Numerics:** the OLMo-Eval olmo_core provider upcasts logits to fp32 before
  `log_softmax`; the **HF provider (what SC used) does log_softmax in the model's
  loaded dtype**. Marin will use fp32. See risks (§7).

## 3. Table 9 = 51 components, macro = unweighted mean

Authoritative definition: OLMix paper Table 9 ("Details of the BPB evaluation
suite", `tab:eval_suite`) and its faithful re-implementation in SC's offline
`fit_olmo_base_easy_paper_faithful_olmix_300m.py` (`table9_component_order()`).
Verified: `table9_macro_bpb == mean(51 components)` to 1e-15 over 280 oracle rows.

The 51 (in canonical order): 7 `minerva_math_*`; `codex_humaneval`, `mbpp`; 17
`mt_mbpp_*`; `arc_easy`, `arc_challenge`; **4 derived MMLU buckets** `mmlu_stem`,
`mmlu_humanities`, `mmlu_social_sciences`, `mmlu_other`; `csqa`, `hellaswag`,
`winogrande`, `socialiqa`, `piqa`, `coqa`, `drop`, `jeopardy`, `naturalqs`,
`squad`, `sciq`; 6 `basic_skills_*`; `lambada`, `medmcqa`.

- 47 components are leaf tasks scored directly (`bits_per_byte`).
- Each of the 4 MMLU buckets is a **size-weighted micro-average** over its subjects'
  `<subject>_rc` BPB: `bucket = Σ_subject w_subject · bpb(subject_rc)`, weights sum to
  1 per bucket (`MMLU_CATEGORY_WEIGHTS`, copied verbatim from OLMix `aggregate_mmlu`:
  57 subjects = 18 stem + 14 other + 12 social + 13 humanities). MMLU is the **only**
  component not treated as standalone subtasks.
- **Macro = unweighted mean of the 51.** (The paper's prose "52 tasks" / fit
  `obj_weights` use 52 only because the offline fit double-lists a `basic_skills`
  aggregate; Table 9 itself prints 51 and the macro is mean-of-51.)

## 4. Architecture

Three cleanly separated layers; only layer A touches OLMo-Eval, and only offline.

**A. Request-set generation (offline, build-time, OLMo-Eval).** Drive official
OLMo-Eval to export the exact per-instance scoring requests for the 51 tasks as a
**model-independent** artifact (JSONL per task: `{task, doc_id, context,
continuation}`; gold continuation only). SC ran with `--no-save-requests` so these
were never saved and must be regenerated. Validate via per-task instance counts vs
oracle `metrics.json` `num_instances`. Cache to
`gs://marin-us-east5/raw/eval-datasets/olmo_base_eval_table9/<version>/`.

**B. Marin-native scoring kernel (runtime, Levanter/TPU).** Given the loaded model +
tokenizer + a task's request records: tokenize each `ctx`/`ctx+cont` (replicating
OLMo-Eval `encode_context_and_continuation` incl. trailing-space move + BOS rule),
teacher-force, sum continuation logprobs in **fp32**, divide by
`len(cont.encode("utf-8"))`, take the unweighted instance mean → per-task BPB.
Reuses Levanter `LmExample.from_prompt_and_completion` +
`compute_next_token_loss(reduction=None, loss_dtype=fp32)`; uses the exact-UTF-8-byte
denominator (NOT Levanter's per-token byte-table). Checkpoint load via
`load_hf_checkpoint` (handles single `model.safetensors` AND sharded
`model.safetensors.index.json` from `gs://`).

**C. Aggregation + emit.** Collapse 57 MMLU subjects → 4 buckets, assemble the 51
components, compute macro = unweighted mean, log to W&B via the Levanter tracker.

Runtime is an Iris job pinned to `regions=["us-east5"], zone="us-east5-a"`, modeled
on `lib/marin/src/marin/evaluation/trace_labeled_eval.py`.

Modularity (each independently testable): `components.py` (registry + MMLU weights),
`bpb.py` (tokenize/byte/mask/score kernel), `request_set.py` (artifact schema +
loader), `aggregate.py` (MMLU collapse + macro), `metrics.py` (W&B naming),
`run.py` (Levanter/Iris runtime), `generate_requests.py` (layer A tool).

## 5. Metric naming (W&B)

Primary (per user request): `olmo_base_easy/table9/<component>/bpb` and
`olmo_base_easy/table9_macro_bpb`. Also emit SC-compatible
`olmo_base_eval/easy_bpb/<task>/bpb` (the 47 leaf tasks + 57 MMLU subjects) so existing
analyses and writeback are mechanical. Provenance logged: checkpoint URI, request-set
version, task list, git SHA, model dtype.

## 6. Parity plan

1. **Offline kernel-semantics tests (no model):** byte denominator (incl. multibyte /
   leading space), trailing-space move, continuation slice, BOS rule, the bpb formula —
   each asserted against the OLMo-Eval algorithm. Catches the §7 silent-mismatch classes.
2. **Offline aggregation parity (vs SC oracle):** reproduce `table9_macro_bpb` from
   real per-task oracle values (MMLU collapse + mean-of-51). Already verified to 1e-15.
3. **Live end-to-end parity (the headline):** run the Marin evaluator on
   `baseline_proportional` 300m (`LlamaForCausalLM`, single safetensors, oracle in the
   wide CSV) on a us-east5 TPU; compare per-task BPB + macro to SC oracle.
   Targets: macro `<= 2e-4`; components mostly `<= 1e-3`; diagnose any larger diff
   (request mismatch / SC-wrapper bug / dtype / numerical).
4. **Canary:** run `proportional_3e18` and `dsp_effexp_table9_kl0025_3e18` end-to-end
   (no oracle — demonstrates pipeline + W&B writeback + provenance).

## 7. Known parity risks (and the tests that guard them)

- **Request reproduction fidelity** (highest risk): regenerated `(ctx, cont)` strings
  must match what SC scored. Guard: per-task `num_instances` equality vs oracle; match
  OLMo-Eval commit + SC fan-out patch. If irreducible drift remains, document per-task.
- **Softmax dtype** (fp32 Marin vs HF-provider model-dtype SC): expected to sit within
  the `1e-3` component tolerance; if a component exceeds it, re-run that component in the
  SC dtype to attribute the gap. *Prefer official olmo_core fp32 semantics.*
- **`arc/csqa/piqa` accuracy quirk:** unpatched OLMo-Eval emits accuracy for these; SC
  patched arc/piqa and re-ran csqa separately (`*_csqa_bpb_fix`). Marin computes BPB
  uniformly for all 51 — matches Table 9 intent and the curated oracle. Documented divergence.
- **Byte denominator class:** must be UTF-8 bytes of the continuation string, not token
  count and not Levanter's per-token byte table. Guarded by dedicated tests.
- **Aggregation class:** per-task = unweighted instance mean; macro = unweighted mean of
  51. Guarded by tests + the 1e-15 oracle check.
- **BOS/EOS drift:** BOS-to-context iff `add_bos_token`; no EOS. Guarded by tests.
- **Metric-naming drift:** exact key strings asserted.

## 8. Acceptance-criteria mapping

(1) this doc. (2) canary launcher for the two checkpoints. (3) all 51 emitted with
stable names (`components.py` + test). (4) macro = unweighted mean of 51 (`aggregate.py`
+ test + 1e-15 oracle check). (5) parity: §6.1 vs official semantics, §6.3 vs SC oracle.
(6) W&B writeback + provenance. (7) tests for the 7 silent-mismatch classes (§7).
(8) canary + full-run launchers. (9) no SC at runtime (layer A is build-time only).

## 9. Autonomous decisions (review on return)

- Reuse OLMo-Eval's own request construction as a frozen artifact rather than hand-porting
  51 formatters — maximizes byte-parity, isolates parity risk to a small tested kernel.
- Compute BPB for all 51 uniformly (do not replicate the accuracy quirk).
- Parity checkpoint = `baseline_proportional` 300m (the named canaries lack SC oracle on
  disk; they validate the pipeline, not parity).
- Emit both the requested `olmo_base_easy/table9/*` names and SC-compat keys.
