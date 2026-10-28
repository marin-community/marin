---
last_modified:
  agent: Aster (WorkBuddy)
  model: session default
  timestamp: 2026-10-27T15:45:00+08:00
  task_id: marin-10042-r15-reviewer-delivery
  change: >-
    Reviewer-facing rewrite. The heuristic split of the post-action messages and
    the local correction narrative are removed; the logprob correspondence is
    stated as token-count agreement with positional correspondence assumed.
---

# Trace -> Preference: Screening Findings (Marin #10042)

**Feasibility screening only.** Nothing is trained, no representativeness is claimed, no execution quality is ranked.

## 1. The question

Marin #10042 asks for same-history local DPO pairs mined from agent traces. Two actions may be compared only when they share the **exact conditioning token prefix** and the **same served tool schema**; execution-level comparison additionally needs a **verifiable equivalent pre-action environment state**. Parser validity, execution success and task success must be recorded separately. Two results up front, because they are easy to conflate:

* **Parser-tier screening is feasible here.** Exact prompt ids, an extractable served schema, provable action spans for 123 of 156 turns.
* **Natural preference pairs are NOT established.** One trace per task id, so there is no second branch under the same history to compare.

## 2. Data source

| Item | Value |
|---|---|
| Dataset | `penfever/exp_rpt_curriculum-hard-qwen3.5-122b-131k-opencode-traces` |
| Revision | `404f3fc` |
| Cohort | `qwen35-122b-131k-opencode` (1 of 4; 142 repos total) |
| Tokenizer | `Qwen/Qwen3.5-122B-A10B-FP8` @ `a099dee` |
| Sample | 22 rows -> 156 completion turns (the whole file) |
| SHA-256 | `25a1fc93a79125da5546e0e5e1e9027a54f9d467bf09334861f8e92095f899dc` |
| License | **none declared** (`cardData.license == None`) |

This cohort is the only one of the four storing literal token ids; the other three have neither an `instruction` column nor token ids, so no action boundary is provable for them.

## 3. Method

Decode each per-turn prompt prefix of each row with the pinned tokenizer to recover the served tool schema; the schema is verified identical across the row's turns and is marked **unavailable** (never silently valid) when a prefix carries no `<tools>` block. Per completion turn, decode the retained native ids, prove the action span, run the syntax parser on the proven block, and compare the recorded logprobs against the proven span position. The span proof has three gates reported separately: `encode(decode(ids)) == ids`; exactly one `<function=...>` block; `encode(pre) ++ encode(action) ++ encode(post) == ids`. The third gate **is** the cross-boundary check -- one token covering characters on either side of a boundary breaks it, after which character-offset spans are a guess. The `conversations` column is never used for offsets: it holds a different serialization with different token boundaries.

**Parser scope.** Five checks on the literal text of one block: one opener, a closing tag, every parameter closed, the function served in that prompt, every parameter declared for it. Not checked: execution, success, parameter values, framing, environment state, action quality. A `valid` verdict is evidence about **format only**.

## 4. Findings

### F1 -- Pairing evidence, and why "no execution column" is the wrong summary

**Provided** (row level, N=22): `instruction` 22/22; exact prompt token ids 22/22; action token ids 22/22; served tool schema **extracted** 22/22; per-token completion logprobs 22/22 (completion side only, **one scalar per token, not top-k**). "Extracted" means a `<tools>` block was found and parsed in every per-turn prefix of every row, and the schema is byte-identical across the prefixes within each row (22/22 rows) -- **not** verification that the extracted schema equals what the engine served, because the export carries no independent record to compare against. A prefix carrying no `<tools>` block (none here) is reported as schema-unavailable, not silently schema-valid. **Missing**: **no execution-result column**; **no pre-action environment-state column**; the task verifier is **episode-level**, not per step.

The traces nevertheless carry post-action text. In this sample `conversations` has **no `tool` role** -- only `user` (155) and `assistant` (156) -- and **133 of 156** assistant turns are followed by at least one non-empty message carrying tool-result characteristics (opencode acknowledgements such as `Wrote file successfully.`, `Exit code:` lines, `ls -l` headers, Python tracebacks, test markers, read-tool envelopes). Those 133 are **following messages**; calling one an *observation* asserts it came from the environment in response to the preceding call, and the data carries no such guarantee. What they are is **unlabeled, unframed**, and limited to the one realised branch. So the gap is structure, not text: a **per-action success/failure label**, a **comparable pre-action environment state**, and a **second branch under the same history**.

The role inventory is an **observation about this sample**, not an invariant: the analyzer accepts both `user` and `tool` as carriers of post-action text and prints what it sees, so a future export with a real `tool` role is handled and reported, not silently miscounted.

**Turn -> message mapping.** Completion turn *k* is matched to the **k-th `assistant` message** of `conversations`, in order. Checked by count (22/22 rows agree, 156 = 156) and by content (the function names the native stream encodes for turn *k* equal the names in that message's JSON envelopes: **156/156**, 0 disagreements, 0 unmapped). A positional shift would break the content check. One row (15) breaks strict role alternation with two consecutive assistant messages; the content check confirms the mapping there too. This content check is on **function names only**: it does not assert that parameter values, the complete action, or the environment state are consistent between the two serializations. The 156/156 therefore measures **self-consistency between the two in-export serializations** -- the native token stream and the `conversations` JSON envelopes -- of the same function-name sequence. It is **not** an independent verification that the engine actually executed those calls; this dataset carries no such record.

### F2 -- Natural format phenomena (156 completion turns)

| Class | Count | Status |
|---|---:|---|
| `clean-parsable-action` | 91 | single block, schema-valid, span proven |
| `action-exceeds-max-tokens` | 32 | proven span over an operator cap; length, not format |
| `text-turn-no-call` | 22 | **not a tool call** -- must not be counted as malformed |
| `multiple-function-blocks` | 8 | **not counted as malformed** -- see below |
| `tokenizer-roundtrip` | 3 | data/tokenization artifact; not attributed to the model |

**Multiple-function-blocks (8 turns): a phenomenon, validity not fully established.** The harness adapter we inspected (a fetched copy of the opencode trace ingester, Copyright The Marin Authors) parses each `conversations`-column message with an inline `<tool_call>` envelope regex and then `findall`, building a **list** of tool calls -- so the **exported envelope format** supports more than one call per message. That is a property of the `conversations` column, not of the native literal token stream, and the adapter never operates on the native `<function=...>` blocks. Whether the native multi-block stream is a supported/valid construct therefore rests on the (unverified) link between the two serializations; the 156/156 function-name agreement is consistent with that link but does not by itself establish it. These 8 turns are therefore reported as a **multiple-block phenomenon; harness validity not fully established** -- not confirmed valid, and not confirmed format errors.

The same adapter expects **one observation message per pending call** and would have fewer following; this export typically supplies one message per turn, so the analyzer counts **9 turns** where the pending-call count exceeds the following-message count -- turns that would trip that rule. That is an adapter/export contract mismatch, not evidence about model behaviour, and is a screening caveat, not a confirmed rejection.

**Zero turns are established as malformed tool calls** under the narrow syntax parser -- the `confirmed_format_malformed` count printed by the script, equal to the `syntax-*` classes only. That count deliberately excludes every case the single parser cannot resolve: span-unprovable turns, the 8 multiple-block turns, schema-unavailable turns, and unmapped gates -- all reported as their own categories, never folded into "malformed". Absence of confirmed positives is **not** evidence the data is clean: truncation mid-call, wrong-but-declared parameter values, state-dependent failures, retries and timing errors would all pass the parser.

### F3 -- Recorded on-policy logprobs vs action spans (descriptive, small n)

**Threshold** `logprob < -1.0`, fixed a priori; **completion tokens only**; pooled counts per region.

| Region | low | total | share | exact denominator |
|---|---:|---:|---:|---|
| `action-span` | 656 | 42,648 | **1.538%** | tokens **inside** the proven span, over the 123 proven-span turns |
| `verified-non-action` | 260 | 5,384 | **4.829%** | tokens **outside** the span, those same 123 turns |
| `unverified-turn` | 235 | 10,044 | 2.340% | **all** tokens of the 33 turns with no proven span |
| **all (aligned turns)** | 1,151 | 58,076 | 1.982% | over the 156 turns that entered the statistics (0 omitted on this sample) |

The **all** row sums only turns that entered the statistics (token-count agreement met). Tokens omitted from the statistics -- misaligned turns (`len(logprobs) != len(ids)`) and turns with no recorded logprobs -- are reported separately by the script and never folded into these totals; on this sample none were omitted, so the all-row total equals the file's full completion-token count only incidentally.

**What the correspondence check establishes.** For 156/156 turns the logprob vector has the same length as the token id list (0 mismatches) -- **token-count agreement, a shape check**. Neither fact establishes that element *i* is the logprob of token *i*: the parquet footer describes types only and the producing code path was not available, so **positional correspondence is an assumption of this comparison, not a verified property**.

**Reading, and how far it travels.** Low-probability tokens are less frequent inside action spans than in the surrounding text of the same turns. These are **on-policy** logprobs -- the generating model's own confidence over tokens it produced -- not a disagreement signal. With one scalar per token and no top-k, nothing here bears on whether a better alternative existed. Low probability is **not** used, and must not be used, as an error label. The 32 `action-exceeds-max-tokens` turns have **proven** spans and **do** contribute to the table; only emitted records drop them. The 512-token cap is operator-set, not a data property.

## 5. Actionable conclusions

**Supported today:** exact-prefix, same-schema **parser-tier** checks, and diagnostics that report each gate separately. Read "22/22" and "123/156" as **availability**, not verified agreement with what was served. **Not supported:** execution-tier pairing, for three independent reasons -- no per-action outcome label, no pre-action environment state, one trace per task id. No preference pair, natural or synthetic, is warranted from this file yet.

**Ordered next steps.**

1. **Reconcile the adapter's observation-pairing rule with this export.** Either emit one observation message per call, or relax the rule. Cheapest item, and it currently drops rows for a reason unrelated to model behaviour.
2. **Get a candidate slice with >= 2 attempts per task id**, or per-step environment results instead of the episode-level verifier. A **screening condition, not permission to pair**: per candidate the prefixes must be equal **token by token** (not merely the same first prompt); the served tool schema identical at both decision points; the **pre-action environment state verifiably equivalent** (the load-bearing requirement for execution-tier claims, absent here); and the contrast must fall on the **action**, not the episode outcome. A second attempt that carries the first attempt's failure in context is a *different history*, not a same-history pair.
3. **Only then** derive a real error taxonomy from harness-side parse failures, and use it to judge any synthetic corruption distribution.

## 6. Limits, and what was NOT done

* **Snowball rescoring was NOT executed.** Differing vocabularies block reusing the stored Qwen ids directly as Snowball input (ids here reach 248,069; Snowball's vocabulary is 128,256). Rescoring the **text** is a different route and is not ruled out -- decode with the pinned Qwen tokenizer, re-encode with Snowball's, then score. **We did not do that**, did not verify that a re-encoded stream preserves the action boundaries this analysis depends on, did not confirm the model is reachable or at what precision, and have **not** estimated memory, runtime or cost. Retained logprobs are indexed per Qwen token, so after re-encoding they are not positionally comparable.
* One cohort, one harness, one Qwen native serialization. n = 156 turns. The 512-token cap is operator-set, not a data property.
* No license is declared on the source dataset; only aggregate counts are printed, never raw record text.
* The synthetic corruption rule (dropping one `</function>`) is deliberately outside this PR: it reproduces none of F1-F3, and its value depends on resembling real errors -- the open question.

## 7. Reproduce

```bash
# from the repository root
export TRACE_PREFS_SAMPLE=/path/to/pf_curriculum_hard.parquet
export TRACE_PREFS_TOKENIZER=/path/to/tokenizer.json
uv run pytest experiments/post_training/trace_preferences/test_trace_prefs.py -q   # unit tests
uv run python -m experiments.post_training.trace_preferences.analyze_traces \
    --parquet "$TRACE_PREFS_SAMPLE" --tokenizer "$TRACE_PREFS_TOKENIZER"
```

The sample and tokenizer are **not** in the repo -- download them from the pinned revision above and point the env vars at your copies; paths are passed explicitly and nothing is fetched at import time. Without the env vars the unit tests still pass and the real-data test skips; it must never fail on a fresh checkout. Environment: Python 3.12, **CPU only**; no GPU, no model run, no API calls. Dependencies: `pyarrow`, `tokenizers`.
