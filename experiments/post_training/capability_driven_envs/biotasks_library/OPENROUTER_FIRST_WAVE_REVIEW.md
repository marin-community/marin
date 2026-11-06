# Offline review of BioTasks v2 OpenRouter first wave

2026-11-06. **No new model requests were made.** Source: the immutable
`openrouter-first-wave-v1/snapshots/complete-1/` response matrix, v2 frozen
TaskSpecs, original/recovery trajectories and the deterministic offline script
`audit_openrouter_first_wave.py`. Analysis ran on the CoreWeave cluster beside
the private S3 data; logs: `/zack/biotasks-first-wave-audit-strict`,
`/zack/biotasks-first-wave-inspect-items`, and
`/zack/biotasks-first-wave-inspect-consensus`. The original frozen scores,
references, solver policies, TaskSpec digests and raw trajectories are unchanged.

## Calling / execution problems

1. **Wrong relative file paths, across the bank.** All 247 tasks have
   `environment_requirements.working_directory=/workspace`. All 627 worker
   resources have paths starting with `workspace/`, and 185 task prompts refer
   to `workspace/...` relative paths. The actual mounted paths are
   `/workspace/data/...` and `/workspace/reference/...`; from the solver's
   current directory `/workspace`, `cat workspace/data/...` instead looks for
   `/workspace/workspace/data/...`. The first adduct-correction trajectory
   demonstrates this failure directly; the model then discovers the correct
   absolute path. Across the selected 2,964 trajectories, **1,501** had a tool
   response with `/workspace/workspace/`; **527** of those eventually scored
   correct despite the wasted step. The defect is in task path instructions
   relative to the runtime working directory, not missing resource files.
   Investigate whether changing the working directory to `/` or rendering
   absolute paths is the correct versioned fix; do not silently change the
   frozen v2 execution.
2. **Bare-JSON contract dominates scores.** Of the selected attempts, **1,633**
   had a `JSON value submission is malformed` grade. A conservative offline
   JSON-object scan of final assistant text found **444** malformed submissions
   containing a JSON candidate equal to the frozen reference (same JSON keys,
   numbers comparable as numbers, booleans strictly typed). This is *not* a
   regrade: a response may contain prose or multiple candidates, and the
   frozen answer format demands exactly one bare JSON value. In particular,
   Sonnet has **159** such reference-matching embedded candidates but only
   **6/247** officially correct; Haiku has 86, DeepSeek 79, Mistral 56, and
   Qwen 52. Illustrations: the protein-design task has the correct answer
   embedded in Sonnet's prose; on adduct correction, DeepSeek's JSON candidate
   matches the reference while the frozen submission is malformed. The task
   gives only the generic additional prompt “Give your final answer as one JSON
   value, without Markdown fences.” A later study should version a clearer,
   provider-validated submission mechanism or a deterministic, reference-blind
   extractor, and run a cheap smoke **before** any new full matrix. Do not
   reinterpret these v2 zeros as scored successes.
3. **Length/tool failures are not uniform.** Among 703 ungraded selected trials,
   88 remained provider 429s and 17 Gemini responses lacked token-usage data;
   428 stopped at `max_turns`, 71 at `length`, 153 at `tool_calls`, and 51 at
   `error` (these stop groups include the 105 infrastructure-missing trials).
   The 99 `ValidationError` failures in rollout grading include malformed
   `tool_calls.0.function.arguments` from provider outputs, notably Llama;
   they need model-vs-provider attribution before being treated as biology
   failures. Do not impute ungraded outcomes as zero for IRT without a policy.

## Four newly independently demonstrated task defects

These four are in the **245 GLM-audit-confirmed but unverified** v2 items, not
among the 58 excluded disputed items. Consensus is a *triage signal*, not the
proof; each finding below is verified directly against the frozen prompt and
solver-visible resources.

| Frozen task ID (prefix) | Independent check | Defect |
|---|---|---|
| `biotasks-microarray_preprocessing-a8d21e94279b2b7d` | Quantile-normalized log2 values are ranks 1, 2, 3, 4, reassigned to each probe's original position. P2 is 1 in S1 and 4 in S2. | `max_abs_diff_between_samples` is **3**, not frozen **0**. Nine models produced the former full answer; six bare-JSON submissions were scored wrong. The task's own reference note says quantile normalization equalizes *distributions*, not each probe across samples. |
| `biotasks-cigar_arithmetic-212da74cd61ea63e` | r3 CIGAR `25M3I22M`: query width 25+3+22=50; reference span 25+22=**47** because I does not consume reference. | Frozen `r3.ref_span=50` is wrong; seven models submitted/extracted 47, four bare-JSON submissions scored wrong. |
| `biotasks-variance_stabilizing_transformation-33390457abb3578f` | With `a0=0.01`, `a1=3`, `x=1000`, high-precision Decimal sqrt/ln gives `u=10.2186939728...`, `log2(x)=9.9657842847...`, difference `0.2529096882...`. Half-up to three decimals is **10.219** and **0.253**. | Frozen `u_1000=10.218` and `diff_1000=0.252` are both wrong. Eight models converged on the corrected pair; six bare-JSON submissions were scored wrong. |
| `biotasks-alpha_diversity_metrics-eb3a136672356774` | For S2 probabilities `.9,.05,.05`, `exp(Shannon)=1.4834904028...` and `1/sqrt(sum p²)=1.1076975512...`, hence **1.4835**, **1.1077**; S4 analogues are **3.1842**, **1.5523**. | Frozen S2/S4 fields differ at four decimals; eight models converged on the independently recomputed answer and six bare-JSON submissions were scored wrong. **Separate scientific defect:** the task calls `1/sqrt(lambda)` a Hill number of order 2, but the standard Hill q=2 is `1/lambda`, which also contradicts the general Hill formula in the task's own reference note. This requires a scientific-definition decision, not only numeric replacement. |

The adduct-correction, protein-design-metrics, peptide-composition and
unitary-event examples were also spot checked against the instructions and
resources. Their inspected frozen answers were consistent; incorrect scored
submissions in those examples were model arithmetic or malformed formatting,
not evidence of a reference defect. This is not a claim of full validation.

Another **17** provisional tasks (21 total including the four above) have at
least three models producing the same structured nonreference candidate;
these are *leads only*. In particular some may be shared model errors, output
schema drift, or apparent agreement caused by a common path failure. About
36/247 instructions omit at least one literal top-level expected key; this is
a wording review queue, not 36 proven defects.

## Budget-conscious decision

**Do not expand models or fit IRT on the current 247-item matrix.** This bank
has newly verified incorrect frozen references, systematic path friction and
major submission-format confounding. Preserve v1/v2 results as diagnostic
records; never silently regrade or change a frozen digest. Next, (1) check the
other 17 candidate-consensus tasks using deterministic resource recomputation,
then survey remaining task families and instructions; (2) create new versioned
tasks/IDs and a manifest of independent corrections and exclusions; (3) fix the
path and output contract in a versioned solver policy/task cohort and test a
small, price-capped, mixed-model smoke; (4) only then decide if a broader
response matrix is worth paying for. The 58 original disputed items stay out.
