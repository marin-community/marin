# OpenRouter BioTasks item-response experiment (design; no official trials launched)

## Inputs and gating

Freeze the **v2** bank at
`s3://marin-us-east-02a/marin4sc/taskforge/task-library/campaigns/biotasks/combined/v2/snapshots/verified-corrections-1/`.
It contains 305 items, of which two have independently corrected references and
new task IDs/digests, 58 retain disputed references, and 245 received only
model-audit confirmation. Neither a GLM opinion nor Taskforge structural
acceptance is an independent reference check. **Do not launch official panel
trials on the 58 unresolved items.** Before making bank-wide difficulty claims,
independently validate the other 245 as well. One possible staged pilot uses
only the two corrected items plus synthetic *non-panel* tasks to test transport;
it is not an IRT sample.

Keep v1 GLM calibration (182/305 frozen-score correct) as a separate diagnostic
population: its two misgraded items and v1 digests must not be silently rewritten
as v2 trials. Recalibrate GLM on the final, audited v2 item set under the same
solver policy if comparing it as a respondent in the new matrix.

## Provider-neutral fixed policy

Use `calibrate_glm_v1.py`'s RolloutEngine `ChatModel`, Taskforge grading and
ShellSim lowering without changing prompts, tools, task resources or verifier.
Freeze one shared policy: OpenAI chat completions; `max_tokens=4096` per turn,
`max_total_tokens=32768`, eight turns, temperature 0 and top_p 1; one *official*
trial per model/item pair. Provider-specific parameters not supported by a
candidate must be an explicit exclusion, not silently dropped. Tool execution
remains ShellSim, never the unsandboxed host.

Use the OpenRouter key from `~/.openrouter` (`api_key=<secret>`). The file was
restricted to mode 0600. A non-billing `/api/v1/auth/key` call succeeded on
2026-11-06 and reported a **$250 key limit with $250 remaining**, not the
previously contemplated $300 authorization cap. Set campaign circuit breaker
at **$200 spent** unless the owner explicitly changes the budget. Never put the
key in git, task rows, logs, or a workspace bundle; provision a secret to the
workers using the approved authenticated mechanism. Check available credit
before official launch. Credential existence alone is not approval to spend.

## Routing and metadata gates

Use `https://openrouter.ai/api/v1`. Pin the exact OpenRouter `model` ID and one
routed provider per panel row. In every request's `extra_body`, set an explicit
`provider.order` with one provider, `provider.allow_fallbacks=false`, and
`provider.require_parameters=true`; fail closed if the provider or requested
parameters are unsupported. Validate the actual routed provider from
OpenRouter generation metadata for **each** completion ID (not just the chosen
model ID); reject or quarantine mismatches rather than scoring them as model
failures. Preserve request/response IDs, model ID, provider, actual token usage,
price, finish reason, route fields, and any API-side revision. The current
`ChatModel` records `response_id` and usage but does **not** itself check routed
provider or cost; implement and test that adapter before official trials.

A 50-model manifest should be frozen only after inspecting `/api/v1/models` and
provider endpoints for tool support, `temperature`, `top_p`, `max_tokens`,
context length >= 32768, quotas and current price. A public catalog query on
2026-11-06 returned 453 models; catalog availability is **not** evidence that a
particular provider accepts our fixed policy. Select approximately ten small,
15 midrange, 15 strong-open and ten frontier models across many independent
model families and serving providers. Do not count multiple routes or repeated
samples of one model as independent respondents. Freeze manifest JSON with model
ID, provider slug, context window, pricing timestamp, capabilities, cohort,
route policy and hash; do not silently replace withdrawn models.

## Trial execution and analysis

1. Run synthetic non-panel smoke items across **every** selected model/provider:
   check tool calls, bare-JSON submission, stable routing, usage accounting,
   timeout categorization, and ShellSim isolation. Smoke calls are billable;
   estimate and cap their cost separately.
2. Freeze eligible TaskSpec digests, model/provider manifest, policy and code
   revision. Schedule one official trial per pair. At 305 verified items × 50
   respondents this is 15,250 trials; if only a smaller vetted cohort is ready,
   declare its exact item count and interpretability limitations *before*
   starting. Never attempt an official trial of a disputed item.
3. Persist raw trajectories and item-response records in private, immutable
   attempt paths, with request IDs and generation metadata, then publish a
   complete snapshot/manifest last. Infrastructure failure is missing data and
   can be retried with an auditable same-pair replacement; genuine model/tool/
   format failure is zero and never resampled to improve the score. Fail closed
   if spend, route, or trial uniqueness constraints are violated.
4. Analyze raw binary responses (retain full trial data): CTT diagnostics and
   Rasch/1PL first; then regularized hierarchical Bayesian 2PL with model-family
   and provider effects. No unregularized 3PL with ~50 respondents. Treat
   all-correct/all-wrong items as weakly identified, and uncertainty intervals
   as mandatory. Stratify future gVisor and ShellSim populations rather than
   pooling them.

Before implementation/launch: agree on the audited cohort and the actual
allowed spend, finish independent reference checks, pin the 50 exact
model/provider routes, then build fail-closed routing/cost instrumentation.
