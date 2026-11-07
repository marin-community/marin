# 20 new respondents: staged OpenRouter v3 calibration

**Design frozen before paid calls.** New campaign
`biotasks/combined/v3/calibration/openrouter-additional20-v1` uses the **same
immutable 230 eligible v3 tasks** (SHA-256
`b893bd40fb1c6676619b32cd11a180969605b2b69a7a3b5b6df1974ae4bbb8d6`),
policy, grading contract, ShellSim backend, and eight-turn/4,096-token
per-turn limit. It does not edit or resample the original 12 respondents or
any v1/v2 records. The pinned 20 identities and provider tags are in
`openrouter_v3_additional20.json`. The loader fails if any overlaps with the
original panel. Only fixed model IDs (no rolling aliases), tool-capable routes,
disabled fallback, and verified generation metadata qualify.

**Ability mix:** small and mid-size OpenAI OSS, Qwen, Mistral, Llama, Gemini,
DeepSeek, GLM, MiniMax, Cohere, and Nemotron models; a few stronger anchors.
Selection biases toward lower/middle performance compared with the first
mostly frontier-class 12, targeting an approximately 50% *scored* correct
rate. Size and label do not prove ability. Low-ability models can fail the
tool/submission contract altogether; the pilot is a measurement gate, not a
promise of a particular rate. Tool/format failures are scored zero when
terminal; upstream missingness and ungraded nonterminal attempts stay
separate.

**Budget:** a *new* immutable $40 pre-batch incremental OpenRouter key spend
gate, not the old v3 $50 gate. Key usage baseline and panel digest are written
once to `.../openrouter-additional20-v1/budget.json`. Before each batch, the
runner rechecks actual key usage and reserves the larger of $1 or **2× the
first-wave p95** provider-specific token cost projected to that batch size;
never submit if spent + reserve exceeds $40 or available key credit. This is
not a mid-response hard cutoff, so no unbounded parallel batches or automatic
retries. The prior campaign's ~$26 remains accounted separately. Storage and
Iris CPU overhead are not OpenRouter API spend.

**Pilot first:** derive exactly 12 distinct v3 tasks from the immutable first
response matrix (SHA-256
`acf9b5aff6a47230e2e0d0c751554e03969d4af9fbbb22099154db1aed039cda`)
by choosing three each nearest prior scored success rates 20/40/60/80%, ties
by frozen task ID. All 20 models attempt those 12 once, across the same pinned
providers as the full run. Inspect valid-response rates, scored accuracy,
missingness, route identities, and costs **before** full dispatch. The full
run reuses each completed pilot output byte-for-byte and submits only the
remaining 218 items for a model. It never re-generates already completed
responses. A model whose pilot has no verified routed generation is reported as
`PILOT_UNROUTED` and **never** enters full dispatch. A pilot with an
unverifiable generation route (including persistent metadata HTTP 500) is
`PILOT_METADATA_UNAVAILABLE` and also cannot enter full dispatch. The pilot found the
pinned `meta-llama/llama-4-scout` Google Vertex endpoint BYOK-only for this
key: all 12 attempts returned upstream 404 without a generation or API
charge. These are retained as missing, not scored failures. Do not silently
replace a failed provider/model or turn a failed attempt into a new sample;
version any replacement panel separately, with no re-calls for previously
completed model/task pairs.

Private attempts and prospective snapshot root:

```text
s3://marin-us-east-02a/marin4sc/taskforge/task-library/campaigns/biotasks/combined/v3/calibration/openrouter-additional20-v1/
```

**Pilot outcome / revision gate (2026-11-07).** All 20 original candidates
finished 12-item pilots, but only **45/170 scored responses (26.5%)** were
correct; 61 were ungraded and 9 infrastructure-missing. This is far below
the requested ~50% scored rate even accounting for the deliberately
prior-stratified pilot. Scout's route was unavailable, while one MiniMax
response's OpenRouter metadata endpoint persistently returned HTTP 500;
unverified generations stay quarantined. Only **$0.64** of the shared $40
ceiling was spent. **Do not full-dispatch the original 20-model panel.**
Preserve all attempts and choose a versioned effective 20 with fewer
weak/format-failing models. `openrouter_v3_topup16.json` freezes 16
non-overlapping additional candidate routes for another 12-item pilot using
*the same* frozen tasks, policy, and $40 budget baseline. Their pilot is
separate and no original model/task is called twice. Choose exactly 13 new
models only after pilot review, alongside seven retained, previously piloted
models; new full jobs must reuse those 12 pilot generations per model.

This remains a *provisional* research bank. After all 20 batches, freeze a
separate scored snapshot with raw-source hashes and provider evidence; refit
the exploratory Bayesian 2PL and report new slope posterior intervals plus
format and missingness sensitivity. Do not merge old and new matrices by
mutating frozen response files. Do not infer biological ability until
references, task key wording, and format/tool failures have independent
validation.
