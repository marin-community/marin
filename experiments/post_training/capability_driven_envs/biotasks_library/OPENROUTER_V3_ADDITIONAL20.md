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

**Initial candidate mix:** small and mid-size OpenAI OSS, Qwen, Mistral, Llama, Gemini,
DeepSeek, GLM, MiniMax, Cohere, and Nemotron models; a few stronger anchors.
Selection initially biased toward lower/middle performance compared with the
first mostly frontier-class 12, targeting an approximately 50% *scored*
correct rate. As the results below show, the final selection did not achieve
that target. Size and label do not prove ability. Low-ability models can fail the
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
separate and no original model/task is called twice.

**Second pilot complete.** The 16 new routes produced 113/175 scored
correct, with 16 ungraded and one infrastructure-missing. The versioned
`openrouter_v3_effective20.json` selects **seven** initial models and
**13** top-up models, excluding the unroutable Scout, the MiniMax route
with unresolved metadata, the zero-correct Nemotron Lightning pilot,
and two expensive, near-duplicate Qwen anchors. The selected 20 have
121/212 scored correct (**57.1%**) and 121/240 assigned correct (**50.4%**)
on the stratified pilot; their full-bank rate can differ. Retaining the
lower/mid anchors targets ~50% without treating ungraded/infrastructure
failures as zeros in IRT. **$2.455** in total OpenRouter API spend so far
includes both pilots under the shared **$40** ceiling. The full drivers
submit only the 218 non-pilot items for each of these 20; immutable batch
sizes in the selection keep premium reserve checks below the ceiling.

**Full 20-model panel completed 2026-11-08.** The two full drivers succeeded
(`/zack/biotasks-v3-effective20-initial7` and
`/zack/biotasks-v3-effective20-topup13`). The effective-20 frozen snapshot is
under `s3://marin-us-east-02a/marin4sc/taskforge/task-library/campaigns/biotasks/combined/v3/calibration/openrouter-effective20-v2/snapshots/complete-1/`;
`item-responses.jsonl` SHA-256:
`e6f79c5d8556f2d4f163b7819909f3b30431871dfb9bed40ad6d7aebec211408`.
Its manifest records the selection hash, 57 hashed inputs, source panel
identities, and provider evidence; both original panels/pilots remain intact.
The **4,600 assigned pairs** comprise **3,008 correct, 963 scored zero**
(including 337 failed final submissions), **453 ungraded, and 176
infrastructure-missing**. Scored accuracy was **75.8%**, versus 57.1% in
the stratified selected pilot; the hoped-for ~50% full-bank scored rate was
**not reached**. The full bank was substantially easier for these models than
the pilot. OpenRouter API spend since the shared pre-pilot baseline was about
**$29.16**, below the $40 gate (excluding Iris/CPU overhead). No completed
pilot or full batch was called again.

Missingness is concentrated: **164 of the 176 infrastructure-missing pairs**
are on the pinned `mistralai/mistral-large-2512` route; 161 involved upstream
HTTP 429 rate-limiting. That model has only 64 scored pairs and two additional
ungraded pairs. Do **not** convert those missing pairs to scored zeros, silently
retry them, or treat its observed subset as representative. The exploratory
combined IRT results and sensitivity are in `IRT_V3_EXPLORATORY.md`.

This remains a *provisional* research bank. Do not merge old and new matrices
by mutating frozen response files. Do not infer biological ability until
references, task key wording, and format/tool failures have independent
validation.
