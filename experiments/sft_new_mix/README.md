# SFT new mix

Tracking: [#10788](https://github.com/marin-community/marin/issues/10788).

## Candidate v1 and control

`sources.candidate_sft_sources_v1()` defines the candidate input map for future
materialization. It starts with `all_sft_sources()` and replaces only the four
`nemotron_sft_v3/science_v2/{rqa,so,syn_mcq,vendor}` components. The shared registry
remains the unchanged control. Other source objects, keys, and rough token
weights are preserved. Historical caches and existing training launches are
unchanged.

The v1 science subset is frozen at Hugging Face revision
`1583049c1df02b418e07d14047eb37db9df46eba` in
[open-athena/Nemotron-SFT-Science-v2-GLM-Accepted-2026.11.05](https://huggingface.co/datasets/open-athena/Nemotron-SFT-Science-v2-GLM-Accepted-2026.11.05/tree/1583049c1df02b418e07d14047eb37db9df46eba).
Keep this pin and v1 selection unchanged; future candidate changes get a new
versioned factory. The base registry follows the checkout, so comparisons must
record and use the same code revision for candidate and control.

The dataset is private. Materialization requires Hugging Face credentials with
access to it. Exact serialized text and duplicate rows are retained without
Harmony rendering. The source map normalizes text; it does not consume the
archived token IDs or loss masks directly.

## Selection and provenance

Of 268,164 science rows, 99,453 were accepted: 70,387 unchanged passes and 29,066
approved repairs. The 168,711 excluded rows include holds, failures, reserved
rows, and unprocessed rows; exclusion is not a scientific-error verdict.

The accepted set includes 31,984 model-only audits and 67,469 optional-web audits.
Acceptance does not guarantee factual correctness. The dataset contains the
original records, review evidence, request receipts, and manifest hashes.
It retains NVIDIA attribution and the source's
[CC BY-SA 4.0 license](https://creativecommons.org/licenses/by-sa/4.0/) terms;
see the dataset card for attribution and provenance details.

## Frozen cache preparation

The Exp3 preparation modules bind immutable input manifests, tokenizer bytes,
source loss modes and per-transform retention receipts. `exp3_plan` and
`exp3_build` prepare source ranges; `finalize_shards` applies admitted tool
replacements and Atlas dispositions. `assemble_candidate` combines complete
cohorts, and `publish_candidate` publishes a catalog, supervision sidecar,
natural mixture schedule and a measurement-job spec for an external consumer.
The spec is frozen data, not an executable: no repository code implements or
runs the consumer yet. Publication stays unqualified until that external
consumer and a release checker succeed. These modules do not submit GPU
training.

### Exact OpenCode question repair

`opencode_questions` joins frozen token caches to the original `question` field
from `nvidia/Nemotron-SFT-OpenCode-v1` revision
`556d5237acff203f3e1a0be49428634c3606cda2`. Stage the large raw files through an
Iris CPU job directly into regional S3. Preserve each source file's byte hash,
row count and partitioned raw JSON records in the staging manifest.

Run each phase with an immutable recipe and its SHA:

```bash
uv run python -m experiments.sft_new_mix.opencode_questions \
  --recipe <recipe-uri> --recipe-sha256 <recipe-sha256>
```

The recipe selects `mode=index`, `bridge` or `repair`, an output prefix,
worker count and jobs, plus hashes of the implementation files. The
implementation hashes must cover the raw-row adapter chain —
`chat_normalize.py`, `chat_render.py` and `nemotron_sft_v3.py` — and the stage
refuses to run without them: if those files change between the index build and
the frozen caches, every bridge row would silently become unjoined and the
repair would no-op. Bridge recipes must also declare `min_joined_fraction`;
the bridge fails when joined rows fall below that bound instead of quietly
succeeding with no changes. Index jobs
render each raw row through the frozen adapter and record its UUID, normalized
ID, original question and complete native-content SHA. Bridge jobs cover every
original cache row in contiguous ranges and bind it to the same partition's
raw identities. Equivalent byte-identical raw documents may retain multiple
UUIDs only when they agree on the question. Missing questions, ambiguous
questions and unmatched fingerprints are counted as unjoined.

Repair jobs follow the scripted and actual-tool-gate row ledgers back to that
original bridge. Set `repair_enabled` explicitly on every job. Only a user
message equal to the quoted original question plus zero to three trailing
`<note>` blocks loses its wrapper and question-quote escaping — the
`<note>`-restricted repair rule proposed in
[PR #11193](https://github.com/marin-community/marin/pull/11193), kept local to
this pipeline until that PR exports a shared rule. Note bytes,
unrelated quoted subagent prompts and non-user messages remain unchanged.
Unjoined rows retain their original tokens and masks. Changed rows receive
fresh prediction-indexed assistant masks and must preserve the exact assistant
target-token array. Every retained row is counted as changed, unchanged or
unjoined; repair drops no documents.

Each version writes new caches and receipts. Keep earlier releases intact.
After repair, rebuild the affected Atlas populations and rerun the full
candidate Atlas scan, application, assembly, publication, native consumer and
release checker. Preserve the approved natural mixture schedule and runtime
geometry independently of the changed prompt-token counts.
