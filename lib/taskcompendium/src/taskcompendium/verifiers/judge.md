# TaskTrove judge support

TaskCompendium has a source-independent `judge` verifier kind that reuses the TaskTrove rubric scorer.

## Current contracts

TaskCompendium schema 0.16 represents the model-visible prompt as `ConversationInput`, separates answer extraction through `SubmissionConvention`, and stores verifier configuration in `VerifierSpec(kind, parameters_json)`. `grade_answer` extracts once and passes the submission and full `GradingAttempt` to a verifier. Harbor's `SemanticVerifier` records infrastructure failures and returns no reward when grading fails.

`tasktrove-verify` defines `Mode.JUDGE` and `JudgeSpec`. Its `reference` rubric prompts for 0, 0.5, or 1 and has an optional normalized exact-match gate. Its `checklist` rubric asks one yes/no question per criterion, prompts for 0 or 1, and returns the fraction passed. Both can run IFEval constraints first. It uses temperature-zero OpenAI-compatible chat calls, retries once when a response has no parseable score in [0, 1], and disables SDK retries so the runner request budget is exact. A nonempty source `JudgeSpec.model` must match the selected judge model only when a remote judge call is needed; an exact-gate hit completes locally without contacting or requiring that model.

TaskCompendium exposes a `judge` optional extra that installs `tasktrove-verify[judge]`. The `openai` package is imported only when grading a judge task. This adapter uses the existing TaskTrove scorer and does not add a separate verifier implementation.

The shared judge mode now raises an infrastructure error after bounded retries when a response has no parseable score in [0, 1]. Checklist grading fails as a whole if any criterion has no valid score. Provider, timeout, quota, transport, or deterministic-constraint checker failures also return no reward; a checker crash cannot be treated as a failed candidate answer. Successful and failed grades retain judge prompts and every judge response attempt in verifier evidence, without credentials.

## Verifier contract

`VerifierKind.JUDGE` selects `JudgeVerifier` from the explicit verifier registry. The private JSON parameters contain the existing `JudgeSpec` rendered as TOML plus an optional inline context string. This reuses TaskTrove's parser for rubric fields and constraints. Inline context is limited to 60,000 characters.

The verifier passes `TextSubmission.value` and inline private context directly to `tasktrove_verify.modes.grade_judge.grade_candidate`. The shared file-based `grade` entry point reads answer and context files and delegates to the same scorer. TaskCompendium passes through the shared reward without validating or clamping its range. The shared parser retains its existing numeric syntax and [0, 1] range; checklist criteria pass only at score 1. Normalization, IFEval checks, exact gating, prompt construction, score parsing, and checklist aggregation remain shared. A valid score maps to `GradeResult(GRADED, reward)` with judge evidence. Provider and parser failures map to `GradeResult(INFRA_ERROR, None, diagnostic)`, which Harbor records without a reward.

`run_trial` accepts a separate `JudgeRuntimeConfig` with `model`, `base_url`, `api_key_env`, `request_timeout`, and `max_requests`. It passes this config to the verifier in memory; it is absent from `TaskSpec` and exported task files. The API key is resolved from the named environment variable inside the grader and is never stored in task metadata or result evidence. The grader clamps the source timeout to the runtime limit and counts every call, including malformed-response retries, against the request budget. The exact gate runs before a remote request.

The source `JudgeSpec.model` acts as a private model requirement for remote grading. If it is present and differs from the runner-selected judge model, a request that reaches the endpoint fails as infrastructure; a deterministic exact-gate hit remains local. Production still needs an explicit model allowlist and request-budget policy. Do not make paid judge calls during import, validation, or tests.

## Field mapping

| TaskTrove `JudgeSpec` field | Private TaskCompendium representation | Runtime behavior |
| --- | --- | --- |
| `references` | `references: tuple[str, ...]` | Private reference rubric input. |
| `criteria` | `criteria: tuple[str, ...]` | Private checklist rubric input. |
| `question` | `question: str` | Passed to the existing prompt builder. |
| `context` | Optional inline private context string | Converter reads the source file and embeds its text; shared reference and checklist prompts receive it; never add it to `ConversationInput`. |
| `constraints` | Existing `Constraint` values | Resolve with the shared IFEval check registry before judging. |
| `rubric` | Existing `reference` / `checklist` enum | Reject unsupported values during conversion. |
| `model` | Private requested-model value | Require a match with the runtime-selected model when a remote judge call is needed. |
| `exact_gate` | `exact_gate: bool` | Run the deterministic gate before calling the judge. |
| `request_timeout` | `request_timeout: float` | Clamp to the runner's timeout limit. |
| `output` | Not source-controlled in TaskCompendium | Pass the extracted submission text directly to the shared scorer. |

TaskTrove's `context` names a file relative to `tests/`. The adapter passes its decoded contents directly to the shared scorer. A converter must read the referenced file from the bounded source archive and reject missing, unsafe, oversized, or non-UTF-8 content. The adapter rejects absolute paths, parent-directory traversal, and context longer than 60,000 characters. It does not add a general resource graph.

## Public projection and provenance

The release artifact carries the complete grading specification, including the original gold and verifier configuration, so a consumer can run the grader. The harness keeps those fields out of the model-visible prompt and supplies them only to the verifier. Preserve task IDs, source provenance, original ordered tags, and required attribution in the released artifact. Exclude judge endpoint settings and credentials from source rows, task specs, and released grading data; provide them only through runner configuration.

Schema 0.16 provides original ordered source tags. Preserve each tag string and its source order in any projection. Keep provenance as source dataset, immutable source revision, source row, and converter revision so each imported rubric can be traced to its archive.

## Source audit boundary

The requested internal input is `s3://marin-us-east-02a/marin/tasktrove/clean/2026.09.18.3/`, limited to `tasks/` and `sft/` and excluding `repeatedrl/`. A regional metadata-only inventory completed; the S3 manifest SHA256 matches the pinned HF manifest below (`5b8b0337d1473fbf2dd6bc2e682b4c597ca988edcb398e5abcd6f009c3aef75a`). No S3 archive payload bytes were downloaded or compared, so S3/HF archive-byte equivalence remains unverified. The source importer parsed and converted two bounded archives extracted from the immutable published HF snapshot for private Harbor trials; this does not establish internal S3 archive equivalence, public acceptance, or bulk import. Both samples use importer revision `taskcompendium-tasktrove-judge-v0.1`; the refreshed branch uses TaskCompendium schema 0.16.

The published clean HF dataset is pinned to revision [`9065fa568394f286dab0081e43dc76fc87c48984`](https://huggingface.co/datasets/open-athena/task-trove/tree/9065fa568394f286dab0081e43dc76fc87c48984), with manifest SHA256 `5b8b0337d1473fbf2dd6bc2e682b4c597ca988edcb398e5abcd6f009c3aef75a`. A bounded metadata audit confirmed 861,848 rows in that snapshot; its manifest reports 379,355 judge rows. An independent scout reported 272,429 OpenQA-family rows, including 150,468 science and 121,961 knowledge rows, plus 105,874 rubric rows. These are provisional metadata-filter counts, not accepted imports; the exact query predicates were not retained here, and their reported total is 1,052 below the manifest's judge-row count. Do not use them as a complete partition or source-batch coverage claim. Representative physical offsets 2,336 and 296 were matched to the immutable parquet and their archive identities verified. The unpinned `/rows` response must be matched by bytes to the immutable parquet revision before making source-row claims.

The upstream repository reference is `open-thoughts/TaskTrove` at commit [`02923004846e4e73862c20962f823a6d05100e7a`](https://github.com/open-thoughts/TaskTrove/tree/02923004846e4e73862c20962f823a6d05100e7a). The repository and consolidated Clean HF cards declare Apache-2.0 as package-level metadata; this does not establish rights for every upstream source row. The packaging audit identifies only 121,961 knowledge OpenQA rows as current public-release candidates, under NVIDIA's pinned [Nemotron-RL-knowledge-openqa card](https://huggingface.co/datasets/nvidia/Nemotron-RL-knowledge-openqa/blob/3604d4119623f2961c9cd0a3a5365e0cff0dd393/README.md), which identifies NVIDIA Corporation and CC BY 4.0. A release of those candidates must carry appropriate attribution, a license link, and a notice of changes as required by [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Science OpenQA and rubric/safety rows remain held from public release while row lineage and rights are unclear. These rights findings do not establish the content or licensing of the internal S3 release.

Two immutable-parquet rows were matched to extracted archive identity and imported by the new converter. Row 2,336 is `laion__nemotron-gym-knowledge-openqa-v4`, converter `nemotron_openqa`, template `qa-short-answer`, archive `openqa-0b723ddc24b3.tar.gz` (1,616 bytes; SHA256 `a30762217d62a48939ae2d082b605114f5b33485dcf5ab90cacda903a0075e96`). Its source field names the knowledge OpenQA cohort and its six ordered source tags include `knowledge`, so this sample maps to the 121,961-row public-candidate cohort. It has one reference and no context or constraints. This bounded private conversion and trial does not constitute public acceptance or upload. Row 296 is `laion__wizardlm-orca-v4`, converter `judge_rubric`, template `llm-judge-freeform`, archive `wizardlm_orca-8818` (1,954 bytes; SHA256 `0296358e3411e46fd73f49b592917f6f24cefc958dd65ce68352305d77775442`). It has three checklist criteria, no context or constraints, and four ordered source tags; rubric/safety rows remain held from public release while lineage and rights are unresolved. Both sidecars report matching source archive identity and the same immutable parquet LFS SHA256 `83320be884448b96ec715738520c0989fbc99a914cef6b97811f6d66c43a1a94`.

The actual-import Harbor evidence was kept in private local temporary storage and is not committed because it contains verifier parameters and gold rubric data. The scripted trials imported those archives through `read_archive` and `import_task`, exported Harbor tasks, and ran `run_trial`: OpenQA exact gate scored 1 without a judge call; semantic reference grading scored 1; an incorrect answer scored 0; and the three-criterion checklist scored 2/3. All four trials completed with `status=graded`. The script used a local scripted endpoint and made no paid calls. Archive identity and SHA256 records are listed above.

See [importing tasks](../../../IMPORTING.md) for prompt cleanup, submission conventions, provenance, and source coverage evidence.

## Harbor acceptance

Every included source batch needs an end-to-end Harbor trial from an actual exported task produced from a pinned source archive. Use a scripted OpenAI-compatible endpoint or deterministic response replay. Exercise an exact-gate hit, semantic reference score, multi-criterion checklist, failed deterministic constraint, provider timeout/429/5xx/transport failure, and malformed response after bounded retries. Confirm that infrastructure failures have no Harbor reward. Verify that references, criteria, context, private verifier JSON, and judge configuration are absent from the public projection and agent-visible prompt.

Retain the task digest, source pin and row, converter revision, public projection, typed submission trace, endpoint/model identity without credentials, judge requests and responses, verifier result, and Harbor trial result. A converter test or direct `grade_answer` call does not satisfy this gate.

Five synthetic TaskCompendium tasks have run through Harbor replay against a local scripted endpoint: exact reference gating, semantic reference grading, checklist averaging, malformed-response retries, and request-budget exhaustion. Malformed-response and budget failures produced `infra_error`, no reward, and retained judge evidence. The two source-backed converter trials above establish bounded examples, not source-batch coverage. Neither the synthetic tests nor these source trials test HTTP 429/5xx, transport failure, deterministic constraints, nonempty private context, or the release public projection. The internal S3 source batch and alpha projection therefore remain unvalidated.

## Runtime policy

Production deployment needs an explicit endpoint/model allowlist and request budget. Conversion coverage remains unknown until the referenced S3 archives are sampled and parsed. Each included source batch must pass the source-pinned Harbor acceptance gate above.
