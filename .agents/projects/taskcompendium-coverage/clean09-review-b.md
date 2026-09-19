# TaskTrove Clean `.9` independent representation review (group B)

Reviewed 2026-09-14 against the authoritative Clean `.9` release:

- HF mirror: `open-athena/task-trove`, Parquet train export
- S3 root: `s3://marin-us-east-02a/marin/tasktrove/clean/2026.09.10.9`
- Manifest: `/tmp/tasktrove-clean-09/manifest.json`
- Deterministic sample metadata: `/tmp/tasktrove-clean-09/sample-metadata.json`

I fetched exactly the two metadata-selected paths for each of the 21 requested
sources (42 archives). The HF Parquet has one large physical row-group file, so
I resolved each `(source, path)` to its row-group/offset using metadata columns,
then read the 34 unique required row groups from an existing running Iris pod
with its injected S3 credentials. Only the 42 selected task BLOBs were emitted
to the host. Every archive was read as a tar stream; I inspected `task.toml`,
`instruction.md`, `tests/verifier.toml`, all schema/checker/expected-resource
files, and the Dockerfile. I did not run source checkers, submitted programs,
or imported scripts.

The current checkout has `RELEASE = "2026.09.10.8"` and verifier revision
`b2b68d8b...` in `tasktrove.py`, while every sampled `.9` Dockerfile pins
`b76d03131c`. Consequently, a direct importer dry-run rejects the otherwise
supported `.9` archives at the revision guard. The representation judgments
below separate that checkout pin mismatch from source semantics; the release
and verifier pins must be updated before claiming `.9` importer results.

## Disposition by sampled task

`keep` means the two samples preserve the source task contract and have a
current importer path after the release pin is corrected. `reject` means the
sampled source checker or task contract is confirmed defective. `unsupported`
means the source is coherent in the samples but no current TaskCompendium
importer handles that family/converter or output protocol.

| source | exact sampled task paths | disposition | representation findings |
| --- | --- | --- | --- |
| `laion__nemotron-gym-competitive-coding-v2` | `comp-coding-28b11b2c1e7b.tar.gz`; `comp-coding-585551561f1a.tar.gz` | unsupported | Both are complete Python 3 batch stdio problems with exact per-case outputs and `/app/solution.py` contract. The private verifier runs all 16/50 cases and has no special judge. The raw wrapper says “verifier will run”, which a future importer must remove; no semantic defect was established. Current coding importer only accepts `codeforces`/`python_unit_tests`, not `nemotron_competitive`. |
| `laion__nemotron-gym-instruction-following-calendar-v3` | `agent-calendar-2f6d4ea91d7e.tar.gz`; `agent-calendar-3a1e8a946d71.tar.gz` | unsupported | The custom checker validates JSON parsing, exact event IDs and normalized names, positive integer durations, time windows, natural-language before/after/at/between constraints, and pairwise overlap. Expected events provide all required facts. Raw instructions mention the checker, but that is removable delivery text. No current importer handles this stateful calendar protocol. |
| `laion__nemotron-gym-instruction-following-structured-v3` | `if-structured-700bbfb7ae3d.tar.gz`; `if-structured-a252a3bbe924.tar.gz` | reject | The first schema requires viewer rating/recommendation/viewing context/platform/duration absent from the *Everything Everywhere All at Once* document. The second requires `version` and `releaseYear` absent from the wearable-technology document. Both include an evaluation-contract preamble permitting arbitrary schema-valid values, and the JSON-schema verifier checks structure only; invented values therefore receive acceptance. Confirmed defects are recorded in the problematic ledger. |
| `laion__nemotron-gym-instruction-following-v3` | `Nemotron-RL-instruction_following-57d39940cdab.tar.gz`; `Nemotron-RL-instruction_following-cb2b2e78637d.tar.gz` | unsupported | IFEval samples state explicit lowercase-frequency and sentence-final-word constraints, and the private verifier contains the corresponding typed checks. Required answer-file behavior is clear. The raw shell/verifier wrapper needs cleaning, but no source contradiction was found. No current TaskTrove IFEval importer exists. |
| `laion__nemotron-gym-knowledge-mcqa-v2` | `Nemotron-RL-knowledge-mcqa-1961bdb52b5a.tar.gz`; `Nemotron-RL-knowledge-mcqa-cce3426cf566.tar.gz` | keep | Both have complete ten-option questions, one expected option (`C`/`G`), and a single-letter MCQ verifier. `tasktrove_answers._clean_mcq` removes the delivery and verifier explanation while retaining the question and answer-format requirement. Expected answers remain private. |
| `laion__nemotron-gym-knowledge-openqa-v4` | `openqa-80f6c461ebcf.tar.gz`; `openqa-c7e9374b56ea.tar.gz` | keep | Both questions contain the facts needed for an answer and have one private reference, `rubric = "reference"`, and `exact_gate = true`. The judge importer removes the LLM-judge delivery wrapper and keeps the boxed-answer source contract. |
| `laion__nemotron-gym-math-openmathreasoning-v2` | `Nemotron-RL-math-OpenMathReasoning-e8223ea14ef5.tar.gz`; `Nemotron-RL-math-OpenMathReasoning-b59f94dd5bab.tar.gz` | unsupported | Both are self-contained scalar math questions with nonempty exact expected answers and a boxed final-answer contract. They use `nemotron_math`, which the current math importer deliberately does not support (it only accepts `all_puzzles`). No source contradiction was found. |
| `laion__nemotron-gym-math-stack-overflow-v3` | `Nemotron-RL-math-stack_overflow-b4f0cd74a84a.tar.gz`; `Nemotron-RL-math-stack_overflow-735dcfb50cf9.tar.gz` | unsupported | Both contain complete scalar math problems with exact expected answers and no external resources. They are coherent answer tasks but use the unsupported `nemotron_math` converter. |
| `laion__nemotron-gym-math-v5` | `Nemotron-RL-Math-v2-d893974eeac7.tar.gz`; `Nemotron-RL-Math-v2-3b9806b4543d.tar.gz` | unsupported | Both contain complete mathematical questions and nonempty scalar exact answers, including the source boxed-answer contract. The current importer does not accept this `nemotron_math` family. |
| `laion__nemotron-gym-multichallenge-advanced-v4` | `multichallenge-bd743fdd3788.tar.gz`; `multichallenge-d5285d7544da.tar.gz` | unsupported | Both are explicitly multi-turn conversations. Their checklist criteria cover remembered names, schedules, ordered answers, negative facts, and required sign-off. Required facts occur in prior turns. The current TaskCompendium model excludes reactive multi-turn simulators, so no equivalent static lowering is available. Raw text exposes grading/checklist guidance and would need source-specific cleaning. |
| `laion__nemotron-gym-reasoning-gym-v2` | `reasoning-gym-54ac10b07937.tar.gz`; `reasoning-gym-66a9f5b4098f.tar.gz` | unsupported | Both include the question plus private `tests/entry.json` answer/metadata and use the `reasoning-gym` verifier with dataset/entry identifiers. The procedurally generated equation and letter-counting tasks are answerable, but no current importer supports the upstream scorer or this mode. |
| `laion__nemotron-gym-safety-v3` | `safety-81126840c306.tar.gz`; `safety-8e0de17b8424.tar.gz` | unsupported | The two harmful-request samples have private checklist criteria requiring refusal and resistance to jailbreak framing. No reference answer is needed for these open-ended safety responses, and the question itself supplies the needed context. Current judge importer accepts only `qa-short-answer`/`nemotron_openqa`; `judge_rubric` is unsupported. |
| `laion__nemotron-gym-science-so-openq-v3` | `science-38fd8e5eabe0.tar.gz`; `science-624d2b4da1fe.tar.gz` | unsupported | Both are self-contained science questions with one private reference and an exact reference judge. Their answer/boxed-format contracts are clear. They use `nemotron_openqa` metadata but `llm-judge-freeform` family, which the current judge importer rejects; no semantic defect was established. |
| `laion__nemotron-gym-structured-outputs-v4` | `if-structured-v2-478906f3fa67.tar.gz`; `if-structured-v2-529e76d630c5.tar.gz` | reject / unsupported | The YAML sample asks extraction from a detailed 1913 Indianapolis 500 document but declares `schema_type = "yaml"`; the current structured importer has no YAML path, so it is unsupported. The XML sample has an inline extraction schema in the public prompt but its private verifier only declares `required = ["output"]` and has no schema resource; `<output/>` bypasses all fields, types, and source fidelity. The XML defect is recorded in the problematic ledger and should be rejected pending a semantic checker. |
| `laion__stackexchange-codereview-sandboxes-verified-v2` | `codereview-68706`; `codereview-73823` | unsupported | Both contain the complete submitted code/question and private checklists requiring technically correct, concrete, context-specific review. No reference answer is required for these open-ended reviews. Current judge importer does not support `llm-judge-freeform`/`judge_rubric`. |
| `laion__stackexchange-overflow-sandboxes-verified-v2` | `overflow-9766`; `overflow-9815` | unsupported | Both have complete Python/C# questions and private checklists covering correctness, completeness, explanation, and version/context fidelity. Current `judge_rubric` importer is unsupported; no source contradiction was found. |
| `laion__stackexchange-superuser-sandboxes-verified-v2` | `superuser-59539`; `superuser-99834` | unsupported | Both have complete Windows/Excel questions and checklists requiring actionable, technically correct, context-specific answers, with risk warnings where relevant. Current `judge_rubric` importer is unsupported. |
| `laion__stackexchange-tezos-sandboxes-verified-v2` | `tezos-0100`; `tezos-0323` | unsupported | Both provide the Tezos/Docker or Michelson context needed by their checklist judges. The criteria cover correctness, actionable steps, and operational safety. Current `judge_rubric` importer is unsupported. |
| `laion__stackexchange-unix-sandboxes-verified-v2` | `unix-51635`; `unix-60316` | unsupported | Both provide complete Debian/Linux questions and private checklists requiring runnable commands, distro specificity, and risk awareness. Current `judge_rubric` importer is unsupported. |
| `laion__swesmith-oracle-filtered-v2` | `swesmith-06888`; `swesmith-13308` | unsupported | Both contain repository clone/install instructions, concrete bug reports, trusted test paths, and private pytest must-pass/must-not-break selections. The code remains in a source repo/image boundary and must execute only in an isolated verifier runtime. No current TaskTrove SWE importer handles `swe_trusted_paths`. |
| `laion__wizardlm-orca-v4` | `wizardlm_orca-8407`; `wizardlm_orca-1725` | unsupported | Both are complete general-assistant questions with private correctness/completeness/relevance checklists and no required external resources. The first has an irrelevant “multiple choice” preamble despite asking for a JSON data example, but this is noisy source instruction rather than a confirmed contradiction. Current `judge_rubric` importer is unsupported. |

## Confirmed defects and code status

The three structured-output defects above are recorded in
`.agents/logbooks/taskcompendium-problematic-tasks.md` with the exact Clean `.9`
paths. I made no source-archive or checker changes and added no tests because
this review produced no safe, generic repair: fixing the JSON tasks requires
recovering aligned source facts/schema or changing the task provenance, and
fixing the XML task requires a source-backed semantic extractor/checker. The
current checkout's release/verifier pin mismatch is a separate implementation
blocker for any importer validation and should be corrected by the release
integration work.

Focused validation was limited to tar integrity, TOML/JSON parsing, archive
member/path inspection, metadata-to-path matching, and static checker review.
No host execution of `tests/test.sh`, custom checkers, submitted code, or
source-provided scripts was performed. S3 access required an existing running
Iris pod with injected credentials; local host credentials and `aws` CLI were
unavailable. The 42 exact task archives and their decoded inspection outputs
are retained under `/tmp/tasktrove-clean-09/` for this review session.
