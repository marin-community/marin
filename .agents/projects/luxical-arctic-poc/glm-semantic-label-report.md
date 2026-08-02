# GLM-5.2 semantic-label pilot

## Decision

GLM-5.2 can produce useful source-blind semantic labels for this data. The pilot did not show label collapse.

The current 38-bucket vocabulary is not ready as a single-label target. Several buckets overlap. The next iteration must define primary-bucket precedence or use multi-label targets.

## Run

- Model: `zai-org/GLM-5.2-FP8`
- Model revision: `ba978f7d347eaf65d22f1a86833408afdb953541`
- Description, taxonomy, and first 150 assignment revision: `e926fa215`
- Remaining assignment and summary revision: `f1ee6a014`
- Completed job: `/rav/lux-glm52-semantic-1000-b200-004`
- Sample: 1,000 documents from 146 sources
- Sampling: balanced by source, then stable hash
- Prompt data: a source-blind document view capped near 6,000 characters
- Source names and source categories in prompts: none
- Descriptions: 1,000
- Assignments: 1,000
- Buckets: 38, including `OTHER_UNCLEAR`
- Requested nonfallback bucket range: 30 through 50

The run resumed from stored checkpoints. Its final attempt took 1,729 seconds. This time is not a clean full-run benchmark.

For long documents, the view joins 2,000 characters from the start, middle, and end. The stored sample does not keep original lengths. Therefore, the truncated-document fraction is not available.

Two earlier attempts found response-shape faults. One JSON response ended early. One assignment omitted an optional rationale. The final code accepts the optional field and retries truncated JSON with a larger token limit.

## Collapse checks

| Gate | Result | Status |
|---|---:|---|
| Complete assignments | 1,000 of 1,000 | Passes the registered completeness gate |
| Used buckets | 38 of 38 | Diagnostic |
| Largest bucket | 13.5% | Diagnostic |
| Five largest buckets | 44.9% | Diagnostic |
| Effective buckets by Shannon entropy | 24.53 | Diagnostic |
| `OTHER_UNCLEAR` | 0.3% | Passes the registered 10% gate |
| Mean GLM confidence | 0.9375 | Diagnostic only |

The largest buckets were assessments and question-answer data at 13.5%, software code at 8.9%, AI interaction logs at 7.7%, educational material at 7.6%, and legal text at 7.2%. The report derives these values from stored primary-bucket counts. The current code writes them into future summaries.

These results reject a simple concentration failure. They do not prove that each bucket is coherent.

## Blinded Claude check

Claude classified 20 documents with the frozen 38-bucket vocabulary. It saw document views and bucket definitions. It did not see source metadata or GLM assignments.

The sample selected the lowest-confidence document from 20 stable-hash-ordered primary buckets. Eighteen buckets had no review document. The hash-fill path did not run.

This is a low-confidence stress sample. It is not a representative accuracy sample. The Claude CLI default model produced the labels, but the run did not record its model ID.

| Metric | Result | Meaning |
|---|---:|---|
| Exact primary-bucket agreement | 50% | Stress-sample diagnostic |
| Any-bucket-set overlap | 95% | Nineteen documents shared at least one GLM and Claude bucket |
| GLM primary in Claude label set | 95% | Claude usually kept the GLM primary as a secondary choice |
| Claude primary in GLM label set | 75% | GLM kept fewer Claude primary choices in its label set |

The 50% result cannot be compared with the registered 70% gate because the sample is biased toward low GLM confidence. It still shows weak primary-label stability in hard cases.

The 95% set overlap shows that both models usually identify the same semantic area. They often select a different primary bucket.

Two inspected disagreements support this diagnosis:

- A one-line request for a political discussion was `CREATIVE_NARRATIVE` for GLM and `OTHER_UNCLEAR` for Claude.
- A Python debugging dialogue was `TECHNICAL_SUPPORT` for GLM and `EDUCATIONAL_INSTRUCTIONAL` for Claude. Both models included the other semantic role in their full label sets.

Other disagreements had the same form. They crossed policy and news, government and technical research, or reference text and unclear fragments. Document splicing can also increase ambiguity.

Claude is an independent model check. It is not ground truth. This stress sample gives no population accuracy estimate.

## Main finding

GLM-5.2 does not collapse these 1,000 documents into one topic. It creates a broad vocabulary that covers code, research, legal text, dialogue, public records, and other content.

The main observed fault is label precedence. The vocabulary mixes topic, document form, intent, and source-like genre at the same level. A document can validly match several buckets.

The 30-bucket minimum also pushes the model toward a detailed vocabulary. The pilot did not test whether a smaller vocabulary reduces overlap.

Training a student on one primary bucket would force unstable choices into the target. A multi-label or hierarchical target is safer.

## Next experiment

1. Create 15-, 20-, and 30-bucket candidates from the stored descriptions.
2. Keep topic and document form on separate hierarchy levels, or define primary-label precedence.
3. Draw a representative 100-document sample and a separate 50-document low-confidence stress sample.
4. Pin and record the Claude model.
5. Make language an ISO 639 code in both teacher prompts.
6. Use the full primary and secondary label set as the teacher target.
7. Label 5,000 documents with the selected vocabulary.
8. Train a Fast Transformer student on multi-label targets.
9. Compare student and teacher with top-k overlap, cluster coherence, collapse checks, and throughput.

Do not start the student scaling ladder from these single primary labels. First fix the target structure and run the representative review.

## Peer review disposition

The review found the stress-sample bias, document splicing, mixed artifact revisions, forced bucket floor, and unrecorded Claude model. This report now states each limitation.

The code now computes concentration diagnostics, rejects extra or unknown secondary labels, restores cached taxonomies before candidate work, and removes raw document packages from durable job logs. It also removes invalid free-text agreement metrics.

The task-output chunk stream remains because the local Claude client cannot read the private object store. The stream now runs only through direct task output. The pipeline no longer writes document text to durable job logs.

The retry and duplicate-checkpoint refactors remain separate cleanup work. They do not change the stored pilot evidence. Moving these research scripts is also outside this result correction.
