# GLM-5.2 semantic-label pilot

## Decision

GLM-5.2 can produce useful source-blind semantic labels for this data. The pilot did not show label collapse.

The current 38-bucket vocabulary is not ready as a single-label target. Several buckets overlap. The next iteration must define primary-bucket precedence or use multi-label targets.

## Run

- Model: `zai-org/GLM-5.2-FP8`
- Model revision: `ba978f7d347eaf65d22f1a86833408afdb953541`
- Code revision: `f1ee6a014`
- Completed job: `/rav/lux-glm52-semantic-1000-b200-004`
- Sample: 1,000 documents from 146 sources
- Sampling: balanced by source, then stable hash
- Prompt data: document text only
- Source names and source categories in prompts: none
- Descriptions: 1,000
- Assignments: 1,000
- Buckets: 38, including `OTHER_UNCLEAR`

The run resumed from stored checkpoints. Its final attempt took 1,729 seconds. This time is not a clean full-run benchmark.

Two earlier attempts found response-shape faults. One JSON response ended early. One assignment omitted an optional rationale. The final code accepts the optional field and retries truncated JSON with a larger token limit.

## Collapse gates

| Gate | Result | Status |
|---|---:|---|
| Complete assignments | 1,000 of 1,000 | Pass |
| Used buckets | 38 of 38 | Pass |
| Largest bucket | 13.5% | Pass |
| Five largest buckets | 44.9% | Pass |
| Effective buckets by Shannon entropy | 24.53 | Pass |
| `OTHER_UNCLEAR` | 0.3% | Pass |
| Mean GLM confidence | 0.9375 | Diagnostic only |

The largest buckets were assessments and question-answer data at 13.5%, software code at 8.9%, AI interaction logs at 7.7%, educational material at 7.6%, and legal text at 7.2%.

These results reject a simple concentration failure. They do not prove that each bucket is coherent.

## Blinded Claude check

Claude classified 20 documents with the frozen 38-bucket vocabulary. It saw document text and bucket definitions. It did not see source metadata or GLM assignments.

The review sample selected one low-confidence document from each stable bucket. It then filled the sample by stable hash.

| Metric | Result | Meaning |
|---|---:|---|
| Exact primary-bucket agreement | 50% | Fails the 70% stop gate |
| Any-bucket-set overlap | 95% | Nineteen documents shared at least one GLM and Claude bucket |
| Exact language string agreement | 35% | Invalid gate because names and codes differ |
| Exact document-type string agreement | 0% | Invalid gate because both models write free text |

The exact primary result is weak. The 95% set overlap shows that both models usually identify the same semantic area. They often select a different primary bucket.

Two inspected disagreements support this diagnosis:

- A one-line request for a political discussion was `CREATIVE_NARRATIVE` for GLM and `OTHER_UNCLEAR` for Claude.
- A Python debugging dialogue was `TECHNICAL_SUPPORT` for GLM and `EDUCATIONAL_INSTRUCTIONAL` for Claude. Both models included the other semantic role in their full label sets.

Other disagreements had the same form. They crossed policy and news, government and technical research, or reference text and unclear fragments.

Claude is an independent model check. It is not ground truth. Twenty documents give a directional result only.

## Main finding

GLM-5.2 does not collapse these 1,000 documents into one topic. It creates a broad vocabulary that covers code, research, legal text, dialogue, public records, and other content.

The current failure is label precedence. The vocabulary mixes topic, document form, intent, and source-like genre at the same level. A document can validly match several buckets.

Training a student on one primary bucket would force unstable choices into the target. A multi-label or hierarchical target is safer.

## Next experiment

1. Merge or add precedence rules for the main overlap pairs.
2. Make language an ISO 639 code in both teacher prompts.
3. Remove exact free-text document-type agreement as a gate.
4. Use the full primary and secondary label set as the teacher target.
5. Label 5,000 documents with the revised vocabulary.
6. Train a Fast Transformer student on multi-label targets.
7. Compare student and teacher with top-k overlap, cluster coherence, collapse gates, and throughput.

Do not start the student scaling ladder from these single primary labels. First fix the target structure and repeat the blinded review on at least 100 documents.
