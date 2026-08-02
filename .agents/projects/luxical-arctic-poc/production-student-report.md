# Production semantic student trust report

Status: evaluation in progress

## Decision rule

Do not approve a production student until every required gate passes on fixed
held-out documents. Source provenance is not a prediction target. A source can
contain many unrelated semantic domains and document forms.

The current candidate is the 3M Arctic FastTransformer. It has 9,299,200
parameters and returns 256-dimensional vectors. It is the lead candidate, not
an approved production model.

## Current evidence

| Gate | Requirement | Current result | State |
| --- | --- | --- | --- |
| Finite output | 100% finite vectors | 100% on 74,752 documents and the 1,000-document semantic screen | Pass |
| Non-constant output | At least 99% unique vectors | 100% on the semantic screen | Pass |
| Global geometry | Effective-rank fraction at least 0.25 | 0.33326 on the semantic screen | Pass |
| Exact CPU speed | At least 0.85 times Luxical-One | 3.00349 times Luxical-One | Pass |
| Coarse semantic screen | No metric more than 0.02 below the best tested teacher | Passed all eight fixed metrics | Pass |
| Label reliability | Independent review gates for a frozen hierarchy | Curated compact passed representative Claude gates; low-confidence tail needs adjudication | Open |
| Fine semantic coherence | Parent, leaf, and form gates on accepted labels | Waiting for hierarchical labels | Open |
| Blind neighborhood review | Student is not worse than the best teacher | Not run | Open |
| Held-out robustness | All large semantic groups pass | Not run | Open |
| Optional ladder input | Bounded loader below 8 GiB peak RSS | 1.62 GB on all 3M rows | Pass |
| Release artifact | Pinned model, tokenizer, loader, and production smoke | Not built | Open |

The exact paired CPU test used 20,000 fixed documents, five alternating timed
repeats, eight CPU threads, and the CPU JAX backend. The student median was
18,492.61 documents per second. Pinned Luxical-One reached 6,157.05 documents
per second.

The coarse 1,000-document screen found coherent recipe, literature, code, and
molecule neighborhoods. It also found weak government-statistics and
technical-support neighborhoods. Its 38 labels have unclear primary-label
precedence. This screen cannot support a production decision.

All semantic neighbor, pair, rank, variance, and cluster metrics L2-normalize
each vector first. The older source-provenance collapse report used a different
evaluation path. Its concentration and source-probe values are historical
context, not production trust gates.

## Required label gates

GLM-5.2 creates two candidate domain hierarchies. Domain and document form are
separate labels. A pinned Claude model reviews a stable 100-document sample and
a separate 50-document low-confidence sample.

An accepted hierarchy must meet all of these gates:

- GLM assigns valid labels to all 1,000 pilot documents.
- The fallback parent has at most 5% of the documents.
- The largest parent has at most 30% of the documents.
- At least 80% of non-fallback parents and leaves are used.
- Claude exact primary-parent agreement is at least 80%.
- Claude any-parent overlap is at least 90%.
- Claude exact document-form agreement is at least 85%.
- The lower bound of the primary-parent agreement interval is recorded.
- The full GLM and Claude model revisions, prompts, taxonomies, sample IDs, and
  artifact digests are recorded.

Reject a hierarchy if primary-parent agreement is below 70%. Review and revise
the taxonomy when the result is from 70% through 79%.

The curated compact hierarchy removes one invalid document-form precedence
rule from the original GLM output. It changes no bucket ID or pilot assignment.
On 100 representative documents, Claude Opus 5 reached 81% exact parent
agreement, 98% any-parent overlap, and 85% exact form agreement. These results
pass the fixed gates. On the 50 lowest-confidence documents, exact parent and
form agreement fell to 38% and 58%. The final held-out labels must therefore
adjudicate the low-confidence tail before embedding metrics are approved.

Use the lowest-confidence 5 percent for this adjudication. This is the same
fraction as the failed 50-document pilot stress sample. Run the embedding gates
once with the raw GLM labels and once with the Claude labels for this tail. No
fixed global student metric can change by more than 0.02. The full gate decision
and the large-group gate decision cannot change. A larger change means that
label noise controls the result.

## Required embedding gates

Use only documents that were not used to train the student. Exclude neighbors
from the query document's source when a metric can otherwise reward duplicate
or templated source content. This exclusion is a leakage control. It does not
make source identity a target.

The student must meet all of these gates against Luxical-One and every saved
teacher:

- Parent-label neighbor overlap is no more than 0.02 below the best teacher.
- Leaf-label neighbor overlap is no more than 0.02 below the best teacher.
- Form-label neighbor overlap is no more than 0.02 below the best teacher.
- Parent, leaf, and form nearest-label macro-F1 are each no more than 0.02 below
  the best teacher.
- Parent and leaf cluster NMI are each no more than 0.02 below the best teacher.
- No large parent or form group has a nearest-label F1 loss greater than 0.03
  against the best teacher for that group.
- All vectors are finite. At least 99% are unique after four-decimal rounding.
- Effective-rank fraction is at least 0.25. Total normalized variance is at
  least 0.50.

The final set must contain at least 10,000 fixed documents from the held-out
manifest. It must cover code, multilingual text, structured data, research,
reference text, instructions, dialogue, news, narrative text, and
administrative text. The 1,000-document result is a taxonomy and metric smoke
test only.

## Blind neighborhood review

Sample 200 fixed queries across accepted domain parents and document forms.
For each query, show five source-excluded nearest documents from the student
and the strongest teacher. Hide model names and randomize left and right.
A pinned frontier reviewer scores semantic coherence and query relevance.

The student passes when its win count plus half of its tie count is at least
50% and the lower bound of the paired interval is at least 45%. Report results
for code, multilingual text, and standard text separately. Any group with
fewer than 30 queries is descriptive and cannot pass a release gate.

## Release gates

- Repeat the exact CPU benchmark from a pinned release artifact. The median
  speed must be at least 0.85 times Luxical-One.
- Record accelerator throughput for capacity planning. Accelerator speed does
  not replace the CPU gate.
- Pin the model digest, tokenizer-map digest, dependency versions, maximum input
  length, pooling method, and normalization method.
- Add a loader smoke test that checks shape, finite values, nonzero variance,
  and deterministic output for fixed documents.
- Run the full trust report from the release artifact, not from an in-memory
  training object.
- Get an independent peer review of this report and its artifacts. Address each
  accepted finding before approval.

## Current artifact identity

- Model SHA-256:
  `8735a4b49de0f7925904b0301516a2c8a5f9651bc2b605e4d27a80bca3f8ac3a`
- Tokenizer-map SHA-256:
  `50c92752d5a1d408234b8eee58c1c0f6179f603253caabe4fcd3a06f990710f0`
- Exact CPU report:
  `s3://marin-us-east-02a/marin/user/rav/luxical-arctic-ladder/manifest-v2/fast-student/speed/cpu-trained-full-full-3m.json`
- Coarse semantic report:
  `s3://marin-us-east-02a/marin/user/rav/luxical-arctic-ladder/manifest-v2/evaluation/semantic-labels/glm-5.2/pilot-1000-20260802-001/embedding-screen-v2/report.json`

## Open decision

Approve the current 3M candidate only if it passes the accepted hierarchy,
10,000-document, blind-review, robustness, and release gates. If it fails a
semantic quality gate, train a nested 3M, 10M, and 30M ladder with the same
architecture and loss. Stop the ladder when two adjacent rungs improve the
failed metric by less than 0.005. Change the teacher or objective only when the
teacher itself passes the failed evaluation and scale does not close the gap.

The disk-backed staged loader passed its 3M canary with 1,623,203,840 bytes of
peak RSS. It scanned every row and saw all 146 sources. This is a 79.1 percent
peak-RSS reduction from the first mapped-page implementation. A 10M or 30M
rung is now permitted only when the 3M candidate fails a semantic gate and the
required prepared rows and teacher vectors exist.

The Arctic teacher embeds three head, middle, and tail windows. Short documents
can repeat the same text, and medium documents can have overlapping windows.
Exact repeats do not change an averaged embedding, but overlap can weight the
middle of a document more heavily. The independent GLM labels and blind
neighborhood review must catch a harmful effect. If the candidate fails those
gates, test a deduplicated-window teacher before more Arctic scaling.
