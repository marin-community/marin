# Production semantic student trust report

Status: evaluation in progress

## TL;DR

The 3M and 10M students are faster than Luxical-One on CPU. Their vectors pass
all health gates. Both students fail the held-out semantic gates and the
model-hidden neighborhood review. The 10M rung improves non-English text and
some global metrics. It does not improve the overall hidden-review score, and
it reduces the code score. The nested 30M rung is the final pure-scaling test.

## Decision rule

Do not approve a production student until every required gate passes on fixed
held-out documents. Source provenance is not a prediction target. A source can
contain many unrelated semantic domains and document forms.

The current candidate is the 10M Arctic FastTransformer. It has 9,299,200
parameters and returns 256-dimensional vectors. It is the lead candidate, not
an approved production model.

## Current evidence

| Gate | Requirement | Current result | State |
| --- | --- | --- | --- |
| Finite output | 100% finite vectors | 100% on the 10,000-document held-out set | Pass |
| Non-constant output | At least 99% unique vectors | 99.97% on the held-out set | Pass |
| Global geometry | Effective-rank fraction at least 0.25 | 0.42414 on the held-out set | Pass |
| Exact CPU speed | At least 0.85 times Luxical-One | 10.36458 times Luxical-One on the paired 10M test | Pass |
| Coarse semantic screen | No metric more than 0.02 below the best tested teacher | Passed all eight fixed metrics | Pass |
| Label reliability | Independent review gates for a frozen hierarchy | Tail adjudication changed each global metric by at most 0.00458 and changed no gate decision | Pass |
| Fine semantic coherence | Parent, leaf, and form gates on accepted labels | The 10M parent macro-F1 gate fails; leaf and form global gates pass | Fail |
| Blind neighborhood review | Student is not worse than the best teacher | 10M score 0.4450; 95% interval [0.3775, 0.5125] | Fail |
| Held-out robustness | All large semantic groups pass | Parent, form, and leaf large-group gates fail | Fail |
| Optional ladder input | Bounded loader below 8 GiB peak RSS | 1.62 GB on all 3M rows | Pass |
| Release artifact | Pinned model, tokenizer, loader, and production smoke | Not built | Open |

Each exact paired CPU test used 20,000 fixed documents, five alternating timed
repeats, eight CPU threads, and the CPU JAX backend. On the 10M test host, the
student median was 6,798.62 documents per second. Pinned Luxical-One reached
655.95 documents per second. This host was slower than the 3M test host. The
paired ratio is the portable release gate.

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
form agreement fell to 38% and 58%.

The held-out evaluation used Claude labels for the lowest-confidence 5 percent.
Claude reviewed 500 documents. The exact parent agreement was 40.2 percent.
The any-parent overlap was 73.8 percent. The exact form agreement was 55.2
percent.

Use the lowest-confidence 5 percent for this adjudication. This is the same
fraction as the failed 50-document pilot stress sample. Run the embedding gates
once with the raw GLM labels and once with the Claude labels for this tail. No
fixed global student metric can change by more than 0.02. The full gate decision
and the large-group gate decision cannot change. A larger change means that
label noise controls the result. The largest change was 0.00458. All full-gate
and large-group decisions stayed unchanged. Thus, label noise does not control
the 3M result.

## Held-out 3M result

The 3M student is not collapsed. All 10,000 vectors are finite. The unique
fraction at four decimals is 0.9997. The effective-rank fraction is 0.35219.
The total normalized variance is 0.84226.

The student does not preserve enough semantic information. The raw-label
parent macro-F1 is 0.44498, compared with 0.47415 for the best teacher. The
form macro-F1 is 0.37281, compared with 0.41156. The leaf cluster NMI is
0.40660, compared with 0.42961. The adjudicated labels keep each failure.

The large-group tests show losses in several important groups. These groups
include software, code form, technical documentation, reference text,
instructions, administrative records, and narrative text. The exact set
depends on the hierarchy level. The vectors do not collapse. The failure is
semantic loss.

## Held-out 10M result

The 10M student is not collapsed. All 10,000 vectors are finite. The unique
fraction at four decimals is 0.9997. The effective-rank fraction is 0.42414,
up from 0.35219 at 3M. The total normalized variance is 0.85427, up from
0.84226.

Some global semantic metrics improve. Form cross-source nearest-label macro-F1
improves by 0.02388 and passes. Leaf cluster NMI improves by 0.00970 and
passes. Parent cross-source nearest-label macro-F1 changes by -0.00010 and
still fails. The parent value is 0.44703. Scale alone did not improve this
failed metric.

Large-group gates still fail at all three levels. Failed parent groups include
corporate and business text, humanities and culture, intellectual property,
medical and biological text, unclear text, and technical documentation. Leaf
and form results also show losses in administrative records, narrative text,
reference text, structured data, and technical specifications. The exact group
set depends on the hierarchy level.

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

Claude Opus 5 completed the model-hidden review on 200 queries. The student had
87 wins, 5 ties, and 108 losses. Its score was 0.4475. The paired 95-percent
interval was [0.3800, 0.5150]. Thus, the overall gate failed.

The code group had 48 queries and a score of 0.55208. Its lower interval bound
was 0.41667, so the code gate failed. The non-English group had 31 queries and
a score of 0.19355. The other-text group had 121 queries and a score of
0.47107. All three group gates failed.

Claude Opus 5 repeated the same model-hidden method for the 10M student. The
student had 84 wins, 10 ties, and 106 losses. Its score was 0.4450. The paired
95-percent interval was [0.3775, 0.5125]. Thus, the overall gate failed and did
not improve from the 3M score of 0.4475.

The 10M code group had 50 queries and a score of 0.4500. This is lower than the
3M score of 0.55208. The non-English group had 31 queries and a score of
0.32258. This is higher than the 3M score of 0.19355, but it still fails. The
other-text group had 119 queries and a score of 0.47479. It is nearly unchanged
from 3M. More training data changed which groups were weak, but did not improve
the overall visible result.

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
- Raw held-out report:
  `s3://marin-us-east-02a/marin/user/rav/luxical-arctic-ladder/manifest-v2/evaluation/semantic-labels/glm-5.2/pilot-1000-20260802-001/hierarchies-v1/hierarchy-1000-20260802-002/compact/heldout-10000-20260802-001/raw-v1/embedding-screen-v1/report.json`
- Adjudicated held-out report:
  `s3://marin-us-east-02a/marin/user/rav/luxical-arctic-ladder/manifest-v2/evaluation/semantic-labels/glm-5.2/pilot-1000-20260802-001/hierarchies-v1/hierarchy-1000-20260802-002/compact/heldout-10000-20260802-001/adjudicated-v1/embedding-screen-v1/report.json`
- Label-sensitivity report:
  `s3://marin-us-east-02a/marin/user/rav/luxical-arctic-ladder/manifest-v2/evaluation/semantic-labels/glm-5.2/pilot-1000-20260802-001/hierarchies-v1/hierarchy-1000-20260802-002/compact/heldout-10000-20260802-001/adjudicated-v1/label-sensitivity-v1/report.json`
- Blind neighborhood report:
  `s3://marin-us-east-02a/marin/user/rav/luxical-arctic-ladder/manifest-v2/evaluation/semantic-labels/glm-5.2/pilot-1000-20260802-001/hierarchies-v1/hierarchy-1000-20260802-002/compact/heldout-10000-20260802-001/adjudicated-v1/blind-neighborhood-review-v1/claude-opus-5-report.json`
  (SHA-256 `d4f8c4b7b679540b367cb4d9f02d7e77151b79bea036e50a841faa1bf1ca394f`)
- 10M model SHA-256:
  `fb3bb5f0e6e625bf72b2052ec7a76e6aa172f90fd2cebd551f045ac7bac473d3`
- 10M exact CPU report:
  `s3://marin-us-east-02a/marin/user/rav/luxical-arctic-ladder/manifest-v2/fast-student/speed/cpu-trained-full-full-10m.json`
- 10M adjudicated held-out report:
  `s3://marin-us-east-02a/marin/user/rav/luxical-arctic-ladder/manifest-v2/evaluation/semantic-labels/glm-5.2/pilot-1000-20260802-001/hierarchies-v1/hierarchy-1000-20260802-002/compact/heldout-10000-20260802-001/student-fast_arctic_10m/adjudicated-v1/embedding-screen-v1/report.json`
- 10M blind neighborhood report:
  `s3://marin-us-east-02a/marin/user/rav/luxical-arctic-ladder/manifest-v2/evaluation/semantic-labels/glm-5.2/pilot-1000-20260802-001/hierarchies-v1/hierarchy-1000-20260802-002/compact/heldout-10000-20260802-001/student-fast_arctic_10m/adjudicated-v1/blind-neighborhood-review-v1/claude-opus-5-report.json`
  (SHA-256 `d149ac1a316bab66e5e87cb8ec29af016e4085e9388d3fde858c4f91ded86b8f`)
- Large 3M control model SHA-256:
  `faa28194a890e0e50326fba28e99f1924a07e15cfc66a1164983f3e75db46e56`
- Large 3M exact CPU report:
  `s3://marin-us-east-02a/marin/user/rav/luxical-arctic-ladder/manifest-v2/fast-student/speed/cpu-trained-large-large-3m.json`
- Large 3M adjudicated report:
  `s3://marin-us-east-02a/marin/user/rav/luxical-arctic-ladder/manifest-v2/evaluation/semantic-labels/glm-5.2/pilot-1000-20260802-001/hierarchies-v1/hierarchy-1000-20260802-002/compact/heldout-10000-20260802-001/student-fast_arctic_large_3m/adjudicated-v1/embedding-screen-v1/report.json`

## Open decision

Reject the 3M and 10M candidates. Train the nested 30M rung with the same
architecture, loss, tokenizer map, and Arctic teacher. This is the final
pure-scaling test. The 3M-to-10M hidden-review score changed by -0.0025. If the
10M-to-30M score improves by less than 0.005, two adjacent scaling changes will
have failed the stop rule. Stop scaling and change the teacher windows or the
training objective at that point.

The disk-backed staged loader passed its 3M canary with 1,623,203,840 bytes of
peak RSS. It scanned every row and saw all 146 sources. This is a 79.1 percent
peak-RSS reduction from the first mapped-page implementation. A 10M or 30M
rung is now permitted only when the 3M candidate fails a semantic gate and the
required prepared rows and teacher vectors exist.

The Arctic teacher embeds three head, middle, and tail windows. Short documents
can repeat the same text, and medium documents can have overlapping windows.
Exact repeats do not change an averaged embedding, but overlap can weight the
middle of a document more heavily. The independent GLM labels and blind
neighborhood review must catch a harmful effect. If the 30M candidate fails
those gates, stop scaling and test a deduplicated-window teacher.

The 28.4M-parameter control fails the CPU release gate at 0.78952 times
Luxical-One speed. It also fails the semantic and large-group gates. More
capacity improves some fine-label metrics, but it does not correct the broad
semantic loss. Thus, capacity alone is not the next production direction.
