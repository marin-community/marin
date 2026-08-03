# Production semantic student trust report

Status: evaluation in progress

## TL;DR

The rank-preserving GLM projection is the lead candidate. It folds into the
9.3M-parameter FastTransformer head and adds no inference operation. The
1,000-label pilot passes all global semantic and vector-health gates on 10,000
fixed documents. It fails 12 large content-group gates. A disjoint 50,000-label
run is active to determine whether semantic supervision fixes these remaining
groups. The 30M Arctic student, larger student, longer-input student, and Qwen
neighbor student all fail release gates.

## Decision rule

Do not approve a production student until every required gate passes on fixed
held-out documents. Source provenance is not a prediction target. A source can
contain many unrelated semantic domains and document forms.

The current candidate is the projected 30M Arctic FastTransformer. It has
9,299,200 parameters and returns 256-dimensional vectors. GLM-5.2 supplies
semantic labels. Arctic supplies the base vector geometry. The pilot candidate
is not an approved production model.

## Current evidence

| Gate | Requirement | Current result | State |
| --- | --- | --- | --- |
| Finite output | 100% finite vectors | 100% for the pilot projection on the held-out set | Pass |
| Non-constant output | At least 99% unique vectors | At least 99% for the pilot projection | Pass |
| Global geometry | Effective-rank fraction at least 0.25 | 0.39314 for the pilot projection | Pass |
| Exact CPU speed | At least 0.85 times Luxical-One from a stable paired test | The base graph reached 0.94729 with float32 CPU math. The final projected artifact still needs the exact test | Open |
| Coarse semantic screen | No metric more than 0.02 below the best tested teacher | Every global pilot-projection metric passes | Pass |
| Label reliability | Independent review gates for a frozen hierarchy | Tail adjudication changed each global metric by at most 0.00458 and changed no gate decision | Pass |
| Fine semantic coherence | Parent, leaf, and form gates on accepted labels | The pilot projection passes every global level | Pass |
| Blind neighborhood review | Student is not worse than the best teacher | Deferred until the 50,000-label model passes visible gates | Open |
| Held-out robustness | All large semantic groups pass | 12 pilot-projection group gates fail | Fail |
| Fixed production buckets | Parent, leaf, and form NMI and purity pass for 40 buckets | Added for the 50,000-label model | Open |
| Optional ladder input | Bounded loader below 8 GiB peak RSS | 1.62 GB on all 3M rows | Pass |
| Release artifact | Pinned model, tokenizer, loader, and production smoke | A staged runtime now binds evaluation and speed to the production loader; final candidate bundle remains open | Open |

The previous paired CPU test used 20,000 fixed documents, five alternating
timed repeats, eight CPU threads, and the CPU JAX backend. The projected
student rates were stable from 6,589 through 6,836 documents per second.
Luxical-One varied from 367 through 4,735 documents per second. Thus, the
reported ratio is invalid. The final protocol uses a full-workload warmup, a
fixed eight-CPU affinity, and a 20-percent spread limit for each model. It also
requires the exact model hash, rung, and baseline revision.

The float32 CPU path passed the same stable protocol on the 30M base model at
batch size 8,192. The student median was 2,951.38 documents per second. The
Luxical-One median was 3,115.60 documents per second. The ratio was 0.94729.
All five rates for each model stayed within 6% of their median. This result
removes the known graph-speed risk. It does not replace the required test of
the final projected model hash.

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

## Held-out 30M result

The 30M student keeps the same 9.3M-parameter architecture. Training used
30,000,000 rows from 146 sources. It completed 21,975 updates in 17 minutes 12
seconds.

The student is not collapsed. Its final training audit has effective rank
67.96590 and total variance 0.42014. On the fixed held-out set, parent, leaf,
and form macro-F1 values are 0.45021, 0.37371, and 0.40299. Parent macro-F1
improves by only 0.00318 from the 10M result. Every large-group level still
fails.

The model-hidden review gives 92 wins, 10 ties, and 98 losses. Its score is
0.4850, with a 95-percent interval of [0.4175, 0.5525]. The score improves by
0.0400 from 10M, but the overall and subgroup interval gates fail. Pure Arctic
scaling does not produce an approved student.

## GLM semantic projection result

The first projection used 760 GLM-labeled training documents and 190 validation
documents. A raw learned projection improves the mean semantic score by
0.01989. It reduces the effective-rank fraction from 0.31188 to 0.17904. This
raw projection fails the 0.25 rank gate.

A fixed identity-mix ladder selects a projection weight of 0.6. The mixed
projection passes the validation rank gate at 0.26271. Its mean semantic gain
is 0.01247. Folding the projection into the embedding head gives a minimum
cosine of 0.99999 against the separate projection calculation.

The fixed 10,000-document result improves all primary semantic values. Parent
macro-F1 increases from 0.45021 to 0.49406. Leaf macro-F1 increases from
0.37371 to 0.40247. Form macro-F1 increases from 0.40299 to 0.42825. Parent,
leaf, and form cluster NMI increase by 0.06417, 0.05477, and 0.04196.

All global semantic and vector-health gates pass. The effective-rank fraction
is 0.39314, and total variance is 0.87190. Large-group failures decrease from
18 to 12. The remaining failures include humanities, intellectual property,
medical text, narrative, opinion, procurement, technical documents,
instructions, unclear text, and structured data. The 760-row training set does
not support production approval.

The next private diagnostic used the first 17,250 saved GLM labels. The
selected projection weight is 0.9. Parent macro-F1 increases from 0.40407 to
0.52707. Leaf macro-F1 increases from 0.29320 to 0.41250. Form macro-F1
increases from 0.34076 to 0.42995. The mean gain is 0.11050.

The effective-rank fraction is 0.32607, and total variance is 0.77909. All
vectors are finite and unique at four decimal places. The folded output has a
minimum cosine of 0.9999956 against the separate projection calculation. All
private gates pass. This result supports the projection design, but it does
not approve a release. The fixed 10,000-document evaluation has not run on
this model.

## Capacity and input controls

The 28.4M-parameter control does not fix the semantic loss. Its parent, leaf,
and form macro-F1 values are 0.43780, 0.36117, and 0.39163. It also reaches
only 0.78952 times the paired Luxical CPU speed. More model capacity is not the
main limit.

A matched short-view Arctic teacher also fails. Parent macro-F1 decreases from
0.46499 to 0.43920. Leaf macro-F1 decreases from 0.37307 to 0.35364. Form
macro-F1 stays near 0.4113. The short teacher target loses code, research,
narrative, and technical-document quality.

The 512-token student keeps the full Arctic teacher. It reads twice as many
token positions and twice as many characters from each document region. Its
vectors pass all health checks, but its paired CPU ratio is only 0.40383. Its
parent macro-F1 is 0.43890, its leaf NMI is 0.40145, and its form macro-F1 is
0.38207. Only form F1 improves over the 256-token 3M student, by 0.00926. It
still fails all three semantic decisions. Student input length is not the main
quality limit.

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
- Parent, leaf, and form NMI and purity for one fixed 40-bucket partition are
  each no more than 0.02 below the best teacher.
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
- Use five timed repeats after a full 20,000-document warmup. Each model rate
  must stay from 0.8 through 1.2 times its median.
- Reject the CPU report when its model hash, rung, baseline revision, JAX
  backend, or compute data type differs from the evaluated release artifact.
- Record accelerator throughput for capacity planning. Accelerator speed does
  not replace the CPU gate.
- Pin the model digest, tokenizer-map digest, dependency versions, maximum input
  length, pooling method, normalization method, and backend compute data types.
- Add a loader smoke test that checks shape, finite values, nonzero variance,
  and deterministic output for fixed documents.
- Stage the exact model, tokenizer, token map, input view, and compute data
  types in an immutable runtime bundle. Run the full trust report and CPU test
  through that runtime. The release publisher must reject evidence from a
  different runtime manifest.
- Get an independent peer review of this report and its artifacts. Address each
  accepted finding before approval.

## Current artifact identity

- Pilot projection model SHA-256:
  `1aba2f7ab48841f87b63a23b9c35d154cfd4655feaa7f1808eb1d1e1dca76192`
- Pilot projection training report:
  `s3://marin-us-east-02a/marin/user/rav/luxical-arctic-ladder/manifest-v2/fast-student/full-glm-semantic-projection/pilot-1k-mix-v2/training.json`
- Pilot projection held-out report:
  `s3://marin-us-east-02a/marin/user/rav/luxical-arctic-ladder/manifest-v2/evaluation/semantic-labels/glm-5.2/pilot-1000-20260802-001/hierarchies-v1/hierarchy-1000-20260802-002/compact/heldout-10000-20260802-001/student-fast_glm_projection_pilot_1k_mix_v2/adjudicated-v1/embedding-screen-v1/report.json`
- Pilot projection CPU report:
  `s3://marin-us-east-02a/marin/user/rav/luxical-arctic-ladder/manifest-v2/fast-student/speed/cpu-trained-full-full-glm-semantic-projection-pilot-1k-mix-v2.json`
- 30M base model SHA-256:
  `981388da726eb2dff8d19dd84fff17749f2b6dd974c93ad223fee581139c9c7f`
- 30M base held-out report:
  `s3://marin-us-east-02a/marin/user/rav/luxical-arctic-ladder/manifest-v2/evaluation/semantic-labels/glm-5.2/pilot-1000-20260802-001/hierarchies-v1/hierarchy-1000-20260802-002/compact/heldout-10000-20260802-001/student-fast_arctic_30m/adjudicated-v1/embedding-screen-v1/report.json`
- 3M model SHA-256:
  `8735a4b49de0f7925904b0301516a2c8a5f9651bc2b605e4d27a80bca3f8ac3a`
- 3M tokenizer-map SHA-256:
  `50c92752d5a1d408234b8eee58c1c0f6179f603253caabe4fcd3a06f990710f0`
- 3M exact CPU report:
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
- 512-token 3M model SHA-256:
  `a0fe19c545620f5f49f4e9943c3a39d5e5d33cb8cc3464568d975c7e4bdcc43b`
- 512-token 3M exact CPU report:
  `s3://marin-us-east-02a/marin/user/rav/luxical-arctic-ladder/manifest-v2/fast-student/speed/cpu-trained-context512-context512-3m.json`
- 512-token 3M adjudicated report:
  `s3://marin-us-east-02a/marin/user/rav/luxical-arctic-ladder/manifest-v2/evaluation/semantic-labels/glm-5.2/pilot-1000-20260802-001/hierarchies-v1/hierarchy-1000-20260802-002/compact/heldout-10000-20260802-001/student-fast_arctic_context512_3m/adjudicated-v1/embedding-screen-v1/report.json`

## Open decision

Reject the pure Arctic, capacity, input-length, and Qwen-neighbor candidates.
Keep the 30M Arctic student as the base geometry. Use GLM-5.2 labels to train a
rank-preserving projection for that base.

The active label set contains 50,000 source-balanced documents. It excludes
the 1,000-document pilot and the fixed 10,000-document evaluation set. The
training code verifies the document count, identity digest, exclusion run,
sequential indices, and completed label summary before it loads the model.

Train one projection after the label job completes. Select the projection mix
on a separate 5-percent validation set. Do not use the fixed 10,000-document
set for projection selection.

The 17,250-label private result passed its development gates with a 0.11050
mean semantic gain. Complete the 50,000-label run before the final projection.

Run the fixed 10,000-document evaluation once when the internal validation
passes. Reject the model when one global gate or one large-group gate fails.
Stage the exact production runtime after training. Run the stable CPU test and
the fixed evaluation through that runtime. Start the 200-query blind review
only after the visible and CPU gates pass.

If large-group failures remain, use the same labels for end-to-end
FastTransformer training. Keep the 256-token production input and 256-number
output. Add targeted labels only for groups that the 50,000-label projection
still fails.
