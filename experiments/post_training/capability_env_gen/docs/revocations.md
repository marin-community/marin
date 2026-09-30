# Accepted-input revocations

`data/revocations.json` is the append-only source of truth for proposal hashes that
must no longer enter synthesis even if an older admission record says `accepted`.
Each record binds the capability and slot, a concise reason, immutable evidence
hashes, and the only permitted reuse. Removing or weakening a record requires a new
reviewed repository change; a replacement proposal receives a new hash rather than
mutating the revoked record.

Normal synthesis fails closed on every `state: revoked` proposal. A repair or
re-admission controller may opt in to `allowed_use: admission_repair_input_only` so
the exact failed proposal can be supplied as repair context. That bypass must never
reach construction, runtime validation, export, or publication.
The admission controller also refuses an accepting model verdict on an unchanged
revoked hash. It retains that raw verdict, records the controller block, and uses
any remaining repair round to address the revocation before fresh review. The
original defect remains visible to reviewers after the proposal hash changes;
a cosmetic change is not evidence of resolution.

The first record revokes c30 redesign 001 after deterministic audit showed that its
equal-weight native mean rewarded severe causal, window, and action errors above
0.84 despite labeling them negative. The full arithmetic and retained-run hashes
are in [`audits/c30_redesign_001.md`](audits/c30_redesign_001.md).

Two refinement-003 proposals were subsequently revoked after independent
arithmetic checks: c12's servo task gives an incorrect least-squares reference,
and c18's payment task treats a Luhn-invalid card as valid, changing the request
classifications and totals. The original 100-proposal acceptance report remains
unchanged; 98 of those entries still pass the structural loader. The measured
constants, formulas and results are retained in
[`audits/simple_pilot_arithmetic_003.json`](audits/simple_pilot_arithmetic_003.json).

Six proposals newly accepted by review005 are revoked after a stratified static
source audit found direct internal contradictions:

* c08 labels the looser clipping threshold as a masker while its own expected
  experiment says the tighter threshold delays divergence;
* c19 requires a no-valid-case citation scraper to pass a Stage-1 gate that
  requires at least two valid cases;
* c21 requires public `R1`-`R12` keys whose mapping exists only in private
  evaluator data;
* c23 requires every escalation threshold to be historical-pack-derived while
  its stated recurrence/remount examples are not observable in that pack;
* c32 fixes a 94-item queue using an unsupported deduplicated count of eight;
* the logic-verification task excludes negative balances from its state domains
  while requiring a negative-balance counterexample, and its repair grammar
  cannot express one advertised repair.

Each record binds the exact proposal hash to
[`audits/proposal_review_005_new_accepts_sample.md`](audits/proposal_review_005_new_accepts_sample.md).
That audit retains c05 and c13 as construction clarification gates rather than
revocations. The review005 acceptance artifacts remain immutable; normal
synthesis rejects these six unchanged hashes, while the repair-only bypass may
use them as context for a new hash and fresh independent review. Together with
the c12/c18 records, this leaves 192 of review005's historical 200 accepts
structurally loadable; the entry-by-entry loader result is frozen separately in
[`audits/proposal_review_005_revocation_status_20260918_r2.json`](audits/proposal_review_005_revocation_status_20260918_r2.json).
The prior eight-record status sidecar remains immutable historical evidence.

The original admitted c02 slot-10 proposal is separately revoked after its
construction audit exposed an unresolved reward contract: thirteen 0-3 items
have raw maximum 39, while the proposal states a 20-of-24 pass rule and leaves
the two-judge/third-adjudicator outcome implicit. The exact hash may enter only
the repair admission prepared in `data/c02_slot10_clarification_input.json`.
That clarification must preserve all thirteen partial-credit anchors and machine
gates, normalize `raw * 24 / 39` with threshold 20 (integer raw at least 33),
and undergo a fresh GLM repair and independent rereview. That admission completed
with new hash `18bae8109572090a70629cb75a1611f79241f366d9ef92a3e8e5c54c01354f4e`;
the static receipt audit is
[`audits/c02_clarification_admission_001.md`](audits/c02_clarification_admission_001.md).
That new hash was subsequently revoked too: the
[anchor-threshold audit](audits/c02_anchor_thresholds_002.md) found that several
binary predicates contradict the retained partial-credit anchors. The first
receipt audit's anchor-preservation conclusion is superseded. Repair input
`data/c02-anchor-repair-002/accepted.json` preserves the rejected hash and all
lineage for a fresh repair and independent review. Neither c02 hash is in
review005's 200 accepted set, so these revocations change the registry total
but not review005's 192/8 current count.

The subsequent [semantic-boundary checklist](audits/c02_semantic_boundary_checklist_003.md)
corrects a provenance overstatement in the anchor audit: the original proposal
fully specified intermediate levels for nine groups, but the other four groups'
intermediate anchors were introduced during clarification. Their internal
anchor/threshold contradictions still justify revocation. Future repairs must
label these levels as reviewed interpolations, rather than claim every level
was present in the original proposal.

The bounded explicit rejected-proposal repair subsequently obtained fresh
acceptance for `03c05bb576832e76c656e2ebc74b537af8835a2c345a8ef69cf0e366918d7b4d`.
The [new receipt audit](audits/c02_anchor_repair_admission_003.md) records repaired
threshold predicates and remaining construction conditions. Both preceding
hashes remain revoked; acceptance of a replacement does not erase their history.
