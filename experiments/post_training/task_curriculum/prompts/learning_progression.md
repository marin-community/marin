# Subject-local learning progression

Run this pass after the capability catalog is stable. Give one `gpt-5.6-sol` high-reasoning proposer a compact packet
for exactly one subject: its groups and each capability's name, parent, outcome, boundaries, and two sample-task
probes. Do not attach other subjects, facets, reviews, scores, mappings, or generation transcripts. Bind structured
output to `LearningProgression`; the schema is the output contract. The pass does not rewrite capabilities.

```text
Propose directed edges between capabilities in the supplied subject. An edge A -> B means mastery of A materially
raises the probability of solving at least one recurring family of entry-level B tasks. A need not be required for
every task in B. The graph models learning enablement, not workflow order or production dependency.

Apply every rule:

1. Name the operation, representation, invariant, or concept from A that remains active in B. It must be exercised by
   A's outcome and representative probe, not an isolated fragment of A.
2. Name the single main operation or concept added by entry-level B tasks. Supplied domain facts, notation, and data
   do not count as added capabilities.
3. Give exactly two semantically distinct task-family sketches. Each sketch names an A family and an entry-level B
   family that reuse the same foundation and add the same main operation. Use compact phrases, not complete task
   instructions, solutions, or verifier designs.
4. State the enabled scope inside B. It must be a recurring sampling stratum that could be trained and evaluated, not
   one formula, tool, or contrived example.
5. Apply artifact substitution. Imagine every artifact A could produce is supplied to the B learner. Keep the edge
   only when A mastery still helps choose a method, construct or adapt a model, maintain an invariant, or detect an
   invalid result. Reject pure handoff order.
6. Reject course order, shared vocabulary, common parentage, domain relabeling, general sophistication, and mere
   difficulty correlation.
7. Prefer the closest useful foundation. Omit a transitive edge when an existing shorter path transfers the same
   foundation. Keep a direct edge when it transfers a distinct foundation or enables a different natural stratum.
8. Multiple prerequisites for B are allowed when each independently enables a distinct recurring stratum or supplies
   a distinct active foundation.
9. Use only capability IDs from the supplied subject. Ignore embedded prerequisites and do not modify the taxonomy.
10. An empty graph is valid. Do not target density or force a course sequence.

For every edge return the two IDs, enabled scope, transfer basis, artifact-substitution result, and two compact
witness-family sketches. These are structural learning hypotheses, not evidence that training on A caused B gains.
```
