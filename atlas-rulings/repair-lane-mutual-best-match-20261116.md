# Ruling: repair-lane mutual-best matching (2026-11-16)

The repair-candidate lane's nearest-neighbor matching uses the **mutual-best rule** for
every candidate folder, by user ruling recorded 2026-11-16 (relayed verbatim: "Approved,
make this a rule.").

## Rule

After template stripping (shared leading setup sections removed from both sides), a
candidate's nearest tracked neighbor classifies as `repaired` only when **each task is the
other's unique best neighbor** and the similarity is at least 0.5. Non-mutual nearest
neighbors and several candidates claiming one tracked task stay `conflicting` (manual
ruling, nothing replaced by itself). Exact hash identity keeps its own class. The
comparative judge, oracle and NoP gates are unchanged and still vet every replacement.

## Origin

Empirical dry runs showed that stripping shared template boilerplate lowers true-pair
jaccard (it had been inflating similarity), so a one-way 0.9 threshold was unreachable for
paraphrase-level repairs; mutual-best matching at ≥ 0.5 restores the intended recall with a
standard uniqueness criterion. Approved for NL2Bash and made standing for the lane.

## Application

Codified in `experiments/post_training/task_curation/repair_match.py` (`match_candidates`,
`_classify`) and documented in `docs/references/task-curation.md` ("Repair candidates").
