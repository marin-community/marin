Design a portfolio of exactly {n} genuinely different task proposals for this capability.
Diversity means different workflows, artifacts, failure modes, and reasoning, not renamed entities or
numbers. Choose environment and verification on validity, not to fill a Cartesian-product quota;
choose each slot's verification by the verification rule in the system prompt. Use
an environment only where the capability's work is actually done in it: a capability whose records
are pure reasoning (derivations, judgements, calculations a person does on paper) belongs in
reasoning, and shell or container slots added for variety are a defect.
This host can run these environments: {environments}. Propose no slot in another
environment.
At most {max_per_pairing} proposed slots may share the same environment x verification pairing,
code excepted: its slots differ by workflow.
Never give the solver an environment or tool that performs the capability's own operation (for exact
symbolic evaluation, no computer algebra system; for unit conversion, no conversion library), and
never plan a slot that exercises a neighbouring capability instead of this one.
Do not copy a sample_task verbatim. Use the source sampling_facets to diversify meaningful workflows
and failure modes. When learning_progression edges are supplied, use their enabled_scope,
transfer_basis, witnesses, and artifact_substitution_test to keep this capability's new operation
essential: prerequisite work alone must not earn success. Absent edges do not prove the capability
has no prerequisites.

Fewer than {n} genuinely different tasks is an acceptable answer: mark each slot you cannot fill with
a real, distinct workflow as null; a padded slot is worse than a null one. To rule out several
verifications for one environment across the whole portfolio, list each pairing separately among the
excluded combinations, and keep limitations of a single slot to the coverage rationale.

For each slot decide research_need. It is {research_criterion}

CAPABILITY RECORD:
{capability_record}
