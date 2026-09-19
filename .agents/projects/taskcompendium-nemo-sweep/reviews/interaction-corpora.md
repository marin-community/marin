# Interaction corpus review

The sampler was run against all ten requested collection items on 2026-09-14. Nine corpora returned two rows at pinned revisions; Structured-Outputs-v2 returned HTTP 500 and is deferred without offsets or row claims.

All inspected rows are deferred. Multi-turn and persona records could be represented as ordered history only where the source interaction is fixed, but their rubrics and LLM judges are not supported deterministic verifiers. Safety, adversarial, citation, free-form, identity, inverse-IFEval, and multichallenge records likewise depend on source-specific judge or reactive semantics. Litmus records require cheminformatics computation absent from existing contracts. Private reference responses, rubrics, and judge prompts are kept out of any proposed model-visible task.
