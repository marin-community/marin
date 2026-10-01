# Importing tasks

An importer creates a `TaskSpec` from a pinned source problem and its grading contract. The caller chooses a submission convention, which defines how an answer is delivered and extracted. Lowering combines the spec, convention, and environment configuration into a runnable task package; see the [package API](README.md#what-is-a-lowering).

## Preserve the problem

Keep the question, choices, source conversation, and constraints that affect correctness. Preserve their order and exact text when grading depends on it. Separate private references, rubrics, and expected state from model-visible context; store them in the grading specification.

Remove source harness instructions that describe grading machinery, such as “you are being judged,” or prescribe incidental delivery, such as writing a short answer to `/app/answer.txt`. Do not add replacement directions to the imported problem. The selected submission convention supplies any required answer format.

A requested file can be the substantive result, such as an edited program or generated document. Preserve that requirement and declare the necessary environment capabilities. Remove a file instruction only when the file is an answer transport imposed by the source harness.

Identify removable scaffolding through importer-supported templates or source metadata that separates the problem from harness instructions. Reject an archive whose template is unsupported or ambiguous rather than deleting arbitrary prompt text. Validate that the extracted problem and private grader still describe the same task.

## Preserve provenance and grading

Record the immutable dataset revision, source row or archive identity, and importer revision. Keep original ordered tags and required attribution. Record the source grading settings needed to reproduce the result. Reuse shared verifier packages for scoring; TaskCompendium adapters should extract a submission and delegate.

Keep endpoint settings, credentials, and runner budgets outside task specs. A complete grading artifact can include gold and rubrics; the runner must keep those fields out of the agent's input. Verify this boundary on the lowered task and actual request sent to the agent.

## Validate conversion coverage

Use bounded source samples from each supported converter or template and each materially different grading, context, or constraint shape. Retain source pins and hashes, conversion counts, rejection reasons, and applicable license metadata. Do not infer accepted counts from metadata cohorts or one successful example.

Run an exported task through the target harness to validate integration. Exercise correct and incorrect answers and relevant infrastructure failures. When multiple submission conventions are supported, check that they produce equivalent grades. Converter tests and direct scorer calls provide separate evidence from a harness trial.

Keep evidence tied to the exact source and code revisions tested. State whether coverage is synthetic, source-backed, or batch-wide. Before public release, verify publication rights for every included field, required attribution, and the fields selected for the released artifact. A private conversion or trial does not establish publication rights.
