# Validation and release readiness

[Planning overview](index.md) · [Task requirements](requirements.md)

An independently controlled harness validates each candidate using executable checks, independent trial solves and LLM review of task design and solver behavior. A model's recommendation must cite evidence and cannot substitute for required execution or scientific review. Passing an oracle alone is insufficient: the oracle and grader can share the same error.

## Required evidence

1. **Scientific contract and data eligibility.** Check the question, experimental units, estimable comparisons, provenance and complete outputs. Verify the recipe's predeclared data-adequacy criteria on the delivered inputs and record the results, including any subset adaptations. Confirm [publication eligibility](storage.md#release-eligibility) before release, following the earlier pre-build screening.
2. **Fresh sandbox execution.** Build the pinned environment and run the native reference from task inputs under the declared [resources](requirements.md#daytona-sandbox-resources). Validate workflow execution, not just package loading.
3. **Reference and grader repeatability.** Check native-reference reproducibility separately from repeated grading of identical artifacts. Require the same reward for identical submissions.
4. **Independent checks.** Use independent calculations, parsers or invariants where feasible. Record any assumptions or code shared with the reference, so shared-bug risk is visible.
5. **Incorrect submissions.** Execute plausible scientific mistakes and trivial artifacts through the grader: sample-join errors, reversed contrasts, wrong counting conventions, incomplete tables, malformed output, empty output and copied input where applicable. Record expected and actual outcomes. Controls must distinguish meaningful errors for this task.
6. **Runtime and isolation.** Record CPU allocation, peak memory including children, peak disk, setup time, reference time, grading time and trial-solve time where applicable. Confirm that solver writes cannot modify verifier code or expected results. Clean up owned sandboxes.

Diagnose infrastructure failures and verifier crashes separately from scientific errors. Fix the cause and rerun affected checks; do not relax scientific tolerances merely to obtain a pass.

## Independent trial solves

Require an independent trial solve for the initial examples used to calibrate authoring. Give the solver only the instructions, input files and tools available in the released task. Do not provide the author's reference, expected outputs or private-to-the-trial grading evidence.

Run its artifacts through the executable grader, then inspect failures for incomplete instructions, ambiguity or environment problems. A solver failure alone does not establish a task defect. Resolve identified defects and repeat affected checks. A worker that has seen expected answers cannot serve as that task's blind solver.

These trials validate usability; collecting teacher traces for downstream training remains out of scope. Define the ongoing sampling policy before scaling, and retain human spot checks of scientific validity.

## Reflection on solver attempts

Give a validation reviewer the task instructions, inputs, reference, grader, automated results, and the independent solver's commands, intermediate outputs, errors, final submission and reward. Its job is to assess what the attempt reveals about the task. This review uses a separate context from the author and the blind solver; the reviewer may inspect hidden reference and grading artifacts.

Require findings to cite concrete task artifacts or trace events. Examine whether:

- The solver performed the intended scientific work or found a shortcut that bypassed it.
- Instructions and inputs supported the required decisions, including points where the solver struggled.
- The reference and grader preserve the scientific objective and handle reasonable alternative solutions.
- The grader accepted a scientifically wrong result or rejected a valid one.
- Changes to instructions, inputs, environment or grading would address an observed problem.

The same reviewer can perform adversarial inspection and trace reflection. It may construct alternative valid answers and deliberately wrong submissions for execution through the grader. Preserve expected and observed outcomes; an unexecuted critique remains a hypothesis.

Return an accept/revise/reject recommendation, supporting evidence, unresolved questions, and concrete repairs with the checks or trials to repeat. A successful solve alone does not establish task quality, and a failed solve alone does not establish a defect. Distinguish infrastructure failures, incomplete attempts, task ambiguity and scientific mistakes. One trace can expose a specific problem; use additional attempts when a proposed revision depends on whether behavior is systematic or solver-specific.

Send findings back to the author, record the resulting task revision, and rerun affected checks and independent trials. The reviewer supplies development feedback; released rewards remain deterministic functions of artifacts and verifier code. Human scientific spot checks remain part of the review policy.

## Acceptance record

Record exact task, input, environment, prompt and validator revisions; the scientific contract; resource measurements; native-reference results; repeatability checks; independent checks; incorrect-submission outcomes; trial traces and reviewer findings where required; repairs and subsequent validation results; and redistribution eligibility. Link artifacts and unresolved concerns.

A released task needs passing executable evidence, resolved scientific ambiguities and publication eligibility. Keep other candidates explicitly pending or rejected. Do not count discovery, implementation or a package-loading check as a released task.

The author cannot edit the harness's acceptance decision. After release, link corrections to a new version and retain prior provenance. Reward changes require revalidation of the affected task and controls.
