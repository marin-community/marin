# Task requirements

[Planning overview](index.md)

These requirements define the future generation pipeline and belong in shared authoring instructions. They do not impose compatibility with the earlier implementation.

## Scientific task

- Ask a concrete scientific question with a declared input stage and complete, checkable outputs. Both focused analyses and connected workflows are eligible.
- Use observed data as the corpus backbone, not a requirement for every task. Label observed, adapted, simulated and mixed inputs. Reduced datasets must preserve meaningful complexity, including biological replication and estimable comparisons where the scientific design requires them.
- Exercise native formats and their semantics, including FASTA/FASTQ, BED, SAM/BAM/CRAM, GFF/GTF, VCF/BCF, PDB/mmCIF, H5AD, matrices, trees and images where relevant.
- Provide an executable reference that reads the task inputs and runs the actual packages. Define the scientific contract before generating expected results.
- Package a terminal-based Harbor task. A notebook may be a source, but there is no separate notebook protocol.
- Preserve the scientific objective when reframing an output for deterministic scoring. Exclude or reframe questions whose essential result requires subjective judging.
- Record provenance and establish redistribution eligibility before release. Follow [storage and publication](storage.md).

## Daytona sandbox resources

Use CPU-only Linux task environments through Harbor's Daytona backend. Each task must declare resources and timeouts; validate at those allocations. A successful run on a larger authoring machine is insufficient.

| Resource | Requirement |
| --- | --- |
| CPU | At most 4 vCPUs per sandbox. Bound tool and numerical-library threads by the declared allocation. |
| Memory | At most 8 GiB per sandbox. Measure peak memory including child processes and leave headroom. |
| Disk | At most 10 GiB per sandbox. Include the environment, staged and decompressed inputs, indexes, temporary files, outputs and logs. |
| GPU | Zero GPUs for solving, reference execution and grading. External hosting of the agent model is separate. |
| Solve timeout | A finite per-task timeout with headroom, informed by the runtime distribution below. |
| Setup and verifier timeouts | Finite and declared separately from solving. Do not move required analysis work into unbounded setup. |

The CPU, memory and disk envelope follows [Daytona's standard sandbox limits](https://www.daytona.io/docs/sandboxes#resources), checked on 2026-09-25. Verify the account's effective limits before launch; no account-specific quota was inspected for this plan. Per-sandbox limits, organization-wide pools and API rate limits are distinct. See [Daytona limits](https://www.daytona.io/docs/en/limits/).

Allocate according to measured need; tasks need not reserve the maximum. If a workflow does not fit, choose a scientifically valid input boundary, reduce data without invalidating the analysis, or defer it. GPU-dependent training and large structure-prediction jobs are outside the initial scope. Analysis of supplied structures or predictions is eligible when that starting point is explicit.

Stage task inputs, analysis reference data (such as genomes and annotations) and pinned dependencies before solving. Keep oracle solutions and expected answers outside the solver environment. Solving is offline by default, with no required cloud credentials, external compute or live database service. Explicit retrieval tasks must declare their network dependency and use a stable grading target. Grading stays offline. Validate workflow-manager execution requirements in the sandbox, not just its component tools.

## Runtime distribution

Favor fast iteration with a smaller longer-running tail. These are provisional agent solve-time ranges, not difficulty labels or quotas:

| Intended solve time | Role |
| --- | --- |
| 2–10 minutes | Most tasks: focused analyses or compact connected workflows |
| 10–20 minutes | A substantial minority with more decisions, integration or computation |
| 20–30 minutes | Selected workflows where the time adds scientific realism |
| Over 30 minutes | Exceptions considered individually for scientific value |

Thirty minutes is a heuristic, not a hard ceiling or platform limit. Do not pad short tasks to meet a lower bound. Measure a varied set before choosing percentages.

Track reference runtime and agent solve time separately: the reference executes a known solution, while the agent also inspects data, chooses methods and writes code. Record the model, configuration and service conditions with trial timings. Model latency is not an intrinsic task property or part of the scientific contract.

Minimize setup and grading time using prepared environments, staged inputs and precomputed reference results where appropriate. Retain executable references and independent checks. Remove waiting and redundant computation before reducing scientific complexity. A count-matrix task can retain all biological replicates; separate tasks can exercise read processing.

## Deterministic rewards

- Reward is a deterministic function of submitted artifacts, pinned grading data and versioned verifier code. No LLM judge, grading-time model call, human judgment or live external lookup is allowed. LLM-assisted authoring and scientific review are separate.
- Specify required files and fields, sample/feature identities, units, eligibility, contrasts, missingness and numerical conventions. State permitted alternative methods or representations.
- Check discrete identities and structural requirements exactly. Declare justified absolute/relative tolerances and boundary behavior for numerical results. Valid floating-point outputs need not be byte-identical.
- Pin reference data, package versions and relevant numerical settings. Control randomness where appropriate and demonstrate reproducibility; a fixed seed alone is insufficient. Reject unstable reward behavior.
- Repeated grading of identical artifacts must produce the same reward. Reference reproducibility is a separate check. Test scientific mistakes, incomplete/malformed artifacts and trivial submissions.
- Keep expected answers and verifier code outside the solver's writable environment. Grade only declared submissions in a trusted environment. Public publication of those assets does not authorize mounting them into the solver sandbox.
- Diagnose infrastructure failures and verifier crashes separately from incorrect scientific answers. Make any partial-credit rule explicit and executable.

Verifiable rewards need not establish biological truth. On observed data, they can establish correct execution of the declared analysis. On simulated data, sampling noise may prevent exact recovery of the generating truth; truth-based criteria must account for that uncertainty. If multiple methods are permitted, agreement with one package's reference output alone is insufficient.
