# Independent semantic review

`capability_pipeline.quality.run_review` runs a fresh GLM-5.3 OMP session over a
frozen build snapshot. It is separate from runtime validation: executable controls
can pass while the task measures the wrong capability or implements a different
reward function than its approved blueprint.

After the initial runtime gate, synthesis automatically freezes and runs three
fresh full runtime attempts through the repeated-evaluation controller. Those
attempts retain authored oracle grades, independent GLM-5.3 solves, controls and
attacks. Export requires all three oracle/control/attack suites to pass and at
least two of the three solver attempts to pass. Complete semantic failures go
to the GLM quality reviewer with their measured artifacts and the existing repair
budget; incomplete or invalid evidence stays pending without spending a task
repair. A reviewer cannot override failed repeated-runtime gates. This added
controller path still needs a live unattended integration run.

The next automatic diagnostic grades supported controls ten times. For Docker
SCRIPT verifiers, it runs each authored control once through Harbor and captures
the complete delivered input before private grading. Ten new private sandboxes
then grade that same capture. Original Harbor trials and direct private grades
have distinct receipts. Capture manifests preserve logical directories and modes
across byte-only snapshot transport. The live c32 probe passed all 70 regrades,
with seven capture trials, distinct private sandboxes and confirmed cleanup.
Its final 194-file snapshot was fully hash-verified. See
[the capture probe](audits/c32_captured_regrade_probe_launch_003.json).
It checks complete grading-input equality before treating different rewards as a
grading defect. Complete fixed-input scoring failures reach semantic review;
ungraded, altered or unsupported evidence stays pending. The current executor
supports single-step SCRIPT ContainerRuntime verifiers and direct runtime-null
TaskTrove modes (MCQ, math, numeric, exact, JSON schema, XML elements, CSV columns,
and IFEval), with no-tool or Docker bindings. Native capture binds the raw response,
transcript, workspace and embedded verifier resources before extraction. An
expected extraction error requires a retained grading artifact and the exact Harbor
extraction exception; it cannot count as a numeric reward. Native support is
locally tested; its exact/no-tool remote probe passed all 50 cells, including ten
expected extraction errors, with a fully verified final snapshot. Other native
modes still need live coverage. See [the native probe](audits/native_fixed_regrade_probe_001.json).
Judge and composed judge tasks continue through repeated blind calibration.
Composed tasks additionally replay every executable check ten times from the first
retained authored positive and negative deliveries. This controller is integrated;
its remote protocol validation remains pending. All retained diagnostic artifacts
are part of the protected review and repair snapshot.

The packet includes the complete final task and Harbor bundles, admitted source
capability/proposal/review, construction evidence, runtime trials, judge calibration
and independent attacks. Build dependency/tool directories outside final bundles
are omitted and declared. Each copied file has a byte hash; external symlinks and
unmaterialized directory symlinks are rejected. A changed source or snapshot
invalidates the review. Copying a file is not evidence that its contents are true.

The reviewer examines capability alignment, realism, the public contract, reward
validity, grounding and rights, isolation, and reproducibility. Every axis and
every candidate-specific blocking build condition needs artifact citations and
an explanation of what they establish. Missing evidence cannot be deferred to
future work at this stage. Outstanding required changes, major/critical findings,
scores below four, missing checklist conditions, or mismatched hashes prevent
acceptance. The review distinguishes partial-credit behavior from actual reward
bugs and checks executable/rubric aggregation against the admitted task.

The manifest also includes the recipe's shared evidence conditions: three clean
builds and oracle launches, five resets, at least two successful blind solves in
three attempts, deterministic repeat grading, resource measurements, extraction
and failure taxonomy, provenance, counterexample dispositions and task-family
declaration. Code/composite verifiers require the targeted mutation evidence.
The reviewer must assess each named condition; one successful protocol trial is
not enough to certify the stronger recipe. Model judges use their repeated blind
calibration thresholds rather than a claim that temperature zero is deterministic.
Corpus-wide split and contamination checks remain a separate publication gate.

Image identity requires evidence beyond a syntactically valid `sha256:` string.
For every candidate and private-verifier image, check the digest against retained
registry manifest metadata or actual Docker image inspection. A Daytona provider
reference's hexadecimal suffix is not, by itself, an OCI digest. Likewise, an
example task's local image ID does not establish availability on another worker.
Require the exact recipe/context, the provider mapping used for measured trials,
and a supported cold-reconstruction path. Keep provider snapshot identity and OCI
image identity distinct. A successful trial against a pre-existing snapshot proves
that measured execution only; it cannot establish portable reconstruction.

`review.json` is the model's receipt; `result.json` is the controller's checked
outcome. Acceptance here is a semantic judgment over the named snapshot, not a
runtime or publication certificate. A same-family model reviewer can share blind
spots with builders and solvers; the raw evidence and its limitations remain
available for external review. Unit tests exercise snapshot, citation, mutation
and contradictory-acceptance gates with clearly labeled fixtures. No live final
task semantic review has passed yet.
