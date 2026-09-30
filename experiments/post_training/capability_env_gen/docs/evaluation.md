# Repeated runtime evaluation

`evaluate-plan` freezes a task package, TaskSpec bundle, controls, controller
source, pinned dependency manifests and inference settings. `evaluate` runs three
fresh full Harbor suites on a configured cluster worker. Each suite contains all
authored oracle cases, one independent solver attempt per positive case, authored
negative controls and the three independent attack strategies.

These are **fresh attempts**, not provider-seeded trials. The current inference
interface does not attest a provider seed. A failed solver remains in the
denominator; no semantic retry or early stopping selects a better answer.
Transport-level output-budget fallbacks remain separately recorded by the runtime.

Synthesis uses the same evaluator in a separate `primary_bound_repeatability`
mode after its original independent attacks have passed or received resolved GLM
adjudication. The plan hashes that primary gate. Its three fresh attempts measure
oracle, solver and authored controls; they do not launch additional unreviewed
attacks. The matrix reports `primary_adversarial_review_bound` separately. The
standalone command above retains its full attack suite.

Prepare the plan with metadata-only local work:

```bash
uv run --frozen capability-pipeline evaluate-plan \
  --package /path/to/harbor --bundle /path/to/contract \
  --controls /path/to/controls.json --out /path/to/evaluation-plan.json
```

The command prints the plan SHA-256. On a worker with the pinned TaskCompendium
source, Daytona and GLM credentials, and durable result uploads configured, run:

```bash
uv run --frozen capability-pipeline evaluate \
  --plan /path/to/evaluation-plan.json --plan-sha256 PRINTED_SHA256 \
  --taskcompendium-source /path/to/pinned/taskcompendium \
  --out /path/to/new/evaluation-output --concurrency 3
```

The generic `scripts/submit.sh --stage evaluate` dispatcher accepts a self-contained
bundle directory containing `plan.json`, its inputs and `manifest.json`. The
manifest uses schema `capability-runtime-evaluation-bundle-v1`, `plan_sha256`, and
a `files` map from relative names to SHA-256 (excluding the manifest itself).
All plan input paths must remain inside that directory. For example:

```bash
scripts/submit.sh --stage evaluate --source data/TASK-evaluation \
  --out runs/TASK-evaluation --run-name cap-TASK-evaluation \
  --concurrency 3 --tier interactive
```

The launcher verifies the complete transport and staged controller identity before
Iris submission; the worker restores once into a fresh directory and uploads
results incrementally. Do not run inference or generated task code locally.
ShellSim additionally requires a bridge file supplied to
`evaluate-plan --shellsim-bridge`; its bytes are frozen too.
Declared control workspaces are resolved relative to `controls.json`, checked
when creating the plan, and bound by file/directory inventories. Keep that file
beside its original replay assets when constructing an evaluation bundle. A
relocated file with unchanged bytes can still have invalid relative references;
validate the packed-and-restored bundle before submitting it.
Daytona candidate environments **or private verifiers** require
`evaluate-plan --daytona-helper /path/to/frozen/tools/dt.py`. Copy the maintained
helper into the self-contained input bundle first. The plan hashes that file and
the evaluator derives `CAPABILITY_DAYTONA_TOOLS` from its frozen location; an
ambient helper path cannot override it. This also applies to no-tool tasks whose
grader runs in a private Daytona sandbox.

For a Docker candidate, `--candidate-resources /path/to/resources.json` freezes an
explicit request such as `{"cpu": 2, "memory_gb": 2, "disk_gb": 10}`. The runtime
binds those bytes in its attestation and passes the request to the Daytona adapter.
Snapshot cache identities include the requested capacity. Other environment kinds
reject this option. It changes only candidate resources; private graders retain
their separate resource configuration.

The candidate adapter records startup and final cgroup samples, including raw
output and hashes, before deleting the sandbox. Missing or unlimited limits and
probe failures stay explicit. These observations do not themselves establish that
the provider enforced the requested resource envelope; disk usage and a complete
resource acceptance gate remain outstanding.

The repeated runtime gate requires all three evidence-valid suites, three oracle
successes, at least two solver successes, and passing authored and independent
negative checks in every suite. The plan freezes `critical_control_ids` from
nonpositive controls whose declared `reward_max` is zero. An explicit `critical:
true` must declare a graded exact-zero negative/malformed case; contradictory flags
are rejected. Those critical controls must receive exactly zero reward. Other
controls retain all preregistered status/range expectations, including noncritical
partial-rubric credit and null-reward extraction failures. Completeness of the
critical-negative inventory remains unassessed. Rewarded independent attacks
are reported for review. Reported Daytona sandbox IDs must not recur between
attempts. This check does not certify all reset/lifecycle properties of no-tool or
ShellSim environments.

Every attempt has raw artifacts, a receipt and an explicit payload manifest.
The payload manifest excludes itself and `receipt.json`; the receipt hashes that
manifest, and `matrix.attestation.json` hashes all receipts and the matrix.
Input and controller hashes are checked before execution and again at the end;
the attestation retains initial/final hashes. Existing output directories cannot
be overwritten. Interrupted output remains evidence and requires a new evaluation
directory for a fresh campaign.

The report records a hash of the live relay endpoint and aggregates returned model
names/fingerprints. Missing fingerprints remain unknown; the report does not claim
a homogeneous model revision. Elapsed controller time and artifact bytes do not
substitute for remote peak memory, disk or startup measurements.

This command does not write acceptance or synthesis status. Even a
`repeated_runtime_passed` result leaves the broader recipe's build reproducibility,
reset determinism, ten repeated grades, full extraction taxonomy, mutation/judge
calibration, resource envelope, provenance and split/contamination rows unassessed.
It is not a training-admission decision. Live evaluation evidence is required in
addition to controller unit tests.

## Ten identical-input grades

`regrade-plan` and `regrade` freeze and run ten grades per authored control with
no model calls. Supported single-step surfaces are native deterministic grading
and a container SCRIPT verifier with a no-tool, Docker, or ShellSim candidate.
Docker and ShellSim replay use the same checked workspace and command staging
as runtime controls. Every cell retains its outcome and candidate evidence;
SCRIPT grading also retains separate private-verifier isolation evidence.

The private grader fingerprints its actual response, extracted submitted files,
transcript, specification and protocol. A case passes the fixed-input condition
only if all ten complete grading-input fingerprints match. Repeating a command
that generates different file bytes or transcript content fails this condition,
even if all rewards match. This mechanism checks equality after execution; it does
not yet capture and freeze one submission for replay. For SCRIPT grading, the
candidate submission is captured once per authored control and its immutable
capture is graded in ten fresh private Daytona sandboxes. Composite verifiers
and judge determinism remain outside this diagnostic's supported scope. A result
does not change the task's repair budget or admission status.

An authored ShellSim infrastructure probe passed the native route with 20 fresh
fixed grades and the SCRIPT route with two captured submissions, 20 private
grades, and real workspace replay. The raw, hash-verified evidence is recorded in
[the ShellSim regrade audit](audits/shellsim_regrade_probe_003.json). These
fixtures validate the grading machinery, not generated catalog task quality.
