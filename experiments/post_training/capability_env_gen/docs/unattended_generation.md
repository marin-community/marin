# Running the proposal-to-construction pipeline

`generate` runs GLM-5.3 proposal planning, independent portfolio review, bounded
proposal repair, and construction of accepted proposals in one worker. Each task
then runs the existing runtime and semantic gates. Measured actionable build
failures and independently adjudicated exploits enter bounded GLM construction
repair immediately, with fresh validation after edits. No Sol/Terra session is
part of this runtime path.

Code-and-judge composition is an authorized task design. Builders may use the
maintained TaskCompendium extension to combine mandatory executable checks with
native rubric scoring; a failed mandatory check prevents credit. The recipe must
generate, test and repair these tasks through GLM-5.3 as well. Missing adapter
evidence remains a pipeline implementation gap, rather than a reason to remove
composition from proposals.

Synthesis also runs a frozen three-attempt oracle/solver/control/attack evaluation
before final quality review. Its receipts stay with the item, and failed or missing
attempts remain in the denominator. For supported single-step executable private
verifiers and direct deterministic native verifiers, synthesis then runs every
authored control through ten grades, verifies equal
grading-input bytes, and checks reward stability and expected scores. Complete
fixed-input scoring failures reach GLM quality review and bounded repair. Changed
input bytes, missing evidence and unsupported executable surfaces stay pending;
they are not evidence of a faulty task. Judge tasks use their preceding repeated
blind calibration gate. Composed verifiers also replay each executable check from
the first retained authored-control delivery; the judge's rubric scores remain
separate. These new controller paths still need live unattended
integration evidence and do not complete clean-build, reset, resource or
training-admission checks.

Docker builders now author `task/reset-policy.json` and an explicit
`task/candidate-resources.json`. After runtime diagnostics, the controller freezes
the task package, policy, capacity request and adapter sources. A separate baseline
start measures public file hashes; five fresh starts repeat the exact public-input
staging and setup commands and compare the measured state. Completed mismatches
enter GLM repair; missing probes, unconfirmed deletion and provider failures remain
pending. An interrupted attempt is retained rather than sampled again. The remote
Docker protocol probe passed a separate calibration start, a baseline and five
cycles, with seven distinct sandboxes and confirmed deletion. All 41 probe files
passed independent hash verification. Its policy was authored for the protocol
fixture; GLM policy authoring still needs generated-task evidence. See
[the Docker reset audit](audits/docker_reset_probe_001.json). Public-state
comparison does not establish private-data isolation or a complete resource envelope.

No-tool tasks use five fresh capture-only Harbor episodes and compare the actual
agent-visible instructions with the canonical exported instructions for every step.
Those episodes invoke neither a solver model nor a task grader. The remote no-tool
protocol probe passed all five episodes; independent verification matched all 59
retained probe files. See [the probe audit](audits/non_docker_reset_probe_002.json).
Broad generated-task integration remains to be verified. ShellSim now uses a
pinned bridge extension to compare complete bounded VFS snapshots and canonical
public prompts after actual Harbor startup. Its production reset controller
passed a remote calibration plus five fresh episodes, including Unicode paths;
independent verification reproduced the result from all 85 probe inventory files.
See [the ShellSim reset audit](audits/shellsim_reset_probe_008.json). This proves
the fixture's VFS reset path, not private or non-VFS state isolation.

Docker SCRIPT controls now run once to capture their actual submission and complete
grading input; ten fresh private graders replay the capture. This preserves legitimate
random data and timestamps. The original Harbor receipt and each direct grade remain
separate, and incomplete captures remain pending. Snapshot restoration reconstructs
logical directory and file modes from the authenticated capture manifest. This
integration passed the live c32 captured-input probe: all 70 grades met expectations
on identical captured input, and all 77 private sandboxes had confirmed cleanup.
This establishes that fixture's replay path; broad unattended integration remains
to be verified. See [the probe audit](audits/c32_captured_regrade_probe_launch_003.json).

The composite adapter also retains each executable check's exact grading input,
with its check index and source hashes. Capture metadata is attached after judging
so it does not enter rubric context. Automatic ten-grade replay of every captured
check is integrated into synthesis, using the first authored positive and negative
controls. Completed fixed-input score changes reach GLM repair; missing captures
remain pending. Remote protocol validation passed three captured controls and
30 fixed-input grades, with confirmed cleanup and exact artifact closure; see
[the composed replay audit](audits/composite_fixed_replay_protocol_probe_003.json).
The normal generated-task path still needs end-to-end evidence.

Long solver requests retain their full history. If the server explicitly rejects
the reserved output budget for exceeding context length, the transport retries
the same messages and tools with the remaining available output budget. A remote
replay reproduced the original c32 32K-reservation failure and completed with this
fallback; no returned tool was executed by that probe. This establishes transport
recovery for that request, not task acceptance or recovery from a full prompt window.
See [the frozen-request audit](audits/context_budget_probe_001.json).

Prepare a development cohort and submit it with one command per step:

```bash
uv run --frozen -m capability_pipeline.catalog new_catalog.json \
  --cohort-json data/new-catalog-cohort-001.json
scripts/submit.sh --stage generate --pilot data/new-catalog-cohort-001.json \
  --out runs/new-catalog-recipe-001 --concurrency 256 --tier interactive
```

For full catalog input, replace the cohort command with `--all-json
data/new-catalog-all-capabilities.json` and pass that manifest to `--pilot`.
The full manifest covers 1,999 capabilities and 19,990 proposal slots. The cohort
covers 45 capabilities and 450 slots; it does not establish full catalog coverage.

For a fresh run distributed across workers:

```bash
uv run --frozen -m capability_pipeline.fleet plan \
  --pilot data/new-catalog-all-capabilities.json --out data/fleet-plan \
  --run-prefix runs/catalog-fleet --workers 4 --concurrency 256
uv run --frozen -m capability_pipeline.fleet submit --plan-dir data/fleet-plan
```

The plan assigns each whole capability to exactly one worker, retains its ten
slots and applicable learning-progression evidence, and checks the complete union
before launch. Four workers each request 16 CPUs, 128 GB and 64 concurrent sessions;
aggregate concurrency remains 256. These allocations respond to the observed
64-GB OOM with 260 OMP processes; they still need load validation. Submissions run
concurrently using separate locks and immutable staging directories. A started
submission without a receipt remains pending observation rather than being
submitted again. A shared local launch ledger under
`~/.cache/capability-pipeline/fleet-launches` prevents copied plans from relaunching
the same workers. Generation identities separate different plans' output prefixes,
and a remote-empty preflight runs before any source archive is written. Source
changes invalidate the plan. This path is for fresh
capability runs; it does not migrate interrupted construction or reset its repair
budgets. Each shard produces its own normal coverage report; automatic fleet-wide
result collection remains to be integrated.

The output freezes `input-pilot.json`, behavior-affecting settings, controller and
helper content, the pinned TaskCompendium source lock, and the construction repair
budget. `proposal/` retains all proposal outcomes; `construction/` retains builder,
runtime, review, and repair evidence. Frozen stage receipts bind their contents.
Interrupted proposal work reuses its exact request cache until construction starts.
After construction starts, proposals are immutable. Construction resumes its same
directory and per-item budget. The default construction budget is two rounds;
changing a run name or resuming does not replenish it.

To resume a terminal interrupted worker, use the same output prefix and inputs
with `--resume`. The source/controller and effective settings must still match
the frozen run; incompatible changes require an explicit migration rather than
silently reusing old evidence. Do not submit a second worker to a live run.

If a terminal worker lost its controller before construction began, a fresh
controller may adopt a complete, verified proposal checkpoint:

```bash
scripts/submit.sh --stage generate --pilot data/new-catalog-cohort-001.json \
  --adopt-proposal-checkpoint /absolute/path/to/full-snapshot \
  --adoption-source-archive /absolute/path/to/original-source.tar.gz \
  --adoption-launch-receipt /absolute/path/to/original-launch.json \
  --out runs/new-catalog-recovery --concurrency 256 --tier interactive
```

This migration verifies the original source archive and every snapshot member,
reconstructs proposal decisions from the final reviews, and preserves all missing
and rejected slots. It rejects prior construction state and changed repair budgets.
It starts construction from the adopted accepted proposals without proposal inference.
The full checkpoint travels through an immutable S3 archive using streaming I/O.
After the adoption receipt is sealed, ordinary `--resume` uses that frozen proposal
tree; it does not require downloading the original checkpoint again. The current
adopter supports checkpoints with a completed first repair/review round.

The top-level report reconciles every manifest capability and all ten slots.
Exit zero means complete **accounting**: every slot has an accepted, rejected, or
null terminal disposition. It can therefore report zero accepted tasks. Quality
yield is separate, and training-ready quality is explicitly unassessed. Missing
proposals, ungraded execution failures, pending construction, and unresolved
adjudications remain visible and return exit two. A successful subset cannot
complete the manifest. Rejected proposals are not silently admitted individually.

This command removes stage handoffs and routine task-repair resumes, but does not
yet establish the final unattended recipe. Provider recovery, uncertain
adjudication, and incomplete judge calibration can still require intervention.
The broader repeated-build/reset, solver, grader, mutation, resource, split and
training-admission matrix remains required. Unit tests verify orchestration and
evidence accounting; live catalog-wide task quality is a separate experiment.
