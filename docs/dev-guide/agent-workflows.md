# Agent workflows

Choose a skill by the requested outcome. Each skill owns one workflow; shared
policy and conditional procedures live in linked guides and references. A task
can move between skills as it moves from implementation to publication or from
experiment planning to execution.

For an ordinary feature or bug fix without a domain skill, implement under
root and scoped repository instructions, then use `manage-pr` when the change
is ready to publish. No general implementation skill is required.

## Changes and knowledge

| Skill | Owns | Boundary |
|---|---|---|
| `manage-pr` | Author-side validation, commit checkpoints, PR publication, feedback, and monitoring | Domain skills own implementation; an assigned issue uses its issue-fixes reference |
| `review-pr` | Independent correctness and design assessment | Read-only except requested comments; correctness and holistic automation keep separate reporting contracts |
| `simplify-code` | Explicitly requested cleanup of a branch or diff | Review findings alone do not authorize edits; publication belongs to `manage-pr` |
| `file-issue` | Deduplicating and filing an issue | Implementing an assigned issue belongs to `manage-pr` |
| `use-echo` | Searching prior work and maintaining shared knowledge or incident records | Repository procedures belong in `OPS.md` or `docs/`; search alone does not require a write |
| `curate-experiments` | Report-index curation and explicitly requested archive tags and issue breadcrumbs | Archival does not authorize deleting experiment files; scheduled issue summaries use an assigned procedure |

## Operations and measurement

| Skill | Owns | Boundary |
|---|---|---|
| `use-iris` | Job, task, cluster, and accelerator operations; Zephyr monitoring and infrastructure troubleshooting | Experiment recipes, production lineage, and telemetry interpretation have separate owners |
| `query-finelog` | Log and telemetry queries, schemas, counter semantics, and forwarding diagnosis | Iris remains the source of job liveness and execution operations |
| `use-fsutil` | Object-storage operations through Marin bucket routing | Accessing storage does not require launching compute |
| `manage-hero-run` | Production run identity, retention, recovery, monitoring, cutovers, and closure | Domain skills own recipes; code-cutover trial gates override ordinary recovery |
| `run-ferries` | Canary and daily ferry recipes, interpretation, and closure | Uses Iris for execution; CI failure diagnosis follows its diagnosis-only procedure |
| `ab-test-zephyr` | Controlled Zephyr comparisons with equivalent workloads | Measures stage efficiency and latency; monitoring belongs to Iris |
| `profile-training` | Training-profile capture and interpretation | Measures bottlenecks; implementation changes use the relevant domain workflow |

## Access and applications

| Skill | Owns | Boundary |
|---|---|---|
| `add-grant` | Preparing a specified access change | Never applies it; a second person reviews |
| `review-grant` | Verifying decrypted identities and roles, obtaining confirmation, and applying the approved change | Never approves or merges before confirmation |
| `trace-pulumi-diff` | Read-only attribution of pending infrastructure changes | Never applies or refreshes state |
| `onboard-marin` | Local developer readiness and identification of missing access | Does not grant access or launch compute without the required authorization |
| `marina-applet` | Marina application build, validation, release, data, and lifecycle | Application releases are separate from Marina runtime deployment |

## Implementation and publication

| Skill | Owns | Boundary |
|---|---|---|
| `refresh-fork` | External fork refresh and validated adoption | Fork-specific references govern source promotion and release |
| `add-dataset` | Dataset registration with verified schema and content mapping | Uses existing dataset builders; general graph design belongs to `write-pipeline` |
| `add-pallas-kernel` | Kernel implementation, reference parity, and requested tuning | Uses profiling evidence; does not own production deployment |
| `change-grug` | Grug variants and promotion into the base template | Launch-only operation and hero cutovers have separate workflows |
| `launch-rl` | RL artifact recipe, role geometry, validation, and submission | Iris owns execution mechanics; hero management owns production lifecycle |
| `write-pipeline` | General artifact graph construction and work/workflow placement | Dataset registration and RL recipes use their domain skills |
| `upload-hf-model` | Model-weight publication with required model-card and license content | Dataset uploads and Hub downloads are outside its scope |

## Shared policy and automation

Read root `AGENTS.md` and path-scoped instructions for every task. Testing
policy lives in root `TESTING.md` and the module testing guides. Use the
[writing guide](writing-style/index.md) for the deliverable's medium and
[experiment records](experiment-records.md) for research evidence and reporting.

CI and scheduler assignments name procedures under `infra/loom/procedures/`.
Those procedures preserve their own context requirements, write permissions,
idempotency markers, and stopping conditions. They are not general skill
entry points. Changing the catalog does not grant permission to run them.
