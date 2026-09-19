# NeMo Gym stateful corpus re-import contract

Reviewed 2026-09-14 against the Hugging Face Dataset Viewer API and the pinned
NeMo Gym implementation at `1e668906d2e69a9e8ee9aaafc60050a4025d9688`.
Sampling used the repository helper `lib/taskcompendium/examples/nemo_gym_sampling.py`
logic (three deterministic train offsets per corpus). No production code or
source archive was changed.

## Verdicts

| corpus | live revision / train rows | sampled rows | initial state | tool surface | verifier fidelity | verdict |
| --- | --- | --- | --- | --- | --- | --- |
| `nvidia/Nemotron-RL-agent-workplace_assistant` | `c86a908379e0a361a573c395e175d3c1aa128e6c` / 1,255 (1,800 total) | 117, 536, 954 | Shared seeded Workplace tables; the row has no state blob because the NeMo environment seed is external to each row | `responses_create_params.tools`: 27 functions over company directory, email, calendar, analytics, project management, and CRM | `ground_truth`: ordered `{name, arguments}` calls; source grades the final mutable tables after replaying those calls | **Safe after importer generalization.** The current provider already supplies the shared runtime and seed, but the importer is deliberately restricted to row `id=0` and its exact fixture digest. |
| `nvidia/Nemotron-RL-agent-calendar_scheduling` | `f4a9cb60a19a56e1cc96628e3456147bf8598ea5` / 3,872 (4,000 total) | 2, 1,294, 2,584 | Prompt says calendar starts empty; `exp_cal_state` is the private expected final state, not an initial provider snapshot | `responses_create_params.tools` is an empty list | `grade_assistant_response` checks IDs/count, duration, timing constraints, bounds, and conflicts, but never checks `event_name` | **Reject as-is; conditionally importable after a name contract is added.** It must remain answer-only; mapping it to the Workplace tool provider would invent a source tool surface. |
| `nvidia/Nemotron-RL-Instruction-Following-Calendar-v2` | `556a3b1ab3eb12bab38327bf0ec4cdbeab452338` / 9,659 (9,915 total) | 903, 4,122, 7,343 | Same empty-calendar premise; `exp_cal_state` contains 1–7 expected event records and constraints | `responses_create_params.tools` is an empty list | Same verifier omission: final JSON event names are ignored even though the public format requires them | **Reject as-is; conditionally importable after a name contract is added.** Treat it as a separate source revision and preserve all conversation turns; do not turn it into tool calls. |

The sampled Workplace rows include, for example, calendar deletion (offset
117), CRM insertion (536), and project status updates (954). Their row fields
are `id`, `responses_create_params`, `ground_truth`, `category`,
`environment_name`, and `agent_ref`; no row-specific database snapshot is
present. The existing vendored CSV seed and tool modules are therefore a
required pinned provider resource. The current provider's `SEED_SHA256` and
interface digest checks are the right preservation mechanism.

The sampled calendar rows contain only `responses_create_params`,
`exp_cal_state`, and `agent_ref`. The response input is a sequence of system,
user, and assistant messages; assistant messages include pre-materialized JSON
calendar answers and user filler turns. `exp_cal_state` records event IDs,
durations, natural-language constraints, and `min_time`/`max_time`, including
multi-event states. It does not describe a mutable service state or an action
trace. The source calendar implementation explicitly declares no tool surface
and verifies only the last assistant text. Its `is_constraint_satisfied`
compares only `duration`, bounds, and the natural-language timing constraint;
`grade_assistant_response` additionally checks event IDs/count and pairwise
overlap. It never compares `event_name` (and does not require the field to be
present), so the pinned source verifier accepts a schedule with incorrect or
missing names.

Names are recoverable in the inspected rows from earlier materialized
assistant JSON lists in `responses_create_params.input`; for example, all
non-null IDs in sampled rows 2,584 and 7,343 have names in those lists. That is
useful evidence for a repair, but it is not an authoritative expected field:
`exp_cal_state` has no names, and names are absent for some intermediate states
in other sampled rows. They cannot be silently inferred and claimed as source
verifier semantics. Under the project invariant that every nontrivial task
detail is checked, both calendar corpora are rejected until the importer
establishes a deterministic source-backed name mapping (or the source data is
amended) and the checker enforces exact normalized names.

## Implementation contract

### Workplace

Generalize `importers/nemo_workplace.py` from the special id-0 fixture to a
Hub-row importer keyed by `(dataset revision, split, offset)` (or a canonical
row digest), while retaining the raw row as a verifier-only resource. Validate
the six source fields, all 27 advertised schemas, and nonempty JSON-object
`ground_truth` arguments. Keep `TaskRequirements(action_interfaces=(INTERFACE,))`,
`ProviderStateVerifier`, `ChatWithTools`, and the existing provider adapter.

The provider contract is one small shared runtime image/package plus the
vendored CSV seed and tool modules. Per task, pin only the source row and its
private expected calls; pass the fixed interface/seed digest in private
provider setup. Do not build or publish an image per row. The existing provider
already creates a fresh isolated seed per trial and compares authoritative
calendar/email/analytics/project/CRM tables. A regression suite should sample
at least one mutation from each toolkit and prove wrong/no-op calls score zero.

### Both calendar corpora (blocked pending name repair)

Add one reusable calendar answer importer that accepts the full `input` message
list, preserves the source system/user/assistant turns, stores the raw row and
`exp_cal_state` as verifier-only resources, and records `(dataset, revision,
split, offset)` provenance. Use `TaskRequirements()` and an answer rendering;
there is no action interface, provider environment, or public tool binding.

Add one pinned calendar checker implementation (prefer a shared verifier image
or the repository's executable verifier runtime) that reproduces the source
`grade_assistant_response` behavior: extract the final assistant text, parse
the JSON list, and apply the source event IDs/names/durations, time bounds,
constraint, uniqueness, and overlap checks. The current source utility does not
check names, so this is a required semantic repair, rather than a faithful
import of the existing verifier. Names must be derived deterministically from
source user requests plus materialized assistant JSON history, validated for
every expected non-null ID, and stored privately as an importer-produced
expected-state extension. Reject rows where that mapping is missing or
ambiguous. Keep the checker and expected state private. The public
instructions may retain the observable JSON response format and scheduling
rules, but must not mention graders, rewards, hidden checks, or reference state.

The calendar verifier is not a `ConstraintVerifier`: its constraints are a
whole schedule plus source-specific natural-language constraints, and the
existing answer-only importer accepts only one user message and scalar
instruction constraints. A new typed verifier or an executable checker is the
minimal architecture addition. The same checker image can serve both datasets;
only the pinned row resource, source revision, and private expected-state data
vary per task. Do not label either corpus safe for deployment until the name
repair passes a corpus-wide completeness audit.

## Source references

- [Workplace dataset](https://huggingface.co/datasets/nvidia/Nemotron-RL-agent-workplace_assistant)
- [Calendar scheduling dataset](https://huggingface.co/datasets/nvidia/Nemotron-RL-agent-calendar_scheduling)
- [Calendar v2 dataset](https://huggingface.co/datasets/nvidia/Nemotron-RL-Instruction-Following-Calendar-v2)
- [NeMo calendar config](https://github.com/NVIDIA-NeMo/Gym/blob/1e668906d2e69a9e8ee9aaafc60050a4025d9688/resources_servers/calendar/configs/calendar_v2.yaml)
- [NeMo calendar verifier](https://github.com/NVIDIA-NeMo/Gym/blob/1e668906d2e69a9e8ee9aaafc60050a4025d9688/resources_servers/calendar/app.py)
