# Experiment: adapt fixed behavioral tests to a submitted interface

Source: `other_thoughts.md`, September 19, 00:11 PDT. This is an experimental
verifier family, not an accepted replacement for current task gates.

The hypothesis is that a task can specify behavior without prescribing names,
signatures, classes, or result wrappers. A small model can adapt existing tests
to the implemented interface while preserving their behavioral meaning.

## Bounded first experiment

Use one deterministic task: compute total covered time for finite half-open
integer intervals. Input order is arbitrary; overlaps and duplicates count once;
empty and zero-length intervals contribute zero; negative endpoints are valid;
reversed intervals must be rejected. These are behavioral requirements. The
task does not prescribe a Python API.

Prepare four independently written correct interface variants: a function over
tuple pairs, a class over records, a variadic function over interval objects with
a result wrapper, and a JSON command-line interface. Freeze the source, a common
test-case table, expected outcomes, and per-interface behavioral mutants before
asking a model to adapt tests. Execute candidate and adapted code only in fresh
network-blocked sandboxes. Local metadata/controller tests are permissible.

Compare two adaptation strategies using the same model and source views:

1. Rewrite the fixed tests to call the submitted interface, explicitly preserving
   every case, assertion, and exception check.
2. Generate a thin interface adapter for unchanged canonical tests. The adapter
   may translate inputs, invoke the implementation, and normalize result shape;
   it must not implement the target behavior itself.

Use three independent adaptation attempts per interface and strategy. This is a
24-attempt transport and semantics pilot, not evidence of broad effectiveness.
Record the actual model, request parameters, completion counts, transcript hashes,
and elapsed time. GLM-5.3 is the available initial model; model-size comparisons
can follow if the basic method works.

## Measurement and rejection conditions

- Every correct interface must pass the same complete behavioral case table.
- Run each adapted suite against deliberately wrong implementations, including
  double-counted overlaps, skipped containment, wrong endpoint arithmetic,
  mishandled empty input, negative-endpoint clipping, and accepted reversed ranges.
  Precompute a witness for every mutant so an equivalent mutation is not counted
  as an undetected defect.
- Preserve case IDs and actual executed counts. A skipped test, removed assertion,
  catch-all exception handler, unconditional expected answer, or empty suite is a
  failed adaptation, never candidate correctness.
- The thin adapter must still detect behavioral mutations of the implementation.
  If it computes the answer independently, mutation sensitivity will expose that
  failure even when all correct candidates pass.
- Re-run the frozen adapted suite to distinguish adapter-generation variation
  from test-execution nondeterminism. Do not regenerate tests in response to a
  candidate's failing behavioral result.
- Record adaptation/import/runtime infrastructure errors separately from graded
  behavioral failures. No timeout, model refusal, or empty completion earns zero
  as if it were a measured semantic rejection.

Keep the adaptation transcript, original and rewritten assertions, adapter source,
test inventory, per-case outcomes, and mutation matrix private. Candidate source
is untrusted input to the adapter model; instructions embedded in it cannot change
the behavioral contract or reveal test artifacts to the solver.

Report correct-interface acceptance, mutation detection, assertion/case preservation,
adaptation failure rate, repeatability, latency, and completion usage per strategy.
Advance to generated-task integration only after inspecting actual disagreements
and retaining the same oracle, solver, adversarial, resource, and semantic gates
as other verifier families. The earlier dynamic multi-step/simulated-user idea in
`other_thoughts.md` remains sequenced after the basic task POC.

## Pilot result (September 19, 2026)

The first measured campaign is `runs/adaptive-test-pilot-005` (source snapshot
`d4a079854e6e32b01c1c130457d6c415b66ae9dd359db7fbe9336048fd3823eb`,
durable result snapshot `917c191d3e46492c82f0921e74d50c49`). The four submissions are interface
variants over a shared union algorithm, not independently derived algorithms.
A trusted candidate-entry trace, rather than a generated test's self-report,
records which canonical inputs were exercised. A mutant counts as detected only
when its frozen witness was reached and execution ended in an assertion failure;
syntax, import, runtime, timeout, and provisioning errors remain separate.

GLM-5.3 interactive received a 128,000-token output budget, temperature 0.7,
and high reasoning effort. All 24 responses ended with `finish_reason=stop`;
none hit the budget. Rewrite responses used 18,086 completion tokens total
(631--3,641, mean latency 8.84 s), while adapter responses used 12,821
(328--2,837, mean latency 9.19 s). Four stop-completions were invalid JSON
(three rewrite, one adapter), so they are adaptation failures rather than
semantic zeroes. Eighteen attempts reached Daytona; all observed network
blocking and all sandboxes were deleted. Two further rewrite attempts failed
during Daytona provisioning and remain null infrastructure results.

For the three in-process Python interfaces, the thin-adapter arm passed all
eight measured attempts: each accepted the correct implementation twice with
identical output and a complete nine-input trace, then caught all six mutants.
Its ninth attempt was a model-format failure. The rewrite arm passed four of
five measured attempts; the other accepted and fully exercised the correct
implementation but caught zero mutants. Three additional rewrites were model
format failures and one was a provisioning failure. This supports the thin
adapter design for this small task, but the sample is too small for a general
effectiveness claim.

The JSON-interface rows in campaign 005 are exploratory only. The initial
trusted trace recorded the negative-clipping mutant after its input
transformation, making an observed assertion failure fail the witness check.
Campaign `007-replay` reused the exact generated bytes after moving that trace
to raw command entry. Its handwritten JSON baseline accepted the correct CLI
and caught all six mutants, but provider aggregate-CPU validation errors left
the generated JSON replays unmeasured. Those values remain null; campaign
005's JSON 5/6 figures are not promoted into the comparison.

Campaigns 001--004 are retained infrastructure diagnostics, and campaign 006
was cancelled before becoming a redundant inference campaign. Controller tests
reject fixed weakening controls for a removed assertion, missing case, explicit
skip, and bare catch-all. This pilot does not alter admission or readmit any
current task cohort.

Future fresh campaigns start at 131,072 output tokens and retry only a length
termination with `max_tokens: null`, letting the serving engine account for its
remaining context. Both responses and the effective request parameters are
retained. Missing usage is not converted into a guessed prompt size. This is a
new controller policy, not a relabeling of campaign 005's 128,000-token requests.
Frozen replay retains the original request and validates generated code against
the recorded response before execution. Missing trusted trace evidence now leaves
the attempt unmeasured rather than counting it as failed mutation detection.
