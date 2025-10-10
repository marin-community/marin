You have been assigned an important task. You are to document and fix the Github issue indicated by the user.

## Backgound

For background on the codebase make sure to read:

@AGENTS.md
@docs/recipes/architecture.md

This will involve multiple steps, and you will update the user or github comment
as the tasks are completed. You may not proceed to a new task until you have
successfully completed all prior tasks. If you cannot complete a task, you will
write a Github comment indicating your last status and why you failed. A task
list is prepared for you at the end of the document.

## Research

Use the `gh` cli tool to fetch the current state of the issue.  Read the
codebase to understand the origin of the issue, including _all_ relevant source
files.  Use the output from your research to add a comment to the issue with the
appropriate level of detail.

* You will write a comment titled "# Agent Research Report"
* Your comment will start with 1 paragraph (max) recapping the issue.
* Your comment will follow this with links to the relevant source code related the issue.
* You may optionally include code snippets, but keep it minimal.

Example research report:

> TPU-to-CPU weight transfers during training achieve only ~1GB/s, well below
> hardware capabilities (4-7GB/s expected for TPU v4/v5). This blocks efficient
> weight synchronization between training and rollout workers, causing large
> models (8B+ parameters) to take 16+ seconds to transfer when they should take
> 4-8 seconds. The root cause appears to be inefficient memory layout from
> jax.device_get() and lack of parallelization, requiring investigation into
> alternative transfer strategies, memory optimization, and hardware-specific
> tuning.
>
> Relevant Code
> - [arrow_flight.py#L384](https://github.com/marin-community/marin/blob/main/src/marin/rl/weight_transfer/arrow_flight.py#L384) - TPU-to-CPU copy in Arrow Flight implementation
> - [jax.py#L394-L402](https://github.com/marin-community/marin/blob/main/src/marin/rl/weight_transfer/jax.py#L394-L402) - TPU-to-CPU transfer in JAX implementation
> - [arrow_flight.py#L380-L384](https://github.com/marin-community/marin/blob/main/src/marin/rl/weight_transfer/arrow_flight.py#L380-L384) - Memory layout context
> - [jax.py#L228-L283](https://github.com/marin-community/marin/blob/main/src/marin/rl/weight_transfer/jax.py#L228-L283) - JAX transfer server (bypasses CPU)
> - [base.py#L32-L35](https://github.com/marin-community/marin/blob/main/src/marin/rl/weight_transfer/base.py#L32-L35) - Transfer mode definitions
> - [test_weight_transfer.py#L211-L255](https://github.com/marin-community/marin/blob/main/tests/rl/test_weight_transfer.py#L211-L255) - Performance benchmark test
>

Use the information you have collected to attempt to fix the behavior (if it's a bug).
When needed, you write an appropriate reproduction test which minimally validates your fix.

## Hypothesizing

You will next write out a hypothesis for how to proceed. There are 2 cases.

* For a small bug fix, show the static code trace which triggers the error. For instance:

> I've identified the source of the bug. In
>   def foo(a):
>      x.abc()
> we fail to handle the case where X is a tuple, and assume it's an object.
>
> This occurs in the RolloutWorker when the environment returns a tuple:
>   env_result = env.run()
>   foo(env_result)

* For a design request, prepare a design specification and attach it to the issue.

Read @docs/recipes/design_doc.md for how to write a design doc. Read this only
if a design doc is warranted.

 - Compact summary of the problem and goals
 - Overview of how the current code works, including source files and snippets
 - Proposed plan to fix the issue
 - Review of the plan and any challenges
 - Testing plan

In both cases, you keep things clean, and you don't over-complicate. You search
for the minimally disruptive fix that leaves the code base in a better state
than when you started.

## Implementation

Once you have researched and prepared your design or planned fix, you may
proceed to implementation. You will implement your changes in a branch with the
following format `agent/{YYYYMMDD}-fix-{issue-id}`.

You will implement a test which demonstrates your changed behavior before
beginning work. Your tests will be minimal and refrain from using mocks.

## Testing

* You write your test _before_ you make your fix, and validate your fix works.
* You use an existing test file in @tests/ if appropriate.
* You never use mocks when testing.
* You keep tests simple and minimal. You do not test obvious behavior like "object has an attr".
* You will test using `uv run pytest -m 'not slow'` before uploading
* You will ensure _all_ tests pass.

## Uploading

When all tests pass, you may proceed to upload your branch.
You will then open a pull request for the branch.
You will attach it to the Github issue with a comment summarizing the fix.

## Verify CI Status

After opening the pull request, you must verify that CI checks pass:

* Monitor the PR using `gh pr view <number> --json statusCheckRollup`
* Wait for the unit tests to complete
* If tests fail, investigate and fix any issues
* Push additional commits to address CI failures
* Do not consider the PR complete until all relevant checks pass

Key checks to monitor:
- **unit tests**: Must pass - validates your changes don't break existing functionality
- **lint-and-format**: Must pass - ensures code style compliance
- **build-docs**: Should pass if you modified documentation

# Tasks

The tasks for this recipe:

- [ ] Fetch issue information
- [ ] Aggressively research codebase for relevant source files for the issue
- [ ] Update issue with "Agent Research"
- [ ] Formulate a design or fix
- [ ] Update issue with proposed plan
- [ ] Create branch for the changes
- [ ] Write a test case as needed for changes
- [ ] Implement changes until all tests pass
- [ ] Upload branch to github
- [ ] Open pull request
- [ ] Verify CI checks pass and address any failures (by polling gh pr view)
- [ ] Update comment with final status

If at any point you are unable to proceed, you must add a comment to the Github
issue with your last status.

# RULES

0. Never credit yourself in commits or comments.
