# Design Doc Template

This file is the canonical 1-pager template for Marin design docs. It contains:

1. **The template** — fill-in-the-blank scaffolding to copy when starting a new doc.
2. **A worked example** — the finelog proposal from [#5210](https://github.com/marin-community/marin/issues/5210), kept as the reference for tone, length, and specificity.

Keep this file up to date as conventions evolve. The `design-doc` skill points here; existing docs in `.agents/projects/` predate the skill and are inconsistent — don't use them as style references.

**Length target**: ≤500 words, preferably less. If it's longer, you're either solving multiple problems (split it) or writing implementation details (defer them). The 1-pager is a snapshot, not a living doc — close the issue when the PR lands, don't sync it as the design evolves.

---

## Template

Copy from here when drafting a new doc. Delete the italic prompts as you fill in each section.

```markdown
# <Project Name>

_Why are we doing this? What's the benefit?_

<2-4 sentences. State the user-visible or system-level outcome. If this is enabling future work, say which work and why it's blocked today.>

## Challenges

_What's hard?_

<Where's the actual difficulty? Unknown territory, awkward tradeoffs, missing infra. Skip if genuinely easy — but then ask whether this needs a design doc at all.>

## Costs / Risks

_What's bad about doing this?_

- <Concrete downside: churn, new dependency, migration burden, etc.>
- <Another. Be honest — "no immediate user-visible improvement" is a valid bullet.>

## Design

_How are we doing this?_

<The actual proposal. Reference real files with line numbers (`lib/iris/src/iris/foo.py:142`). Show the core idea in a 10-30 line snippet if it helps; skip the snippet if prose is clearer. State backwards-compatibility upfront.>

## Testing

_Agents make mistakes — how do we catch them?_

<What's the test strategy? Integration vs unit. What scenarios matter. If this touches live infra, what's the rollout check (e.g. "test against Iris dev cluster with old workers + new controller").>

## Open Questions

- <Things you want feedback on. This is the section reviewers will gravitate to — make it specific.>
- <Avoid "should we do this?" as an open question; the doc itself is the proposal.>
```

---

## Worked Example

A target for tone, length, and specificity. From the original RFC ([#5210](https://github.com/marin-community/marin/issues/5210)).

```markdown
# Project Idea

_Why are we doing this? What's the benefit?_

Lift the logging service out of the Iris controller and add a service independent client per our system design guidelines. This will reduce load on the Iris controller VM and allow us to switch to an external logging vendor in the future if desired. It will also simplify the Iris codebase slightly.

This will allow us to iterate on the logging service independently of the Iris controller, which will reduce the risk of breaking the Iris controller while testing logging changes or introducing a stats engine. This also serves as a test-subject for how to lift resolution & proxy management out of Iris and into a shared set of utilities in our "rigging" repo.

Longer term, this makes it easier for us to work on a stats service as well.

## Challenges

_What's hard?_

Most of the challenges are around structuring services, for which we don't have a great template yet. Thus, this project needs to bear the brunt of figuring out the complexity. For example, we currently start a proxy to the Iris controller automatically for our CLI. If the logging service moves to a separate VM, do we also start a proxy to it? How? When? Is it done during resolution, or manually via a CLI? If it's manual, how do we find the logging service to connect to it, and when do we do it?

Cross-project proto dependencies are a pain, as is the tradeoff of duplicating protos.

## Costs / Risks

_What's bad about doing this?_

* It introduces a new dependency for Iris startup that was previously automated (the log server subprocess)
* It introduces churn with no immediate user visible improvement.

## Design

_How are we doing this?_

We'll introduce a new lib/finelog package with a bulk rename of the Iris logging code. As part of this lift, we'll introduce a top-level "LogClient" which wraps some of the proto details and provides convenience functions. Iris workers will be changed to use this new logging interface. (Workers already directly operate against the log server and it is resolved via an Iris address.)

The Iris controller will continue to proxy the "FetchLogs" requests to allow Iris CLI operations to work without complex proxy changes. Our long-term plans with proxying are unclear, and this avoids taking on this complexity in the short-term.

## Testing
_Agents make mistakes, how can we catch them?_

We'll test this against the Iris dev controller & cluster. We want to check for:

* Rolling out with old workers & a new controller
* Full cluster restart with new workers

As workers are already informed of the logging service host:port via the controller, we expect both conditions to be easy to fulfill.

## Open Questions

* The best way to deal with proxies is unclear to me. Using a bundled service like TailScale or CloudFlare is tempting, but would make new deployments more complicated.
* Similarly, this raises questions around how we should handle auth across services, if at all - or just assume we'll handle auth at the proxy level.
```
