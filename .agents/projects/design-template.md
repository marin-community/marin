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

---

**Length target**: ≤500 words, preferably less. If it's longer, you're either solving multiple problems (split it) or writing implementation details (defer them). The 1-pager is a snapshot, not a living doc — close the issue when the PR lands, don't sync it as the design evolves.
