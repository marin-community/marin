# Marina Page Agent Panel: Background Research

- Effort / stop rule / date: medium; stopped after the current Marina and Loom contracts, prior proposal, authentication alternatives, journey harness, and relevant protocol/UI guidance converged on one integration boundary; 2026-09-08.

## Question

How should Marina wire a page-aware agent panel to Loom, and how should the complete browser journey be tested without calling a real model? The answer must preserve Loom as the agent runtime, carry bounded page context, expose streaming and approval state, and produce deterministic screenshots that reviewers can inspect.

## Conclusion

The Marina browser should call Loom directly as the signed-in Loom user. Marina should not proxy Loom through a shared backend credential and should not implement another agent loop. The shared Marina shell owns page context and the panel; a small Loom client owns session launch, journal reconciliation, SSE, prompts, interruption, recovery, and permission answers. The first release must select a server-enforced read-only projection of Marina's generated MCP catalog.

Journey tests should fake the boundary Marina actually consumes, not the model provider. A reusable `ScriptedLoom` loopback server should implement the small Loom HTTP/SSE subset, record requests, and release canned events through test-controlled gates. This exercises the production client, transcript reducer, shared shell, current application, CORS behavior, and responsive panel in a real browser. A smaller cross-repository smoke should then run that browser client against real Loom and Loom's existing fake ACP process. The two layers cover the integration without model calls.

## Current Context

The prerequisite API path is merged. Marina now generates per-application MCP tools from explicitly annotated FastAPI operations, mounts them behind one authenticated Streamable HTTP server, and lets FastMCP expose only `find_tool` and `call_tool` while discovering the underlying inventory at runtime ([Marina MCP source](https://github.com/marin-community/marin/blob/6331d51cafae893e3f69049e4532361cb5de9624/infra/marina/src/marina/mcp.py#L29-L125), [Marin PR #8971](https://github.com/marin-community/marin/pull/8971)). Loom can pass a remote HTTP MCP descriptor directly to ACP and mint its IAP header from workload identity; it does not run a proxy process ([Loom PR #348](https://github.com/marin-community/loom/pull/348)). Tool inventory is owned by the MCP server's `tools/list`, not duplicated in Loom configuration ([Loom PR #349](https://github.com/marin-community/loom/pull/349)).

The production `marina` profile currently selects the full-risk `marina` MCP group and supplies page-assistant instructions, but it is still a normal interactive ACP session backed by a `marin-community/marin` worktree. The proposal migrates that profile to a distinct `marina-read` group and excludes the old group. The session archives after 50 idle minutes. An active ACP process cannot refresh the Marina IAP header in place and may need Loom recovery after token expiry ([profile declaration](https://github.com/marin-community/marin/blob/6331d51cafae893e3f69049e4532361cb5de9624/infra/loom/Pulumi.marin-loom.yaml#L62-L85), [profile instructions](https://github.com/marin-community/marin/blob/6331d51cafae893e3f69049e4532361cb5de9624/infra/loom/profiles/marina/AGENTS.md#L1-L15)). The worktree is unnecessary for page questions, but it is an accepted v0 cost rather than a reason to reproduce Loom's runtime in Marina.

Marina already has the right UI and test ownership. `Shell.vue` is rendered by every checked-in app and knows the active app and user, but has no agent or page-context extension point ([shared shell](https://github.com/marin-community/marin/blob/6331d51cafae893e3f69049e4532361cb5de9624/infra/marina/web/Shell.vue#L1-L63)). `Journey` starts the real kernel and browser, checks visible controls and text, captures screenshots at chosen widths, and fails on page errors or rejected same-origin API requests ([journey harness](https://github.com/marin-community/marin/blob/6331d51cafae893e3f69049e4532361cb5de9624/infra/marina/src/marina/journeys.py#L110-L216)). Its kernel config has no external-agent override yet ([journey fixture](https://github.com/marin-community/marin/blob/6331d51cafae893e3f69049e4532361cb5de9624/infra/marina/src/marina/journey_plugin.py#L59-L103)).

Loom already exposes the complete conversation control surface the panel needs:

- `sessions.launch` selects the `marina` profile and returns the durable session.
- `sessions.chat` returns journal blocks, the live turn, queued prompt, effective mode, and agent metadata.
- `sessions.chat.stream` emits `block`, `delta`, `tool`, `turn`, and `queue` SSE events, plus `resync` on a dropped bounded broadcast.
- `sessions.prompt.create`, `sessions.interrupt`, `sessions.recover`, and `sessions.permissions.answer` drive the conversation.

These are registered, typed operations rather than an app-specific route family ([session operations](https://github.com/marin-community/loom/blob/12733898021316468345f3a474d2f13280fe2724/crates/weaver-api/src/operations/sessions.rs#L57-L84), [prompt and permission operations](https://github.com/marin-community/loom/blob/12733898021316468345f3a474d2f13280fe2724/crates/weaver-api/src/operations/sessions.rs#L738-L836)). The journal deliberately keeps ACP block payloads extensible while stabilizing their addressing and composer metadata ([chat DTOs](https://github.com/marin-community/loom/blob/12733898021316468345f3a474d2f13280fe2724/crates/weaver-api/src/dto.rs#L2734-L2790)). Loom's own conversation component opens the event stream before loading its snapshot, reloads after a send, and handles explicit `resync`; that ordering closes snapshot/stream races and should be copied, not simplified ([conversation reconciliation](https://github.com/marin-community/loom/blob/12733898021316468345f3a474d2f13280fe2724/crates/loom/frontend/src/components/AcpConversation.vue#L329-L369)).

## Alternatives Considered

### Marina backend proxy with a Loom token

This appears operationally simple but is the wrong trust boundary. Loom API tokens are broad user/admin credentials. A Marina-held token would give the web service access to the fleet, attribute every launch and prompt to one principal, and require Marina to enforce which Loom sessions belong to which IAP user. More importantly, `sessions.permissions.answer` is intentionally human-only: a session credential cannot approve the request it caused. A shared service principal either cannot cross that boundary or crosses it with excessive authority. This option is rejected.

### Direct browser-to-Loom calls

This preserves Loom's own user, audit, session, and permission semantics. Both services use secure `*.oa.dev` origins, so Loom's `SameSite=Lax` session cookie remains same-site; the browser client must opt into credentials ([Loom session cookie](https://github.com/marin-community/loom/blob/12733898021316468345f3a474d2f13280fe2724/crates/loom/src/web/auth.rs#L283-L291)). Loom needs a narrow credentialed CORS allowlist for `https://marina.oa.dev` on the panel route subset. Users who have Marina access but no Loom session see a connect state and sign into Loom in a new tab. `auth.me` returns HTTP 200 with `authenticated: false` for that state. This is the recommendation.

### Embedding the Loom dashboard in an iframe

An iframe would preserve Loom auth and reuse its full component, but the current dashboard has no context-aware compact route. Its session chrome, tabs, worktree controls, and responsive assumptions are wrong for a 400–500 px application panel. Cross-frame context delivery and focus behavior would become a second protocol. This is rejected for v0.

### Reintroducing a provider loop in Marina

The predecessor proposal chose this while Loom lacked remote MCP and an appropriate profile. Those gaps are now closed. A direct provider loop would duplicate session persistence, streaming, tool execution, cancellation, permissions, and recovery. It is superseded.

## Journey Boundary and Fake Design

The fake should be named `ScriptedLoom`, not `FakeAgent`, because Marina never speaks ACP or a provider protocol. It implements the exact Loom routes the production browser calls. Each script contains expected requests and ordered server events. It records launch and prompt JSON for test assertions and exposes gates so the test can stop the stream at a meaningful UI state before continuing.

A static `route.fulfill` response is useful for ordinary JSON but is insufficient for the important streaming transitions. Playwright officially supports request interception, and Marina can use it for authentication or error variants, but a loopback ASGI server gives real chunked SSE, reconnects, and CORS preflight behavior ([Playwright API mocking](https://playwright.dev/docs/mock)). The fake is test infrastructure, not a production endpoint or frontend-only mock.

The main Plantt journey should:

1. Load a deterministic chart fixture and open `Ask Marina` from the shared shell.
2. Show the chart title and revision as the active context, then submit a scheduling question.
3. Assert that `ScriptedLoom` received profile `marina`, the configured repository, and a bounded context envelope containing the URL, chart id, revision, selection, and dirty/conflict state but no DOM or chart document.
4. Release a turn start, generic tool state using Loom's current payload, text deltas, durable blocks, and turn end. Photograph the in-flight tool state and completed answer.
5. Submit a second read request. Release a pending permission block with Loom's current title and option fields, photograph it, reject once, and verify the exact Loom permission option sent.
6. Disconnect and reconnect the non-replaying SSE stream, reload the durable snapshot on open, and prove that the final durable content is not duplicated.
7. Capture the completed panel at phone, laptop, and wide-desk widths, then close it and verify focus returns to its launcher.

Screenshots are review artifacts, not the only assertions. The current `shoot()` behavior should remain deterministic documentary evidence. Semantic role/text assertions, the fake's recorded requests, and reducer state are the correctness gates. Pixel baselines can be added later for a small component gallery; Playwright notes that visual comparisons vary with browser and host environment ([Playwright visual comparisons](https://playwright.dev/docs/test-snapshots)).

## Interaction and Safety Findings

The panel should be a docked complementary surface on wide screens, not a modal dialog: users need to inspect and manipulate the page while conversing. At narrow widths it becomes a full-screen modal dialog, where focus containment, Escape-to-close, an accessible name, and focus restoration are required by the WAI-ARIA dialog pattern ([WAI-ARIA dialog pattern](https://www.w3.org/WAI/ARIA/apg/patterns/dialog-modal/)).

Loom's existing conversation UI provides useful prior art: typeset dialogue, compact completed tool activity, a live status line, stop, queued feedback, reconnect/recovery, and permission cards. The Marina panel should implement the smaller subset rather than copy model/config/file/worktree controls.

Tool activity and approval need a contradiction check. MCP recommends visible tool invocations, tool-input review, and human confirmation for sensitive operations, while treating annotations as hints rather than authority ([MCP tools security](https://modelcontextprotocol.io/specification/draft/server/tools)). Marina's risk metadata is trusted at its server, but FastMCP's search transform exposes the generic `call_tool`. Current Loom events contain a title, kind, status, content, locations, and permission options; they omit the MCP tool name, raw input, nested operation, risk, and HTTP status. Loom can display a permission only if the ACP provider emits one. V0 therefore uses a separate read-only MCP projection enforced by Marina and renders generic activity from the fields Loom supplies.

Plantt is a strong first demonstration because its operations are typed and its page has useful context that is much smaller than the document. Its whole-document `update_chart` is revision guarded while the UI autosaves, so an external agent write would leave the page stale and could immediately conflict or overwrite. A future write path needs structured pre-execution tool metadata, host-enforced risk, and a post-success invalidation callback that refetches the active chart. Re-reading immediately before the tool call does not refresh the browser afterward ([Plantt workspace](https://github.com/marin-community/marin/blob/6331d51cafae893e3f69049e4532361cb5de9624/infra/marina/apps/plantt/web/src/views/Workspace.vue#L32-L151), [Plantt API registration](https://github.com/marin-community/marin/blob/6331d51cafae893e3f69049e4532361cb5de9624/infra/marina/apps/plantt/app.py#L277-L398)).

## Internal Prior Work

- The earlier `marina_api_registry` artifact correctly identified the shared shell, bounded page context, and operation call cards, but selected a direct Marina model loop because Loom lacked remote MCP and delegated identity. The merged MCP/profile work changes that decision.
- [Marin PR #8971](https://github.com/marin-community/marin/pull/8971) is the accepted deployment contract for the remote MCP and dedicated profile.
- [Loom PR #348](https://github.com/marin-community/loom/pull/348) chose direct ACP HTTP descriptors over a Loom `serve-remote` proxy and documented token recovery.
- [Loom PR #349](https://github.com/marin-community/loom/pull/349) made FastMCP's live `tools/list` authoritative.
- Echo searches on 2026-09-08 found the merged Marina MCP implementation and README but no separate decision for an embedded panel or its test seam. This artifact is the first durable treatment of that layer.

## Evidence Map

### Claim: the browser, not Marina's backend, should be the Loom client

- Support: Loom authenticates human browser sessions and makes live permission answers human-only; Marina and Loom are same-site secure origins.
- Contradiction: users need a second Loom login, and Loom must gain restricted credentialed CORS.
- Directness: high.
- Confidence: high.
- Action: add explicit agent-service configuration to Marina and an allowlisted browser origin to Loom; hold no Loom secret in Marina.

### Claim: journey tests should fake Loom's HTTP/SSE contract

- Support: Marina consumes Loom, not ACP; the journey harness already runs a real kernel/browser and captures screenshots; Loom already tests ACP with a scripted process.
- Contradiction: maintaining a contract fake duplicates a small set of payload shapes and cannot prove live-provider permission behavior.
- Directness: high.
- Confidence: high.
- Action: build one reusable `ScriptedLoom`, validate its fixtures against pinned Loom shapes, and run a cross-repository smoke against real Loom plus its fake ACP process.

### Claim: the pilot needs a read-only MCP projection

- Support: the current search transform exposes generic `call_tool`; Loom cannot see the nested Marina risk; prompt instructions do not enforce policy.
- Contradiction: this adds a second MCP URL and remote capability over the same generated registrations.
- Directness: high.
- Confidence: high.
- Action: filter both search and invocation at the Marina server, and let the panel profile select only that endpoint.

### Claim: a responsive dock is preferable to a universal modal

- Support: desktop users need simultaneous page and agent interaction; mobile needs focus-contained full-screen behavior; Plantt's dense canvas cannot tolerate a fixed overlay at every width.
- Contradiction: dock resizing adds layout state and responsive cases.
- Directness: high.
- Confidence: high.
- Action: push content on wide screens, use a modal side sheet at medium widths, and use a modal full-screen dialog on phones.

## Recommended Experiments

### 1. Cross-origin human authentication spike

- Minimum experiment: from local Marina, call production or a local Loom configured with an exact allowed origin using `credentials: include`; verify `/api/auth/me`, launch, chat snapshot, SSE, and permission answer.
- Falsifier: the browser cannot send the Loom cookie safely, a reverse proxy strips credentialed CORS, or Loom cannot distinguish the human.
- Cost/risk: local and staging session churn only.

### 2. Plantt scripted journey

- Minimum experiment: implement the consumed Loom route subset and the journey above, including screenshots at the permission gate and responsive widths.
- Falsifier: page context contains the chart document or DOM, stream state duplicates on reconnect, or the panel hides primary content at a supported width.
- Cost/risk: deterministic local test only.

### 3. Real Loom with fake ACP

- Minimum experiment: start Loom with its scripted ACP process, loopback trust off, an authenticated test user, and explicit local CORS; drive it through Marina's real browser client.
- Falsifier: launch attribution, SSE recovery, or a permission answer differs from `ScriptedLoom`.
- Cost/risk: local processes and a disposable Loom database; no model or production call.

### 4. Mutation contracts and canary

- Minimum experiment: after Loom exposes structured tool identity/input, a host enforces nested risk, and Plantt implements page invalidation, request a harmless title change in a disposable chart.
- Falsifier: the write executes before the human decision, the card cannot identify the operation and inputs, or the open chart does not reload the new revision.
- Cost/risk: one disposable staging chart; mutation remains outside the pilot until all three contracts and this canary pass.

## Source Ledger

| Source | Type | Claim used for | Confidence |
| --- | --- | --- | --- |
| [Marin PR #8971](https://github.com/marin-community/marin/pull/8971) | merged PR | Production MCP/profile contract and caveats | High |
| [Loom PR #348](https://github.com/marin-community/loom/pull/348) | merged PR | Direct remote MCP, IAP startup token, recovery | High |
| [Loom PR #349](https://github.com/marin-community/loom/pull/349) | merged PR | Live MCP inventory is authoritative | High |
| [Marina shell and journeys at `6331d51`](https://github.com/marin-community/marin/tree/6331d51cafae893e3f69049e4532361cb5de9624/infra/marina) | code | Current UI host and journey capabilities | High |
| [Loom session API at `1273389`](https://github.com/marin-community/loom/tree/12733898021316468345f3a474d2f13280fe2724/crates/weaver-api/src) | code | Session, chat, stream, prompt, and permission contracts | High |
| [Loom fake ACP agent](https://github.com/marin-community/loom/blob/12733898021316468345f3a474d2f13280fe2724/crates/loom/tests/fixtures/fake-acp-agent.mjs#L1-L44) | test code | Lower-layer deterministic scripting precedent | High |
| [MCP tools specification](https://modelcontextprotocol.io/specification/draft/server/tools) | official specification | Visibility, approval, and annotation limits | High |
| [WAI-ARIA dialog pattern](https://www.w3.org/WAI/ARIA/apg/patterns/dialog-modal/) | official guidance | Modal focus and keyboard behavior | High |
| [Playwright API mocking](https://playwright.dev/docs/mock) | official documentation | Browser-boundary request mocking | High |
| [Playwright visual comparisons](https://playwright.dev/docs/test-snapshots) | official documentation | Screenshot determinism caveat | High |

## Handoff

The concrete proposal and contracts are in [design.md](design.md) and [spec.md](spec.md). No implementation issue was supplied. The pilot gates are a server-enforced read-only MCP projection, credentialed cross-origin Loom auth, the scripted Plantt journey, and the real-Loom/fake-ACP smoke. Mutation has separate structured-event, policy, invalidation, and canary prerequisites.
