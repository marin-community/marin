# Marina Page Agent Panel

Marina should give opted-in checked-in applications an `Ask Marina` panel that understands the current page and uses generated MCP operations through Loom. The first Plantt release is read-only: a user can ask about the open chart, watch the agent search and read its API, and answer any provider permission request without leaving the page. Marina will own the application context and compact UI; Loom will remain the agent runtime and durable conversation store. [Research](research.md) records the evidence and rejected alternatives.

## Challenges

Marina and Loom authenticate users independently, and Loom reserves live ACP permission answers for humans. A Marina proxy would need a broad shared credential and erase caller identity. The browser must instead preserve Loom's journal and non-replaying SSE semantics without copying its full dashboard.

Page context must be useful but bounded, and route changes cannot leak state between conversations. The test also needs deterministic control over streaming, tool progress, permissions, reconnects, and errors. Current Loom events omit the nested MCP operation, raw inputs, risk, and HTTP status, so v0 can render only the fields Loom actually supplies.

## Costs / Risks

- Users need a Loom browser session as well as Marina access, and Loom must allow credentialed requests from exact Marina origins.
- The current `marina` profile creates a repository worktree and appears in the normal Loom fleet. That is resource and dashboard noise for page questions, but smaller than designing a second runtime.
- Loom's fleet is co-owned by its authenticated users, so Marina page context and transcripts are not private to their creator. The panel should not imply otherwise.
- Marina will maintain a small Loom client and pinned contract fixtures.
- The pilot needs a separate server-enforced read-only MCP projection. Structured mutation cards and page refresh after a write remain follow-on work.

## Design

### Keep execution and identity in Loom

The browser calls `https://loom.oa.dev` with Loom's human session cookie. Marina stores no Loom token, conversation database, or model SDK. Loom limits credentialed CORS to the panel routes and exact Marina origins. Marina publishes the Loom origin, profile, and repository as non-secret UI configuration; `auth.me` distinguishes signed-out users without a 401.

The `marina` profile selects only a new `marina-read` capability backed by `/api/marina/mcp/read/`. It drops the current artifact, channel, GitHub, issue, messaging, permission, session, and full-risk Marina MCP groups. The read endpoint derives its inventory from the same FastAPI registrations but rejects every non-`read` Marina operation during search and invocation. Before launch, the browser resolves the profile, verifies that its MCP selection is exactly `marina-read`, and submits Loom's profile revision guards with the launch. The existing repository grant remains only because current Loom sessions require a worktree. This boundary prevents writes through the selected Marina MCP endpoint; a future context-only Loom session should remove the unrelated worktree, provider-local tools, and repository credential.

Opening the panel creates nothing. The first question resolves and launches an interactive ACP session using the `marina` profile; its goal contains the request and page-context envelope. Later turns use `sessions.prompt.create` with fresh context. The client opens `sessions.chat.stream` before loading `sessions.chat`, reconciles by stable identities, snapshots after reconnects and `resync`, and calls Loom directly for interrupt, recovery, and permission answers.

One browser tab keeps up to 20 session ids per Loom username and page `contextKey` in `sessionStorage`. Plantt uses `chart:<uuid>`; an EvalDash run uses `run:<id>`. Reloading or returning restores the recent transcript and offers older journal pages on demand. If the route changes during a live turn or permission request, the panel asks whether to stay or switch after the turn; it never hides active work automatically. `New conversation` discards only the local mapping. The old Loom session remains auditable and visible under Loom's normal fleet policy.

### Make page context an explicit shell contract

The shared shell supplies the application id, canonical origin and path, route, and document title. It excludes URL fragments and query parameters; an app exposes useful filters through typed state instead. A shared composable lets the active view publish a versioned context with a stable key, short label, and bounded JSON state. The shell snapshots it only when a user submits.

For Plantt, the state contains `chart_id`, `revision`, `selected_item`, `dirty`, `saving`, and `conflict`. It does not contain the chart document or rendered text; the agent obtains those through `plantt_read_chart`. The serialized envelope is limited to 16 KiB, labeled as untrusted data, and placed separately from the user's request. Context is a navigation aid, not authority: it cannot choose a Loom profile, MCP group, or operation.

### Use a responsive dock

`Ask Marina` sits in the shared top bar beside the user identity, so it never covers an application's controls. While a hidden panel has a live turn, the button shows a spinner; when the turn completes, it shows a quiet unread dot. On wide screens the panel docks on the right and shrinks the page, with a keyboard-resizable width between 360 and 560 px. At medium widths it becomes a modal right side sheet. On phones it becomes a full-screen modal dialog. Both modal forms have a visible close button, contained focus, Escape handling, and focus restoration.

```text
┌─ Marina · Apps · Plantt ──────────────── Ask Marina · user ─┐
├──────────────────────────────┬───────────────────────────────┤
│                              │ Plantt                        │
│  chart and editor remain     │ Snowball Post-Training · r2  │
│  visible on a wide screen    │                               │
│                              │ You  What is on the path?     │
│                              │      Reading chart…           │
│                              │                               │
│                              │ ┌ Ask about this page…  Send ┐│
└──────────────────────────────┴───────────────────────────────┘
```

The header says what the agent sees and offers `New conversation`, `Open in Loom`, and close. The empty state shows up to three app-provided starter questions. There is no model, profile, repository, attachment, shell, or permission-mode selector.

The transcript follows Loom's restrained conversation style: readable prose, one live status line, collapsed completed activity, and expanded failures. It never invents operation details missing from Loom. Unsupported blocks remain visible as `Agent activity unavailable · Open in Loom`, and tool results are capped in the DOM.

The fixed composer preserves a per-context draft, sends with Command/Ctrl+Enter, and becomes `Stop` during a turn. Auto-scroll follows only at the foot; otherwise `Jump to latest` appears. Markdown, code, copy-answer, retry, offline, reconnecting, recovery, and context-change states are first-class.

A pending Loom permission becomes the most prominent card in the turn. It shows Loom's title and advertised option names. Reject options precede allow options in keyboard order, but an asynchronously arriving card does not steal focus. Choosing an option disables the card until the refreshed journal confirms its outcome. The panel never invents an operation name, argument preview, risk, or approval result.

Mutation support requires three additional contracts. Loom must journal structured, redacted tool identity and input before execution. A host-side policy must enforce the nested Marina operation's risk instead of relying on the provider to request permission. After a successful mutation, the panel must tell the active application context to invalidate and refetch its revisioned data. Plantt writes remain outside the pilot until all three exist and a disposable-chart canary passes.

### Script the Loom boundary

Marina's journey fixture starts a `ScriptedLoom` ASGI server on a second loopback port. It implements the consumed JSON/SSE routes, matches and records requests, and releases canned events through test-controlled gates. It supports multiple sessions, non-replaying reconnects, snapshots, signed-out auth, `resync`, archive, permission races, and malformed events. Acknowledgements return before gated events so launch cannot deadlock.

This is intentionally above ACP. Loom already tests its ACP adapter with a scripted agent process. Marina tests that its real browser transport and transcript reducer consume Loom correctly. A shared contract fixture copied from Loom's generated schema, plus a lightweight compatibility check against Loom's OpenAPI document when both repositories are available, limits drift.

## Testing

Reducer tests cover block upserts, delta shadows, terminal tools, mutable permissions, snapshot races, reconnects, `resync`, and duplicate suppression. Component tests cover keyboard behavior, focus restoration, scrolling, disclosures, and empty/error/recovery states.

The Plantt journey uses a deterministic chart and the real shared shell. It submits a read question, asserts the exact launch/context envelope, pauses on an in-progress tool, takes `agent-reading.png`, completes the answer, and takes `agent-answer.png`. A second scripted read turn pauses on a provider permission using Loom's current title/options shape, takes `agent-permission.png`, rejects it once, and checks the exact request and option ids. The journey then forces a reconnect and snapshot, verifies one durable answer, captures 390/900/1400 px layouts, closes the panel, and verifies focus returns to `Ask Marina`. Semantic assertions and recorded requests are the test gate; screenshots are review evidence.

After local tests pass, a cross-repository smoke starts real Loom with its existing fake ACP agent, disables loopback trust, signs in a test user, and runs Marina's browser client through explicit CORS. It covers launch attribution, journal/SSE, recovery, and human-only permission answers without model calls. A final live read smoke checks FastMCP discovery, chart read, and the grounded answer. Mutation work starts only after the structured-event, enforced-risk, and page-invalidation contracts land.

## Open Questions

- Is a 20-session, per-user `sessionStorage` map sufficient, or should the first release expose a small recent-conversation picker backed by Loom?
- Is carrying the current repository-backed `marina` profile acceptable for the pilot, or should Loom add a context-only interactive session before broad rollout?
- Should structured mutation enforcement live in a Loom MCP policy hook or in an approval-aware Marina endpoint?
