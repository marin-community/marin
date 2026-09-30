# Durable construction continuations

Builder process exit status is not task completion. A session completes only when
its declared handoff exists, names real artifacts, and records nonempty checks
whose exit codes are zero. OMP usage and termination metadata are retained as
diagnostic evidence even when the process exits successfully.

Substantial sessions must write
`.capability-progress/<session>.json` before extended analysis and split work into
small units. Each completed unit names its durable artifacts and checks. The
progress file is recovery metadata: it cannot satisfy a handoff, and updates to
the progress file alone do not count as workspace progress.

When a process exits without a valid handoff, the controller resumes the same OMP
transcript with an attempt-specific continuation prompt. It preserves prior
attempt records and logs, requires the first continuation action to write a
checkpoint, and directs the agent to create a real artifact skeleton before
planning the next unit. A continuation must build from the existing workspace;
it must not restart the task or repeat the complete original prompt.

The controller hashes the builder payload while excluding controller plumbing
(`.capability-progress`, `handoffs`, and staged `tools`). A real payload change
resets the no-progress counter. Two consecutive process attempts without a
payload change stop the session. If either attempt contains an OMP
`stopReason=length`, the session remains `continuation_required` with reason
`model_output_budget_exhausted_without_workspace_progress`; otherwise the reason
is `no_workspace_progress`. The controller never turns either condition into a
successful handoff.

The motivating c22 evidence is frozen in
`docs/audits/c22_continuation_exhaustion_001.json` (SHA-256
`1375e1d7a26257ca42129c1edffc94fddbfa6652c2eda6ae73e294913bf11059`).
It records nine zero-exit attempts, 32 length stops totaling 1,048,576 truncated
output tokens, 21 compactions, and no s2 generator or handoff. Unit tests verify
counter extraction, bounded recovery, incremental prompting, truthful pending
status, and preservation of prior attempt logs.

The live continuation then completed s2 in one resumed attempt (745.89 seconds),
with 135 tool calls, no output-length stops and no compactions. The root verified
all 30 fixture files against the generator manifest: 173,251 bytes and 1,395 lines.
All nine prior attempt records and logs remain identical, and all 230 original
transcript events with IDs remain unchanged. OMP updated the title header, so the
resumed transcript is not a byte-identical prefix; the original raw transcript
remains in the frozen original pull. The evidence is
`docs/audits/c22_continuation_recovery_001.json`, SHA-256
`0a6733e6896a7b12378cf3e911b26531a8153ba210cc82ceb74846915008687b`.
This proves recovery of the previously stalled fixture stage. Packaging was in
progress at that checkpoint; it does not prove completion or acceptance of the
whole task. A subsequent independent network-blocked Daytona replay regenerated
all 30 files three times with identical hashes and passed the validator each time.
Changing the stale semaphore-holder flag, while recomputing all manifest hashes,
failed specifically on the semantic invariant. The sandbox was deleted and later
confirmed absent. See `docs/audits/c22_fixture_replay_002.json`; the earlier 429
provisioning failure remains recorded separately.

The retained original c22 artifacts do not establish the OMP executable version.
A separate local inspection used OMP 18.1.14 and found a 33K model maximum with no
output-token CLI override; that inspection is informative only and is not evidence
about another worker version. The current continuation worker reports OMP 18.2.6,
whose help and model registry must be captured from that worker before making a
version-specific configuration claim. The incremental recovery policy does not
depend on the absence of an override: it responds to the observed transcript
length stops and lack of durable workspace progress.
