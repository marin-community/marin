# Debugging log for ops stub session identity

Prevent stub-mode agent sessions from colliding after a service restart.

## Initial status

The second local Playwright run failed while launching a manual question. The restarted `StubAgentGateway` generated `stub-1`, which already belonged to an earlier case in `agent_sessions.loom_session_id`.

## Hypothesis 1

The in-process counter cannot satisfy the database's cross-process uniqueness constraint. A case-derived session ID is stable across retries and distinct across cases. A restarted stub also needs to reconstruct enough in-memory state to accept a follow-up for a persisted session.

## Changes to make

- Derive stub session IDs from `case_id`.
- Reconstruct a missing stub session when a persisted session receives a follow-up.
- Cover different cases and a resumed follow-up across two gateway instances.

## Results

The regression test fails with the counter-based IDs and passes with case-derived IDs. A fresh local workflow run can create automatic and manual sessions after restarting the service.

## Future work

No follow-up work is required while production uses stub mode only as a temporary agent implementation.
