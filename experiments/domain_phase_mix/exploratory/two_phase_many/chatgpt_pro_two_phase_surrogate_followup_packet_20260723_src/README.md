# ChatGPT Pro Two-Phase Surrogate Follow-Up Packet

This is a focused follow-up packet for five existing ChatGPT Pro modeling sessions. It narrows the problem to the 39-bucket Delphi 3e18 setting and gives every session the same sanitized data, cross-session synthesis, prior-session terminal reports, frozen protocols, and standalone code.

Use one common ZIP for all five sessions. Send the distinct ready-to-send prompt assigned in `SEND_MAP.md` to each session. Each prompt contains the full shared task plus one independent assignment, so no second prompt needs to be concatenated.

## Read Order

1. `START_HERE.md`
2. `docs/SHARED_FINDINGS.md`
3. `docs/TASK_AND_EVIDENCE_POLICY.md`
4. `docs/ACCEPTANCE_GATE.md`
5. The session's file under `prompts/ready_to_send/`
6. `evidence/cross_session_phase_transport_20260723/FINAL_SYNTHESIS.md`
7. `evidence/cross_session_phase_transport_20260723/STRUCTURAL_AUDIT.md`
8. The relevant prior report under `evidence/prior_sessions/`

## Reproduce

From the extracted packet root:

```bash
uv run --no-project --script standalone_code/verify_followup_packet.py
uv run --no-project --script standalone_code/run_phase_transport_synthesis.py
uv run --no-project --script standalone_code/run_fpt_optimization_audit.py
```

The packet contains exposed development evidence only. No result obtained from it is finally confirmed until it succeeds on a newly sealed panel.
