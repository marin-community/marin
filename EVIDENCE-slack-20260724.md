# Additional evidence: Slack thread (#C0AHF5KV11Q, 2026-07-24, rafal/dlwh/larry)

New profile observations from David Hall on the baseline (20.558%) XProf trace
(steps-8-to-11, host s1b62nb4 — same artifact as in comment 5073017396):

1. **"combine backward looks very expensive ... I've seen this before. the crazy kernel
   i was working on i think had gains in that phase (and only in that phase). I think the
   all-gather backward is pathologically bad for this workload and it might be worth
   custom vjp work"** — i.e. the expensive adjoint is not only the gather dispatch:
   the COMBINE backward / all-gather backward is a prime custom-vjp target.
2. **"we shouldn't be needing to call unstack anywhere. unclear if this matters"** —
   look for stray unstack ops in the HLO around the a2a/combine.
3. **"the final reduce-scatter (some gradient thing) looks like it could be overlapped
   with next layer and isn't"** — specifically `reduce-scatter.10`.
   A scheduling/structural overlap opportunity independent of the a2a itself.

Token-drop fidelity data from Larry Dial (bears on the fixed-a2a bucket-granularity
concern — fixed path enforces capacity at 64 senders × 256 experts buckets, strictly more
drop-prone under sender imbalance, rafal concurs):

- "16,000 buckets is probably going to drop too many tokens. Anything beyond 64 buckets
  would require careful inspection. We have never tested beyond 16."
- "as a reference, our 1e23 run at 8 buckets ... dropped 3 pct of tokens, which is very
  small loss hit but not huge deal."

→ Every A/B MUST report measured drop fractions; treat ~3% as the known-acceptable
reference point, and treat the fixed path's fine-grained bucket drops as a first-class
result, not a footnote.

Coordination note: Rafal (rav) said he "will try" the round-robin ppermute pipelined a2a
himself. Before launching rotation experiments, check `iris job list` for /rav/ jobs to
avoid duplicating in-flight work; if he has active rotation jobs, focus on the parts he
is not covering and note it in your log.
