# Code-health data automation

Run the code-health aggregation command requested in the goal from the
repository root. Do not edit repository files, commit, push, or open pull
requests. Finelog writes are the intended side effect.

Treat any skipped classification batch, GitHub fetch failure, or Finelog flush
failure as a failed run. Include row counts in the final result.
