# Code-health data automation

Run the code-health aggregation or report command requested in the goal from
the repository root. Do not edit repository files, commit, push, or open pull
requests. Finelog writes and the requested report gist are the intended side
effects.

Treat any skipped classification batch, GitHub fetch failure, or Finelog flush
failure as a failed run. Include row counts and any published report URL in the
final result.
