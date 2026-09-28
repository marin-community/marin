`collection_final.json` confirms it: 8/8 verified, complete grid, `at_prespecified_grid_boundary: false`, both datasets minimizing at p70 with both neighbours worse. That is an interior turnover above 10 epochs from checkpoints you already own — the thing I said the archive couldn't support. The >10 rationale for new training is gone.

So yes, run Wikipedia on the same math metrics first; it strictly dominates a new sweep. Same evaluator, zero training, and it is the only test separating the two readings: if Wikipedia's math PPL also minimizes near p70, the preference belongs to the *metric*, not to FineMath; if it minimizes low, you have a domain × diagnostic interaction for free.

That reverses my last ranking. Nemotron Math Textbooks is now the weakest candidate, not the strongest — another math corpus would likely replicate FineMath's high optimum, which is confirmation, not separation. If Wikipedia shows the effect is metric-driven and contrast is still needed, Stack-Edu Python is the pick, paired with a code-specific diagnostic so each domain has its own.

One caveat to price: the p70–p50 gaps are 0.046/0.052 loss with no seed uncertainty. Two proxy runs at p50/p70 on a second trainer seed would secure the number the story rests on.
