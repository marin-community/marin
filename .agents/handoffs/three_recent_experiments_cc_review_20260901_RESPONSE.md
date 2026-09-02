# CC review: three completed data-mixing experiments

CC independently recomputed the reported values from the CSV and JSON artifacts. The review used
`claude-opus-5[1m]` at max effort through the `plambdafour@proton.me` subscription, with
`ANTHROPIC_API_KEY` removed and only read-only `Read`, `Grep`, and `Glob` tools enabled.

## Verified arithmetic

Every number in the brief reproduced. The Delphi comparisons use the same common data seed and trainer seed, so
the DSP-versus-aggregate-V comparisons are common-random-number paired comparisons. The StarCoder inventory has
375 rows, 125 coordinates per arm, identical selection classes across arms, and 0.780577 is the global minimum over
all 375 rows.

## Load-bearing corrections

1. The statement that DSP gets the Uncheatable cap ordering exactly right is not statistically informative. Caps
   4 through 10 span only 0.001621 BPB, or about 1.8 conservative same-seed noise SDs; cap 6 versus cap 8 is about
   0.06 SD. Only cap 2 versus the remaining caps is clearly resolved.
2. The DSP cross-target Table-9 inversion should not be stated as a result. The Uncheatable-targeted cap-6 row beats
   the best Table-9-targeted row by 0.002980 BPB, only about 0.73 same-seed Table-9 noise SD.
3. Aggregate-V's Uncheatable cap-8 versus cap-10 prediction is a numerical tie: the predicted values differ by
   1.2e-6 BPB. Its 0.000327 measured selection regret is about 0.36 conservative noise SD. The cap-argmin failure
   claim should be restricted to Table-9.
4. The StarCoder fixed `c109` versus `c020` pair is not an independent defense of the selected-minimum result: it
   was selected at 0.80T and transported into the most deformed 0.60T surface. A better robustness observation is
   that selection inflation is largest at 0.60T, yet 0.60T still has the smallest selected gain.
5. Untied minima search 94 coordinates while tied minima search 26, so the raw-grid gain has asymmetric positive
   winner bias under the null. The selected untied best-to-second gaps are only 0.07-0.12 of the single-run SD.
6. The StarCoder C4 tradeoff is decision-relevant. The 0.60T selected untied policy is a Pareto improvement on the
   two reported metrics; 0.80T buys programming gain with a 0.004406 C4 regression; 0.90T buys programming gain
   with a 0.120262 C4 regression. A post-hoc C4-noninferiority screen leaves gains of +0.003673, +0.004532, and
   -0.002156 BPB. This is sensitivity analysis, not a preregistered result.
7. Improving 34 of 51 Table-9 components is not independent replication because all components come from one paired
   run comparison and include correlated families. It establishes that the macro is not driven by one component;
   the more useful description is math/code improvement versus world-knowledge QA regression.

## Independent interpretation

The aggregate-V result should be stated as a Table-9-specific candidate-family success and Table-9 cap-argmin
failure. Relative to the same-target DSP best, the 0.016275 BPB gain is about 4.0 conservative same-seed noise SD;
relative to the best DSP row across targets, the 0.013295 gain is about 3.2 SD. This is a strong discovery result but
not a validated frontier. DSP's 0.006730 Uncheatable advantage is about 7.3 conservative noise SD.

The StarCoder experiment argues against the claim that moving the coupled onset from 0.80T to 0.60T increases the
programming-BPB two-phase advantage. It remains inconclusive for 0.80T versus 0.90T. It does not identify whether
phase duration or LR schedule is responsible, and it does not establish a joint-objective advantage.

CC recommended doing zero-compute selection-bias and C4-constrained sensitivity analyses first, then considering a
formal amendment from 48 to 32 confirmation runs by retaining only 0.60T and 0.80T. The reason to drop 0.90T would
be decision relevance: its selected programming policy has a 0.120262 BPB C4 regression and the selected-gain
contrast against 0.80T would be unresolved with eight seeds. Executing all 48 remains the more preregistration-pure
alternative; stopping entirely is not recommended because fresh fixed-policy seeds are the correct de-biasing
instrument.

## Cross-experiment conclusion

The surrogates are useful as candidate generators and coarse regularizer selectors, but unreliable as argmin
locators or absolute level predictors. The transferable procedure is to use multiple defensible structural heads to
widen the candidate bank, constrain them with simple physical regularizers such as epoch caps, and select among
materialized candidates with paired same-seed measurements. The claims not supported here are a new validated
frontier, exact fine-cap ordering on Uncheatable, a cross-target DSP winner on Table-9, or a joint-objective
StarCoder winner.
