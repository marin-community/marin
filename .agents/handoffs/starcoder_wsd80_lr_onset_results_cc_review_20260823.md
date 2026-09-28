# Opus-5 review: StarCoder WSD80 LR-onset results

Review performed read-only through Claude Code subscription authentication (`plambdafour@proton.me`,
`stripe_subscription`, no inherited `ANTHROPIC_API_KEY`) using `claude-opus-5` at max effort.

## Verdict

`VERDICT PASS`

The reviewer independently recomputed the load-bearing values from `measurements.csv`: the disattenuated primary
mean difference is 1.235060, the uncorrected mean difference is 1.010485, both have 8/8 positive paired seeds and
exact Wilcoxon p=0.0078125, and both Holm families reproduce. It also verified exact cosine-schedule learning rates
and bit-identical pre-onset states across arms.

## Confirmed interpretation

- The LR-schedule intervention robustly changes raw StarCoder-Nemotron gradient geometry. The uncorrected statistic
  uses no reliability correction, so the effect cannot be a correction artifact. Attenuation would shrink the
  absolute cosine toward zero rather than manufacture the observed sign reversal.
- The final uncorrected means form a duration-ordered response: +0.481 no decay, -0.354 onset at 0.90T, -0.426 onset
  at 0.80T, and -0.529 onset at 0.60T.
- The experiment does not establish optimizer-update conflict, TPP dependence, endpoint benefit, or a two-phase
  performance mechanism.

## Required disclosure fixes

1. The frozen reliability contract is ambiguous. Raw split-half reliability is below 0.5 for all eight primary
   decay rows, but decay-arm reliability is also declared advisory and only the no-decay arm is named as a gate.
   The conservative `primary inconclusive` reading is defensible but not uniquely compelled. All eight primary rows
   pass 0.5 using the Spearman-Brown reliability actually used by disattenuation (Nemotron 0.526-0.639), and the
   correction multipliers are stable.
2. Counterfactual optimizer updates call `state.take_step` at the restored state step and therefore use the next
   optimizer-step LR. Terminal decay updates are exactly zero even though the adjacent checkpoint-LR field records
   the preceding schedule step. The analysis must emit and explain both rates.
3. `combined_source_optimizer_update_statistics` is the average across reference halves of the corrected
   data-induced update, where corrected means `data_update - zero_gradient_optimizer_memory_update`. It is not an
   uncorrected update statistic; the frozen prose needed an explicit JSON path and definition.
4. Cumulative LR and restored-LR-ratio axes were listed but not analyzed. Moving onset changes cumulative optimizer
   distance as well as onset time, so those mechanisms remain unresolved.
5. The terminal optimizer-update secondary is structurally non-estimable because its vectors have zero norm.

## Stronger mechanistic alternative

The reviewer identified a convergence explanation that is more precise than harmful conflict. At stationarity of
the tied 35%/65% mixture, `0.35 g_SC + 0.65 g_N = 0` implies anti-alignment and norm ratio
`||g_SC|| / ||g_N|| = 0.65 / 0.35 = 1.857`. The 0.60T-decay arm moves from mean norm ratio 0.76 and cosine +0.447 at
0.55T to ratio 1.71 and cosine -0.529 at the endpoint while both norms collapse roughly threefold. This is
consistent with LR decay enabling convergence toward weighted-mixture stationarity. It does not identify harmful
optimizer conflict or explain two-phase endpoint gain.

## Reviewer caveats

- The exact Wilcoxon p-value is the n=8 floor and the three metric variants are sensitivities of one paired fact,
  not independent confirmations.
- Percentile bootstrap intervals are narrow at n=8, although zero is far enough away that this does not affect this
  result.
- No-decay stability is a null result rather than an equivalence test; its small drift is practically minor relative
  to the intervention effect.
