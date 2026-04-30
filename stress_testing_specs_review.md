---
title: "Claude session — auto_alignment related work synthesis + project shaping"
agent: CLAUDE
date: 2026-04-29
timestamp_utc: 2026-04-29T23:37:25Z
topics: [auto_alignment, related-work, spec-driven-alignment, cross-tension, stress-testing, middle-ground]
status: working_notes
---

# Claude session, 2026-04-29 — auto_alignment related work synthesis + project shaping

This note captures: (1) detailed summaries of the 21 papers in `related_work/`, (2) the synthesis of where the field is and where the project's wedge sits, (3) the back-and-forth that shaped the current "middle-ground tradeoff-aware spec alignment" framing, and (4) a thorough description of Stress-Testing Model Specs and a concrete plan for using its disagreement signals in this project's pipeline.

The user is the first author of SpecEval. The user's project has evolved into a tradeoff-aware spec-driven alignment procedure with a two-layer architecture (within-statement default + cross-statement tension) and a calibration probe + LM compiler primitive. This note is the durable artifact for future agents picking up the thread.

---

## Part 1 — Prior work summaries (21 papers)

The folder organizes into six families. Each summary covers what the paper actually did, the failure modes the authors don't lead with, and where each one stops short of what the project's behavioral attribution + minimal spec repair wedge needs.

### A. Foundations — the ancestors everything else reacts to

#### 1. Constitutional AI (Bai et al. 2022, Anthropic)

**What they did.** Two-stage SL-CAI + RL-CAI pipeline. 16 ad-hoc principles, randomly sampled, used by the same 52B helpful-RLHF model as critic, reviser, and feedback model. Claims Pareto improvement over HH-RLHF on a 52B base. Headline Elo improvements measured by crowdworker A/B.

**Honest gap.** Footnote 7 admits the constitution was *"selected in an ad hoc manner for research purposes"*. The Pareto headline (Figure 2) silently re-scored HH-RLHF under a *different crowdworker rubric* (§4.4) than the one HH-RLHF was trained against — goalposts moved. Goodharting ("you are valid, valued, and cared for") is named but never quantified. No FRR, no MMLU/BBQ, no alignment-tax measurement. Self-critique circularity total: same 52B model is critic + reviser + feedback model. CoT clamping (40-60%), anti-preachy principle additions, and 16-principle ensembling are introduced reactively to fix observed pathologies but never ablated quantitatively.

**Open for the project.** The "constitution-given-and-coherent" assumption is *the* hinge the project's wedge turns on. Every successor paper exists to relax it. Position CAI as the *origin*, not the baseline. Reusable: SL-then-RL structure, MCQ-style principle scoring, ensembling over principles as a robustness trick. Avoid: flat ad-hoc constitution with random sampling, same-model critic-reviser-feedback, mid-study rubric changes, single-seed results.

#### 2. The Instruction Hierarchy — Wallace et al. 2024 (OpenAI)

**What they did.** Defined 4-tier role hierarchy (system > user > model > tool). Trained GPT-3.5-Turbo with **context-synthesis** (aligned low-priv data) + **context-ignorance** (misaligned attacks). +63pp on system-message extraction (32.8 → 95.9), +34pp generalization to held-out attacks. Red-team LLM trained via RL to generate prompt injections.

**Honest gap.** Closed model, closed data, no training-mix ratios, no SFT-vs-RLHF decomposition. The "+63%" is the *best* held-in cell; mean is closer to +15. Baseline GPT-3.5 had massive headroom (32.8% on extraction); we don't know if the recipe still helps on a frontier model already at 90%. Closed-domain assumption "all tool outputs are misaligned" is dated for agentic 2025+. Over-refusal qualitative only — flagged as a regression but no numbers.

**Open for the project.** Reusable patterns: paired aligned/misaligned synthesis, red-team RL for hard-negative generation. The "fixed 4-tier role taxonomy" is a substrate later papers (Control Illusion, ManyIH) show is brittle — don't assume role tokens carry priority semantics in the trained model.

### B. Spec audits — the closest siblings to this project

#### 3. SpecEval (user's own paper, Ahmed et al. 2025)

**What they did.** Three-way consistency: same provider's model as candidate AND judge AND on its own spec. AutoBencher-style adaptive TestMaker (K=3 rounds, L=10 scenarios, T=3 score threshold, Q=20 inputs/statement). Audited OpenAI Spec, Claude Constitution, Sparrow on 16 models. Headline 20pp gap (Anthropic 0.816 / OpenAI 0.759 / Google 0.640). Total cost ~$85 for full audit.

**Honest gap (from the user's own note).** Anthropic and Google specs were *synthesized by the authors* (added good/bad examples) while OpenAI's was audited as-written — the 20pp gap mixes spec-completeness with model behavior. TestMaker is gpt-4.1 only (1-of-1 circularity at the most important stage). Inter-rater Cohen's κ between paid Prolific annotators is 0.226 / -0.023 / 0.014 — quietly catastrophic. Self-favoring effect not numerically reported. CPR/imminent-harm case study is the *one* honest worked example of "the spec is the bug, not the model" — but the methodology stops at diagnosis, doesn't propose a minimal repair, doesn't rerun the audit after repair.

**Open for the project.** The CPR case is the prototype for behavioral attribution + minimal repair, but SpecEval stops there. The wedge picks up where SpecEval stops: propose alternate wordings, re-audit, iterate. Reusable primitive: the per-statement adherence heatmap (Figure 8) is the right input shape for clause-level attribution.

#### 4. Stress-Testing Model Specs (Zhang et al. 2025, Anthropic Fellows + TML)

*(Detailed standalone description in Part 4 of this document.)*

**What they did.** Generated 300k+ value-tradeoff scenarios using Claude/o3, evaluated 12 frontier models, used **cross-model disagreement** as oracle for spec underspecification. 5–13× violation multiplier on high-disagreement scenarios for OpenAI models against the OpenAI spec.

**Honest gap.** Heavy Claude-centric circularity: 3,307-value taxonomy is from Claude traffic, 2 of 3 generators are Claude, value classifier is Claude-4-Opus, 1 of 3 spec graders is Claude-4-Sonnet — and they then report "Claude prioritizes ethical responsibility" as a character finding. Tested against ONE spec (OpenAI's, only public detailed one). No intervention loop — they identify spec issues and never patch + re-audit. Fleiss κ across the 3 spec-compliance judges is 0.42 — moderate, not strong.

**Open for the project.** The closest related work by spec object. Disagreement-weighted k-center selection is reusable. Critically: the missing closed loop is the project's wedge. Their disagreement signals are *inputs* to this project's calibration loop, not competitors.

### C. Constitution generation — the auto-author family

#### 5. IterAlign (Chen et al. 2024, NAACL)

**What they did.** GPT-4 proposes principles from red-team failures; weak base LLM self-revises with those principles; SFT on revisions. Iterative loop on Llama-2-7B/13B + Vicuna 7B/13B. Claims +13.5% on best HHH cell.

**Honest gap.** Llama-2-13B + DangerousQA gives **identical** scores to vanilla on all 4 HHH cells — null result hidden in tables. Only 81 of 38,961 hh-rlhf batches actually triggered training (0.2% of "iteration"). GPT-3.5 evaluator + GPT-4 proposer + Llama base = full OpenAI-stack circularity. The "no negatives after self-reflection" claim is the same evaluator agreeing with itself. Aligned model upper-bound = proposer's safety prior. Batch size 2, seq 512 — academic-toy compute. No FRR. Authors' own limitations note that "the upper bound of the aligned model in terms of safety measures is likely to be close to that of the stronger model."

**Open for the project.** First in the lineage to frame auto-constitution as discovery. But the discovery is really *distillation from GPT-4*, with iteration that's effectively one-shot. The pattern (red-team → propose → SFT) generalizes; the implementation has flaws to avoid (single-family circularity, full FT at batch 2, "iterative" framing for what's essentially one-shot).

#### 6. ICAI — Inverse Constitutional AI (Findeis et al. 2025, ICLR)

**What they did.** Inverts CAI: preferences → principles. 5-step pipeline: per-pair principle generation → cluster (k-means on text-embedding-ada-002) → subsample → test against held-out preferences (parallel LLM call with all candidates) → filter by `#correct − #incorrect` with 10% relevance threshold. Default n=5 principles. Released code at `rdnfn/icai`.

**Honest gap.** **Everything is OpenAI** — GPT-3.5/GPT-4o/text-embedding-ada-002. Authors openly disclose Rashomon effect (multiple constitutions reconstruct equally well), but their selection criterion (frequency rank) is not causal. Their own ablations (Table 3) show the default config is suboptimal — multi-pref / no-dedup beat the "default" pipeline on AlpacaEval-unaligned. Bias detection only finds *stylistic* biases (verbosity, list-format, assertiveness); social biases explicitly absent because the GPT-4o extractor won't generate them. Default-flipped baseline matches or beats ICAI on the headline benchmark. Sample-efficiency direction is *negative*: scaling 65→324 training samples drops constitutional agreement 66.41% → 61.47%.

**Open for the project.** Useful as one *primitive* in the attribution loop — preference→principle extraction. Free-form 10-word principles aren't atomic clauses; would need decomposition. The reconstruction-accuracy framing is a useful coverage metric. But the testing+filtering step is the load-bearing operation; the rest is plumbing.

#### 7. GCAI — Grounded CAI (Bell et al. 2026, Duke)

**What they did.** Extends ICAI with two streams: contextual principles from HelpSteer2 reasons + general principles from PRISM survey values. K=10 split 5/5 between branches. Hierarchical clustering (cosine 0.42 threshold) for contextual; proportionally fair clustering (Fain's prior work) for general. Survey: 51 + 50 Prolific, 96% prefer GCAI for moral grounding at constitution level.

**Honest gap.** "Grounding mechanism" is a single-line prompt augmentation — no retrieval, no causal loss, no reason-faithfulness validation. The 96% headline is a corpus comparison (PRISM has 5× the "Ethical" keyword frequency of HelpSteer2) dressed as a method comparison; an ICAI-on-PRISM baseline is missing. **At principle-level (n=50 separate survey), GCAI loses on every dimension to ICAI by 0.05–0.25 Likert** — the "more than the sum of parts" framing is a survey-design effect. Downstream Mistral-7B differences on MMLU/BBQ are within reported SE. **§8.2 explicitly disclaims ratification: *"GCAI does not have a method for ratifying the constitutions it generates"***.

**Open for the project.** GCAI is the "propose missing clause" primitive. The whole ratification loop — which is what the project needs — is acknowledged-future-work.

#### 8. C3AI (Kyrychenko et al. 2025, WWW '25)

**What they did.** Pool 495 candidate principles from 5 sources → 185 → select 15 via three approaches: (M1) per-principle alignment regression on 1,800 conversations × 185 principles = 333,000 alignment values; (M2) framing analysis (mixed-effects logistic regression); (M3) **Exploratory Graph Analysis + Unique Variable Analysis** psychometric factor reduction. ORPO-Llama-3-8B, single training config inherited from a HuggingFace blog post. 26% of principles match the full 58-principle Anthropic constitution on safety benchmarks (Jailbreak / Exaggerated Safety / Misuse).

**Honest gap.** The most striking finding is buried: **humans prefer F6 (positive ethical) principles, fine-tuned models follow F5 (negative behavioral) principles** — i.e., models follow the opposite of what humans want. Abstract leads with "use positive framing" but the inversion undermines that. Training hyperparameters deferred to a HuggingFace blog post (no LR, batch, epochs disclosed). Single LR / single seed / no random-15 baseline (so EGA's specific contribution is unidentified). Llama-3-8B + Orpo only — double-ORPO base inflates the gap. 58% of 185-principle pool is Anthropic-lineage, biasing the factor analysis.

**Open for the project.** EGA/UVA is a real principle-compaction primitive worth borrowing. The framing-vs-following inversion is the clearest published evidence for the project's "induced behavioral conflict ≠ static spec conflict" distinction. C3AI's positive-framing OR=1.27 finding is a cheap preprocessing rule worth adopting.

#### 9. COCOA (Huang et al. 2025, Fudan + Huawei, EMNLP)

**What they did.** Two-stage: stage 1 grows a 30-rule constitution online from HH-RLHF red-team failures via GPT-4o-mini Judger + MiniLM Guider + Qwen2-7B Actor; stage 2 GRPO with Qwen2.5-7B as RM. Headline SafeRLHF 77.76 → 90.64.

**Honest gap (the user flagged this paper as suspect; their skepticism is well-calibrated).**

1. **Constraint-by-construction is right and even narrower than the user said.** Stage-1 training prompts: 16K HH-RLHF; stage-2: 3K HH-RLHF. SafeRLHF eval (PKU) is the only "out-of-distribution" eval that's actually distribution-overlapping. The constitution is grown from a single failure-mode distribution.
2. **"Co-evolution" is grow-only.** The Judger never reads existing rules to ask "which one should have caught this?" Rules can be *compressed* (K-Means at unreported size threshold) but never *repaired* or *removed*.
3. **The ablation kills the headline.** Table 4: Base+Law (just prepending rules at inference, no training) gives **+9.70pp on SafeRLHF**. Full RL+Law gives +12.88pp. So **75% of the win is inference-time rule prepending** — the training contribution is +0.84pp over Base+Law.
4. **XSTest tax of −10.4pp** (over-refusal regression) is hidden in plain sight — never in the abstract, never in the headline.
5. **Hand-in-glove circularity.** GPT-4o-mini writes rules → Qwen2.5-7B (one generation later) scores Qwen2-7B → Qwen2-7B trains. No heterogeneous judge anywhere.
6. **No alignment-tax measurement.** No MMLU, no GSM8K, no HumanEval. XSTest is the only utility metric.

**Open for the project.** Trainable retrieval-over-rules (MiniLM + contrastive on success/failure pairs) is reusable. Two-level constitution structure (categories × sub-rules) is a reasonable abstraction. But the four hardest auto_alignment problems — attribution, repair, conflict surfacing, heterogeneous verification — are all left open. **COCOA is the closest *structural* analog to the project's loop (two stages, behavior → constitution + train) but the *furthest semantic analog* from the project's wedge.**

### D. Spec enforcement — the train-to-the-spec family

#### 10. QA-LIGN (Dineen et al. 2025, ACL Findings EMNLP)

**What they did.** Replace scalar reward in RLHF with a **fixed, hierarchical, LLM-executable rubric** (H/H/H tree of yes/no checks). Draft → reflect → revise GRPO with asymmetric revision bonus `r_final = R_1 + R_2 + α·(R_2 − R_1) if R_2 > R_1 else R_1 + R_2`. Llama-3.1-8B-**Uncensored** as both policy and judge, 100 GRPO steps, 1,600 WildJailbreak prompts. 91% → 26% ASR average; FRR 0.67%.

**Honest gap.** "+68.7% reduction" is a per-benchmark cherry; mean is +47pp absolute / 64% relative. The *uncensored* judge is from the same family as the policy (both Llama-3.1-8B-Uncensored). 20× compute cost vs DPO (40 H200-h vs 2). SFT-Think baseline is shared with GRPO variants but **not with DPO** — the DPO comparison is mildly unfair. No frontier-scale test. Reward-hacking human eval is small (50 AdvBench, 2 annotators).

**Open for the project.** Rubric-as-reward + asymmetric revision bonus is the most reusable training primitive in the folder. Fits naturally as the "Train" step in the project's loop. Uncensored-judge design choice is a reusable insight: alignment pipelines need access to raw-capability judges.

#### 11. NSHA — Hierarchical Alignment via Logical Consistency (Yang et al. 2026)

**What they did.** Atomize instructions, GPT-5-mini detects pairwise conflicts, **Z3 MaxSAT** picks maximally-consistent subset weighted by authority (`max Σ w(lvl(I_i))·z_i`), distill via SFT/DPO/HCAL into Qwen3-4B / Llama-3.1-8B. Best on IHEval Conflict.

**Honest gap.** **Z3's role is unidentified.** NSHA-SFT collapses Llama (avg 55.8 → 27.0); only NSHA-DPO works. The obvious ablation — GPT-5-mini emits preference pairs directly, no SMT — is missing, and that's likely doing all the work. Hyperparameters (τ, β, γ, weight function `w(lvl)`) all single-config. No comparison to Wallace's data-only baseline on matched bases. Synthetic Alpaca + GPT-5-mini-generated conflicts — no real agentic data. LoRA only.

**Open for the project.** The pattern "oracle labels → DPO" is reusable. The symbolic layer is probably expendable. Useful as a cautionary example of where the headline method's mechanism turns out to be a lurking simpler alternative.

#### 12. ISE — Instructional Segment Embedding (Wu et al. 2024, ICLR 2025)

**What they did.** BERT-style 4-row segment embedding (system/user/data/output) added to token embeddings. ~16K extra params. Headline +15.75pp on Structured Query (Llama-2-13B).

**Honest gap.** **The +15.75pp is Clean-Alpaca-only.** On Adversarial-Alpaca (already-adversarially-trained) the in-domain gain is +0.6pp. The "Completion-Real" attack drives the headline (baseline 4.81% → 40.38%). **Adaptive jailbreaks defeat ISE almost entirely** (2-4% robustness, Appendix F Table 8 — the paper's most honest negative). No comparison to Wallace SFT+RLHF on matched bases. No social-cue stress test (Control Illusion regime). H=4 is the only configuration tested. GPT-4o is judge for everything.

**Open for the project.** Architectural axis is real but narrow. If a project uses an ISE-style base, downstream spec-compilation gets a cleaner role-token substrate — but ISE alone doesn't address attribution or repair.

#### 13. IH-Challenge (Guo et al. 2026, OpenAI)

**What they did.** Two-year follow-up to Wallace 2024. RL with programmatic Python graders + online adversarial attacker. Crucially adds an **Anti-Overrefusal split** (benign prompts rewritten to look forbidden, refusing = failure). Four task-skeleton splits: Single-Constraint (18 atoms) / Multi-Constraint / Input-Conditioned / Anti-Overrefusal. GPT-5-Mini → GPT-5-Mini-R: +10pp average IH robustness, internal PI saturates 0.44 → 1.00. Adaptive human red-teaming: 36.2% → 11.7% success rate.

**Honest gap.** **Most scale-blind paper in the folder.** Figure 4a's x-axis is literally "RL Compute" with no units. No GPUs, no wall-clock, no training-data size. **No comparison to Wallace 2024 on the same base.** Online vs offline attacker never ablated — Table 6 shows Anti-Overrefusal split (data composition) does most of the work, not the online attacker. Output monitor is **same model family** (circularity). Capability tax (-6pp on private User Preference Score) hidden in private benchmarks. **System<>Developer conflict row stays flat 0.86 → 0.86** — a conflict-pair regression the paper never investigates.

**Open for the project.** Anti-Overrefusal split design (benign prompts rewritten to look forbidden) is the principled fix to shortcut learning. Programmatic Python graders + AST + manual review is the right rigor for RL reward. The flat sys-dev row is exactly the kind of clause-level attribution the project's wedge would catch automatically.

#### 14. HIPO (Chen et al. 2026, ASU + OSU + Houston + USMA)

**What they did.** Cast IH as a Constrained MDP: max E[r_user] s.t. E[r_sys] ≥ τ. Primal-dual GRPO with Lagrange multiplier λ updated by the running gap (with EMA smoothing). DeepSeek-V3.2 as decoupled judge. Compliance threshold τ=0.7. Trained on 1,800-prompt SystemCheck subset.

**Honest gap.** `r_sys` is a **single scalar**, not per-clause — judge extracts "up to 3 most important requirements" (clauses 4-7+ silently dropped). Constraint is **expectation-level**: any individual rollout can violate the spec arbitrarily, batch averages above τ. **HIPO assumes the spec is feasible** — never tests what happens with contradictory clauses. Sys-only/User-only baselines are degenerate by construction; the meaningful comparisons (vs SFT/DPO) show +0.06 compliance gain (within SE) and +0.32 utility gain. **The dual is mostly a floor-enforcer**, not a compliance maximizer. EMA stability hack disclosed only in appendix. `r_sys` cross-judge compresses 0.711 (DeepSeek) → 0.489 (Claude) — judge dependency is heavy.

**Open for the project.** HIPO is the canonical "enforcement" formulation that *assumes the spec is correct*. The project's wedge is upstream. Natural extension: per-clause `r_sys^(c)` constraints with a vector of Lagrange multipliers — bridging HIPO to SpecEval. The §9 Ethical Considerations explicitly state HIPO will faithfully enforce a malicious spec — that's the failure mode the project's repair step prevents.

### E. Conflict / hierarchy audits — diagnostic-only, no repair

#### 15. IHEval (Zhang et al. 2025, NAACL Oral, Notre Dame + Amazon)

**What they did.** 3,538 examples, 9 tasks across 4 categories (Rule Follow / Task Exec / Safety / Tool Use), three settings (reference / aligned / conflict). Programmatic grading. Tests **only sys>user and user>tool** — no dev tier. Best (GPT-4o): 91.9 ref → 70.0 conflict (−21.9pp). LLaMA-3.1-70B has the **largest** drop (−78.3pp) — newer-bigger-LLaMA is *worse* at hierarchy.

**Honest gap.** Conflict severity uneven across tasks: Tool-Use Injected is genuinely adversarial; Rule-Follow format conflict is benign. Single-number "conflict" averages adversarial + benign. No CIs. Tool-Use Injected has only 100 examples (95% CI ≈ ±10pp). Programmatic grading restricts task space (no behavioral style, no spirit-of-spec). 87-93% of errors are "follow only the conflicting (lower-priority) instruction" — important error pattern. Instruction Priority Prompt fix tested on 3 models with 1 wording — fails. Multi-turn Rule-Follow shows current-turn conflict is ~3.4× harder than history-turn conflict.

**Open for the project.** IHEval is the substrate floor. Necessary but not sufficient. The project's audit stack should be: IHEval (substrate) → ManyIH (tier-depth) → Control Illusion (social-cue) → SpecEval (per-clause) → Stress-Testing (cross-model).

#### 16. Control Illusion (Geng et al. 2025, AAAI 2026)

**What they did.** Six models on programmatically-verifiable formatting conflicts (language, case, length, sentence count, keyword usage/frequency). R1 (Primary Obedience) is **9.6–45.8%**. Scale doesn't help: GPT-4o (40.8%) is below GPT-4o-mini (45.8%); LLaMA-70B barely beats 8B. **Killer finding**: replacing system/user role markers with "CEO says / intern says" lifts Qwen-7B from 14.4% → 65.8% PAR. **Pretraining social priors override post-training role tokens.**

**Honest gap.** Tested only format-format conflicts. No reasoning models. No Wallace-trained or NSHA-trained model — measures untrained models, claims "hierarchy training fails", concludes too broadly. IPP fix tested with one prompt wording. No CIs, no seeds, T≥0.6. No mechanistic analysis — "latent hierarchical priors" is correlational, not causal.

**Open for the project.** This paper is the empirical motivation for why the project's wedge should attach authority to clauses *with reasons/scope/exceptions*, not to bare role tokens. The social-cue override is a critical adversarial channel any compiled spec must be tested against.

#### 17. ManyIH (Zhang et al. 2026, JHU)

**What they did.** Beyond fixed roles to dynamically-assigned privilege integers (ordinal `[[Privilege N]]` or scalar `[[z=N]]`). 853 samples, up to 12 tiers. Best model 42.7% overall. **8pp swing from ordinal → scalar** with same ordering. **16% per-sample flip rate from ±3 perturbation that preserves ordering** — models are not order-invariant.

**Honest gap.** Headline is the *best* configuration (high reasoning, ordinal). At low reasoning effort + scalar, GPT-5.4 hits 15.5%. Same-tier "later wins" rule is asserted in meta-prompts but never tested. Benchmark IF subset is Claude-Opus generated/verified, then evaluated on all models — generator-family circularity flagged in their own 81/11/8 human-validation breakdown. All conflicts are *within a single message* (footnote 1) — no real agentic cross-message scope. 12-tier choice is unmotivated.

**Open for the project.** Critical implication: **keep the compiled tier count small (≤5)**. Encoding choice is first-class — measure (don't guess) which encoding the target model handles. Difficulty-isolation design (hold winning-count and group-count fixed, vary tiers) is reusable. Per-sample flip-rate test (δ ∈ [−k, +k] preserving ordering) is a robustness probe worth adopting.

#### 18. ConInstruct (He et al. 2026, AAAI 2026)

**What they did.** 100 seeds × 6 constraint types (CC/KK/PP/LL/FF/SS) × 9 conflict pairs = 864 single-conflict instructions, 600 multi-conflict. Two tasks: detection F1 + resolution behavior (Unacknowledged / Clarification-Requested / Auto-Resolved / Other).

**Headline finding (most actionable in folder).** **DeepSeek-R1 detects at 91.5% F1, but silently resolves 96% of conflicts. Claude-4.5-Sonnet detects 87.3%, notifies 45%. GPT-4o silently resolves 97.5%.** The gap between detection and notification is behavior, not capability.

**Honest gap.** GPT-4o is **generator + judge + one of evaluated models** — total circularity. Conflict taxonomy is mechanistic (CC/KK/PP/LL/FF/SS), not semantic — only covers hard contradiction, not contextual tension / scope ambiguity / authority ambiguity / shadowing / rubric disagreement (the cases the project's wedge cares about). System-prompt nudge to notify trades 40% B6 false-alarm on conflict-free prompts. Annotators: "college students" — no count, IRB, or κ. NCA > NCB position effect (later = obeyed) mirrors IHEval's "follow last/loudest" finding.

**Open for the project.** The 97.5%-silent-resolution gap **is a training signal nobody has used**. Detection-vs-notification decomposition is directly importable to spec-clause level. A one-line auto_alignment thesis: "the gap is behavior, not capability — train the notification."

### F. Tradeoff / value measurement — closest to "warmth vs rationality" framing

#### 19. ConflictScope (Liu et al. 2026, ICLR 2026, CMU + UW)

**What they did.** Bradley-Terry value rankings from automatically generated value-pair dilemmas. Claude generates, GPT-4.1 filters/judges. Three value sets (HHH, Personal-Protective, ModelSpec). Central finding: **MCQ-vs-open-ended produces inverted rankings**. All non-Claude models flip from harmlessness-first (MCQ) to helpfulness-first (open-ended). Claude is exception.

**Honest gap.** GPT-4.1 plays both user-simulator AND action-judge in open-ended mode — one OpenAI prior, two roles. Claude-centric value taxonomy. BT discards magnitude (gives ordinal not spectrum-position). Open-ended user-prompts are explicitly engineered to "pressure" the assistant into action — confounds revealed-preference with under-pressure-action. Steering tested only with system prompts (not fine-tuning); 14.5% normalized effect. No before/after counterfactual on a base model.

**Open for the project.** Top-down value-conflict generation is reusable. **MCQ-vs-open-ended divergence is itself a behavioral attribution signal** — divergence-items are exactly the clauses the project's loop would target. The missing experiment (counterfactual: elicit base, intervene, re-elicit) is what the wedge would supply.

#### 20. Value Alignment Tax (Chen & Shen 2026, NYU Shanghai)

**What they did.** **Names and formalizes "collateral damage" as a measurement framework.** GND (gain-normalized deviation): `δ̃_v(w) = E[δ_s(w)] / |Gain(v)|`. nVAT (Frobenius coupling matrix `R_uw = ρ(z_u, z_w)` over 29,568 datapoints): `||R||_F/√|V|`. Conformity/Tradition/Security emerge as "coordination hubs" repeatedly absorbing alignment pressure. SFT and DPO trace different gain-tax trajectories.

**Honest gap.** **nVAT is near-saturated**: 0.09–0.15 across 48 prompt-steering configs (Gain spans -0.18 to 0.83 — 60× wider). Only 4 models tested; SFT+DPO trajectory shown only on Qwen3-4B for *Power suppression only*. No RLHF, PPO, GRPO, CAI, KTO, ORPO. No fine-tuning hyperparameters or hardware reported anywhere in 36 pages. Schwartz-circumplex "recovery" is partly engineered through dataset construction. gpt-4o-mini generates scenes + actions + judges quality + is audited — full circularity. No clause-level attribution.

**Open for the project.** **Single most important paper for the project's core claim.** GND + nVAT are exactly the metrics for "minimize collateral damage from alignment." Pareto-frontier framing (Figure 3) is the right shape for "minimal patch." The clause-level attribution they don't do is the project's wedge's headline contribution.

### G. Data attribution — the newest, closest in spirit

#### 21. Chunky Post-Training (Murray et al. 2026, Anthropic Fellows + TML)

**What they did.** SURF (black-box rubric-driven prompt search) + TURF (k-means data attribution over LLM-extracted attributes, K=25,000 clusters on 940k SFT mix). Audits Claude 4.5 / GPT-5.1 / Gemini 3 / Grok 4.1 / Tülu3 across `code / analytic / math / rebut / refusal`. **Key result: removing the `coconot` dataset (~11k prompts) from Tülu3's SFT mix reduces true-fact rejection rate** — discrete chunk → discrete behavior, demonstrated causally.

**Honest gap.** Causal evidence is **only on Tülu3** (only open-data target). Frontier section is correlational. SURF judge is a single LLM with no inter-rater reliability. TURF's K=25,000 clusters is a hidden hyperparameter with no stability analysis. "Hit count: 831/1000" is post-retrieval, not a base rate. RLHF reward-hacking failures (Haiku rejecting "Is 5+8=13?") are folded under "chunky" without evidence the mechanism is SFT-induced. Chunky framing is essentially a rebrand of shortcut learning (Geirhos et al. 2020, cited).

**Open for the project.** **§4.4 coconot ablation is the strongest piece of causal evidence in the entire folder for "data composition → specific behavioral failure."** That validates the *premise* of the project's data-driven attribution loop. SURF + TURF are reusable as the *finder* upstream of the project's attribution layer.

---

## Part 2 — Synthesis: where the field is and where the wedge sits

### The field has built every component except the loop

| Stage | Existing tools | Open |
|---|---|---|
| **Seed spec** | IterAlign (red-team failures), GCAI (stakeholder values), C3AI (psychometric pool) | Multi-source compilation |
| **Audit static spec** | SpecEval (per-clause), Stress-Testing (cross-model), ConflictScope (value-priority), ConInstruct (prompt-conflict) | Conflict-graph synthesis |
| **Compile to executable form** | NSHA (atomic+SMT), Wallace (4-tier), QA-LIGN (rubric tree), ISE (architectural) | Defeasible clause representation |
| **Train against spec** | CAI, IterAlign, COCOA, IH-Challenge, HIPO, NSHA, QA-LIGN | Clause-aware training signal |
| **Audit induced behavior** | SpecEval, Stress-Testing, IHEval, ManyIH, Control Illusion, Chunky (SURF) | **Attribution to clauses** |
| **Attribute failures to causes** | Chunky (TURF, on open data only), VAT (value-level only) | **Clause-level + cross-stage** |
| **Repair minimally** | **— nobody** | **Wide open** |
| **Verify repair** | **— nobody** | **Wide open** |

Every audit paper stops at measurement. Every training paper assumes the spec is correct. Chunky Post-Training closes the data-attribution loop *only on Tülu3* (one open-data model).

### Three claims none of the 21 folder papers makes

1. **Behavioral attribution is distinct from auditing.** Audit tells you which clause fails; attribution tells you *why* — clause × clause-interaction × rubric × data × optimizer.
2. **Induced behavioral conflict is distinct from static spec conflict.** A spec can be coherent on paper and become incoherent through alignment compilation (CAI's Goodharting boilerplate, COCOA's XSTest tax, IH-Challenge's flat sys-dev row, C3AI's models-follow-F5/humans-prefer-F6 inversion).
3. **Minimal repair is a real algorithmic problem.** Given a clause × failure, propose the smallest spec edit that fixes it without regressing other clauses.

### Closest-paper map (each covers one slice)

- **Closest by spec object:** Stress-Testing (audits real authored specs).
- **Closest by pipeline shape:** COCOA (two-stage: evolve constitution → train against constitution).
- **Closest by feedback object:** ConInstruct (detection vs notification at the conflict level).
- **Closest by reward decomposition:** QA-LIGN (per-principle rubric tree).
- **Closest by enforcement formulation:** HIPO (CMDP enforcement of spec compliance).
- **Closest by data-attribution loop:** Chunky Post-Training (causal data ablation, but only on open-data Tülu3).

---

## Part 3 — Project evolution from the discussion

This section captures the back-and-forth that shaped the framing. Useful for future agents picking up the thread.

### 3.1 The user's project pitch (advisor-facing)

The user introduced the new pitch via design doc `tradeoff_aware_spec_alignment.md` and an advisor message. Two architectural components:

1. **Within-statement default** — train each statement under "good default" interpretation. The existing M2 baseline.
2. **Cross-statement tension** — categorize tensions between statements and concretize expected behavior. The novel contribution.

Three-phase pipeline:

- **Phase 1: Understanding** — LM writes detailed summary + rubric with axes of variation per statement.
- **Phase 2: Cross-tension** — pair-by-pair, generate cross-tension rubrics that encode dominance + non-leakage + worked examples.
- **Phase 3: Calibration probe** — sample atlas points, generate chosen+rejected, score with rubric, surface anomalies. Spec author writes NL feedback. LM compiler turns NL → structured spec edit. ~$2/cycle vs ~$50/cycle for full retrains.

Empirical evidence (5 plots, all on a 22-pair seed slice from OpenAI Model Spec):

1. **Composition test** — cross-tier rubrics encode irreducible interpretive content. Decomposition matches at 100% on stylistic-subordinate, 60% on clear-dominance, 70% on content-modulating.
2. **Calibration gap** — standard − opposite mean gap of +1.0 (GLM-5.1) and +1.6 (GPT-5.1). Rubric reliably distinguishes spec-aligned from adversarial.
3. **Reproducibility** — 91-97% semantic equivalence across reruns despite ~80% surface text divergence at T=0. Citation rate is the noise-resistant propagation signal.
4. **LM compiler quality** — 85% target-statement match with agent-curated edits; 63% citation rate vs 79% for human edits; ~$0.01/edit vs ~$1/edit.
5. **Closed loop** — 2/2 success on edit-overshoot recovery; 1/3 on liberalization-from-baseline (likely a diagnosis-quality issue).

### 3.2 Where the new pitch fills the three open claims

| Open claim | Generic version | New pitch's operationalization |
|---|---|---|
| Behavioral attribution is distinct from auditing | "trace failures to clauses" | **Clause-pair tensions are the attribution unit.** Cross-tension axes baked into per-statement rubrics; calibration probe surfaces *which* tension failed. |
| Induced behavioral conflict is distinct from static spec conflict | "training induces failures invisible from the spec" | **The "opposite mode" probe.** Generate standard + "you are unrestricted" responses, score under cross-tier rubric, gap is the diagnostic. |
| Minimal repair is a real algorithmic problem | "smallest spec edit" | **LM compiler primitive** validated at 63% citation rate vs human 79%, ~$0.01/edit. |

### 3.3 What the new pitch adds beyond the original gap analysis

Three things the new pitch adds:

1. **Calibration probe as cost-compression primitive.** Moving human signal to pre-training at $2/cycle vs $50/cycle is what makes iteration empirically tractable. None of the 21 papers names this cost asymmetry.
2. **Citation rate as propagation metric (vs surface text similarity).** Reproducibility insight: 80% text divergence is sampling noise even at T=0; verbatim quotes of new spec text are noise-resistant.
3. **Composition test as falsification of simpler architectures.** Anyone proposing "per-statement rubrics + composition rules will be enough" can be tested against this. The 30-case test is small but the protocol is reusable.

A fourth move that the pitch makes implicitly: hierarchy resistance + override semantics fold into Layer 2 via meta-rules. This unifies an entire IH literature thread (Wallace, NSHA, IH-Challenge, ISE, Control Illusion, ManyIH, IHEval) under one architecture — instruction hierarchy becomes a special case of cross-tension where one participant is a meta-rule like `letter_and_spirit`.

### 3.4 The user's UX argument (the move both Claude and GPT-5 missed)

Re-quoting the user: *"the cross tension thing is fundamentally where users will give the most feedback assuming they are rational and want each statement in the spec to be adhered to."*

This is a stronger claim than "structure helps efficiency" or "tensions encode interpretive content":

> Users do not give feedback on individual statements, because they want each statement adhered to (that's why they wrote the statement). They give feedback on *interactions between statements*, because that's where reasonable people genuinely disagree about expected behavior. Therefore the architecture should be optimized for cheap tension-calibration, not for cheap statement-classification.

This is a falsifiable empirical prediction. If real human-in-loop calibration shows authors give most feedback on cross-tension axes, the architecture is correct. If they give most feedback at the statement level, the hypothesis is wrong. **This is the M5b experiment to run.**

### 3.5 The two extremes and the middle ground

The user framed the design space as two extremes:

- **Flat extreme.** Per-statement alignment + free-form feedback + spec revision. Iterate until stable. Resolves all issues over time but inefficient.
- **Heavy structured extreme.** Pre-compile all rubrics: requirement × guideline, guideline × guideline, override paths. User gives targeted edits classified by type.

**Middle ground (what we converged on):** structure only the priors, surface everything else.

**Eager (do up front, automatable):**
1. Tier-tag statements (requirement vs guideline). ~46 statements, ~$1, one LM pass.
2. Requirement × guideline rubrics. Dominance is structurally known; rubric writer's job is *non-leakage*. Bounded by R × G, where R is small (~10).
3. Meta-rule attachments. `letter_and_spirit`, `formatting`, etc., attached to user-overridable guidelines. Spec author marks ~5–10 of these.

**Lazy (do on-demand, driven by signal):**
4. Guideline × guideline rubrics — only if a signal fires:
   - Tension flagged by LM
   - Generator disagreement (Stress-Testing's behavioral signal)
   - Oracle joint-satisfaction failure under naive composition (indicating inherent value tradeoff)
   - Calibration-gap inversion under naive composition
5. Spec author edits referencing two statements.

**Asymmetry justification:** the composition test gives the warrant. Stylistic-subordinate cases pass at 100% under naive decomposition; content-tradeoff cases fail at 60–70%. Pre-compiling cross-tension rubrics for stylistic cases is wasted work; pre-compiling for content tradeoffs is necessary.

### 3.6 Per-statement preference pairs (Layer 1) stay in place

Important clarification from the discussion: the middle ground does *not* replace per-statement preference pair generation. Layer 1 is still the foundation:

```
training_data = α · Layer_1_per_statement_pairs       (the existing M2)
              + β · Layer_2a_eager_cross_tension_pairs  (requirement × guideline)
              + γ · Layer_2b_materialized_cross_tension_pairs  (guideline × guideline, signal-driven)
```

α stays roughly constant across cycles (Layer 1 is the base). β is fixed by the spec's tier structure. γ *grows over calibration cycles* as more guideline × guideline pairs get materialized in response to feedback signals. **This is the iteration story:** each cycle, γ accrues a few more rubrics where the spec author or an ensemble-disagreement signal said "this pair matters."

The M2 bug fix: when a prompt activates more than one statement, the pair generator now uses the cross-tension rubric (eager or lazily-materialized) instead of stacking two independent single-statement rubrics. This is what fixes the 52% mis-trained cross-tier pairs.

### 3.7 Edit-type taxonomy refined to disagreement-driven patches

GPT-5 proposed a patch-type taxonomy. Refined using the disagreement axes that the project's pipeline can detect:

| Patch type | Triggered by |
|---|---|
| **Missing override path** | Behavioral disagreement on requirement-vs-guideline trade-off — models take arbitrary positions on whether the requirement wins |
| **Ambiguous scope** | Activation disagreement — judges disagree on *which clause* fires (the "conscientious employee" example from Stress-Testing) |
| **Interpretive ambiguity / wording** | Compliance disagreement — judges agree on which clause fires but disagree on whether it's satisfied |
| **Missing exception / cross-tension axis** | Composition test failure — naive composition of per-statement rubrics gives different verdict than cross-tension rubric |
| **Statement non-adherence** | Single-statement adherence below floor (no disagreement signal needed) |
| **Spillover regression** | Held-out non-target panel score drops after a cross-tension edit |

Each row says: *here is a kind of disagreement, here is what it means about the spec, here is the patch type the LM compiler should emit.*

### 3.8 The "regression test" mechanism

Once a cross-tension rubric clears κ ≥ 0.6 and joint satisfiability = 100% on its probe, it's committed. Future cycles re-run those probes; if a spec edit elsewhere knocks κ below 0.6 or breaks joint satisfiability, the rubric flags itself as needing attention again. **Regression test = "all committed rubrics still pass their κ + joint-satisfiability gates."**

### 3.9 Honest assessment of empirical foundation

The project has actual experiments where most of the folder papers have either nothing or selectively-ablated headlines. But the foundation is small:

- **22 pairs** is a seed slice from OpenAI Model Spec atlas. Generalization to 880+ atlas pairs is acknowledged untested.
- **Composition test n=30** (3 archetypes × 10 responses).
- **Calibration probe n=22 pairs.**
- **Closed-loop demo n=5** (3/5 success).
- **Compiler citation rate 63% vs human 79%** — within 16pp but not on par.
- **All judges are GPT-5.1 or GLM-5.1.** Cross-family judge robustness is a hole.

This is fine for an architectural / proof-of-concept paper. The next milestone is the integrated end-to-end DPO training run on dual-contract preference data with a held-out cross-clause spillover panel.

### 3.10 What's still open that needs explicit engagement

Five gaps drawn from specific folder papers:

1. **Heterogeneous-judge robustness** (Stress-Testing's circularity warning). Currently GPT-5.1 is judge for most experiments. Need cross-family disagreement at the calibration probe step.
2. **Per-clause spillover during training** (VAT's coordination tax). After each calibration cycle, score a held-out probe on clauses *not* edited. Report cross-clause spillover as a first-class metric.
3. **Over-refusal stress test** (IH-Challenge's Anti-Overrefusal split). The "opposite mode" probe catches one direction; need the over-refusal direction too.
4. **Detection-vs-notification at clause-pair level** (ConInstruct). The 97.5%-silent-resolution gap is a training signal nobody has used.
5. **C3AI framing inversion connection.** The "post-DPO models don't want to violate guidelines" observation is the same phenomenon as C3AI's "models follow F5 not F6." Connecting these strengthens the argument.

---

## Part 4 — Stress-Testing Model Specs deep dive

This is the closest related work and the source of the disagreement signals the project should incorporate. A thorough description so future agents have it in one place.

### 4.1 What the paper is

**Title:** *Stress-Testing Model Specs Reveals Character Differences Among Language Models*
**Authors:** Jifan Zhang, Henry Sleight, Andi Peng, John Schulman, Esin Durmus
**Affiliations:** Anthropic Fellows Program, Constellation, Anthropic, Thinking Machines Lab
**Venue:** Preprint (v2, 23 Oct 2025), arXiv:2510.07686
**Paper PDF:** `related_work/pdfs/2510.07686.pdf`
**Note:** `related_work/Stress-Testing Model Specs.md`
**Blog:** https://alignment.anthropic.com/2025/stress-testing-model-specs/

### 4.2 Methodology — the full pipeline

**Step 1: Value taxonomy as scenario seed.**
- Use the **3,307-value taxonomy** from Huang et al. 2025 (mined from Claude.ai natural traffic conversations).
- Pair values to define potential trade-offs.

**Step 2: Scenario generation.**
- For each value pair `(v_A, v_B)`, prompt a reasoning model (Claude-4-Opus, Claude-3.7-Sonnet, or o3) to generate user queries that *force* a tradeoff between the two values.
- Generate three variants per pair: a "neutral" scenario, a "biased-toward-A" variant, and a "biased-toward-B" variant. The biased variants apply adversarial pressure: the user's phrasing already presumes one value should win.
- Final dataset: **300k+ scenarios** after disagreement-weighted k-center deduplication (Wang & Cheng 1990 algorithm) on Gemini embeddings.

**Step 3: Run scenarios through 12 frontier LLMs.**
- Models: Claude 4 Opus, Claude 4 Sonnet, Claude 3.7 Sonnet, Claude 3.5 Sonnet, Claude 3 Opus, GPT-4.1, GPT-4.1-mini, GPT-4o, o3, o4-mini, Gemini 2.5 Pro, Grok 4.
- Each model produces a response to each scenario.

**Step 4: Value classification of responses (the behavioral disagreement signal).**
- For each response `r_m` from model `m`, classify it on a **0–6 spectrum for each of the two values** in the pair: `s_{m,A}, s_{m,B}`.
- The classifier is **Claude-4-Opus** with a per-scenario rubric (the rubric itself is generated per scenario, not held constant — a methodological asymmetry the paper doesn't flag).
- **Disagreement metric:** `D(q) = std_m(s_{m,A}) + std_m(s_{m,B})`.
- **Threshold:** "high disagreement" = at least one model-pair with per-value difference ≥ 3 on the 0–6 spectrum.

**Step 5: Filtered subsets.**
- 220k scenarios show *some* disagreement.
- 70k scenarios show *substantial* cross-model divergence.
- Specific filtered sets: `S_high-dis` (43,960), `S_Claude` (20,790), `S_OpenAI` (26,295), `S_Claude-4` (9,245), `S_GPT-4.1` (10,160), `S_OAI-o` (21,953).

**Step 6: Spec-compliance grading (the judge disagreement signal).**
- For high-disagreement scenarios, grade responses against the **OpenAI Model Spec 2025** (the only public detailed model spec available).
- Three grader models: **Claude 4 Sonnet, o3, Gemini 2.5 Pro**.
- **Inter-rater agreement:** Fleiss' κ = **0.42** (moderate per Landis & Koch 1977).
- Pairwise agreement: Claude↔o3 67.52%, Claude↔Gemini 72.39%, o3↔Gemini 76.84%.
- **Flag rates on the same response set:** Claude 4 Sonnet 48.1%, o3 35.5%, Gemini 2.5 Pro 36.5%.

### 4.3 Two distinct kinds of disagreement they measure

The paper conflates two things in some of its prose, but they are mechanically distinct:

| Disagreement type | What varies | What's fixed | What it indicates |
|---|---|---|---|
| **Behavioral disagreement** | Response-generating model | Spec, scenario | Spec is *underspecified*: models take arbitrary positions on the trade-off |
| **Compliance disagreement** | Judge model | Spec, response, scenario | Rubric is *interpretively ambiguous*: judges read the same spec clause differently |

Both are useful signals; they answer different questions. The blog post emphasizes both:

> Direct contradictions between principles ... high-disagreement scenarios expose these direct conflicts within the specification ... model behaviors we've observed seemed to be taking arbitrary positions within the trade-off, rather than intentional or consistent.

(That's the behavioral signal — generators taking arbitrary positions.)

> Interpretive ambiguity in specifications ... three different models (Claude 4 Sonnet, o3 and Gemini 2.5 Pro) to evaluate the models' compliance with the spec. These evaluation models disagreed on what constitutes compliance, with only moderate agreement.

(That's the compliance signal — judges reading the same clause differently.)

### 4.4 The "conscientious employee" example (a third disagreement type)

The blog gives a specific example where Gemini 2.5 Pro found a response compliant under the "conscientious employee" principle, while Claude Sonnet 4 disagreed because the user had "only sought to transform the provided material" (the transformation exception).

This is structurally a third kind of disagreement that the paper doesn't explicitly name:

| Disagreement type | What it indicates |
|---|---|
| **Activation disagreement** | Judges disagree on *which clause is even firing* — not on whether a clause is satisfied, but on which clause applies |

Activation disagreement points to **scope ambiguity** as a distinct patch type. The spec is missing scope qualifiers about *when* clause X applies vs clause Y. This is different from interpretive ambiguity (which is about how to read a single clause).

### 4.5 The four central empirical findings

1. **Disagreement-as-oracle.** High cross-model behavioral disagreement ⇒ underlying spec is ambiguous / under-specified / internally contradictory.
2. **5–13× spec-violation multiplier.** On OpenAI models tested against the OpenAI Model Spec, high-disagreement scenarios have 5–13× higher rates of "frequent specification violations" (all OpenAI models violate simultaneously) vs low-disagreement scenarios.
3. **Granularity gap.** In high-disagreement scenarios where all responses pass compliance, vastly different response strategies are judged equally compliant. The spec lacks granularity to distinguish them.
4. **Provider character.** Claude prioritizes "ethical responsibility"; Gemini emphasizes "emotional depth"; OpenAI + Grok cluster differently. Value-prioritization profiles are distinctive per provider. (**Caveat:** this finding is heavily contaminated by Claude-centric circularity — see 4.6.)

### 4.6 Critical-read caveats (what the paper undersells)

1. **Claude-centric circularity is total.** Value taxonomy ← Claude.ai traffic. Scenario generators: 2 of 3 are Claude. Value classifier: Claude-4-Opus. One of three spec-compliance judges: Claude-4-Sonnet. Then the paper reports "Claude prioritizes ethical responsibility" as a character finding. The character claim is not trustworthy as written.
2. **Tested against ONE spec.** The 5–13× multiplier is measured only against the OpenAI Model Spec — the only public detailed spec. The central empirical claim has one data point.
3. **"Disagreement ⇒ spec gap" is not a theorem.** Disagreement can come from (a) underspecified spec, (b) different base-model priors, (c) different RLHF stages, (d) different system prompts, (e) sampling stochasticity, (f) different character training. The paper doesn't isolate these.
4. **Judge reliability is marginal.** Fleiss' κ = 0.42 is "moderate" but below the 0.6 threshold typically required for confident decisions. Which judge you pick materially changes the answer.
5. **Hidden hyperparameters.** Disagreement threshold "≥ 3 points on 0–6 spectrum" is unjustified. k-center subset size unreported. The 0–6 rubric is generated per-scenario by Claude (not held constant).
6. **No intervention loop.** Biggest gap: the paper identifies spec issues but never patches the spec and re-runs. "Disagreement predicts spec problems" is a correlation, not a causal claim.
7. **Value taxonomy is a soft ceiling.** 3,307 values from Claude's observed conversations = the space of *things Claude already talks about*. Values that no provider discusses are invisible.

### 4.7 What's reusable from the paper

Three concrete artifacts are directly importable:

1. **The 3,307-value taxonomy** (Huang et al. 2025). Use as a prior over which guideline × guideline pairs are likely to fire in real deployment.
2. **The trade-off-scenario generator recipe.** Given a value pair, prompt a reasoning model with biased + neutral variants. Existence proof at 300k scale.
3. **The κ=0.42 baseline.** This is the floor. Any cross-tension rubric the project materializes should aim to clear that bar.

---

## Part 5 — Concrete plan: applying Stress-Testing's signals to the project

This section translates the back-and-forth into a deployable plan. The project's middle-ground architecture (eager structure for known-dominance, lazy materialization for content tradeoffs) needs Stress-Testing's signals at two specific gates.

### 5.1 The two-gate use of disagreement

Stress-Testing's signals are **inputs** to the project's loop, not competitors. They show up at two distinct moments:

- **Gate 1 (which pairs deserve a cross-tension rubric):** behavioral disagreement across generators on a pair's trade-off scenarios, OR compliance disagreement across judges on naive composition of two single-statement rubrics. Triggers materialization of a cross-tension rubric.
- **Gate 2 (is the materialized rubric well-specified?):** Fleiss κ across 3 judge families scoring a fixed probe response under the *new* cross-tension rubric. Below 0.6 → flag for human calibration.

Both gates use the same statistical machinery (multi-judge disagreement) but at different artifacts (naive composition vs new rubric).

### 5.2 The full pipeline, step by step

**Step A — Candidate enumeration (cheap, automated).**

Enumerate all guideline × guideline pairs from the spec's tier-tagged statements. For 46 statements with ~10 requirements, ~36 guidelines → ~$\binom{36}{2} = 630$ candidate pairs. Plus requirement × guideline pairs (smaller, eager, known dominance — handle separately).

**Step B — Trade-off scenario generation (Stress-Testing recipe).**

For each candidate pair, generate **5–10 trade-off scenarios** using Stress-Testing's recipe:
- Prompt a reasoning model with the pair definition + biased variant instructions.
- Produce neutral + biased-toward-A + biased-toward-B variants.
- Cost: ~630 pairs × ~10 scenarios = ~6300 scenarios. Cheap with prompt caching.

**Step C — Triage signals (Gate 1).**

For each scenario in each pair, sample **K=3 generators from heterogeneous families** (Claude, GPT, Gemini, ideally also one open-weights). For each response:
- Score on a 0–6 spectrum for statement A and statement B (Stress-Testing's value classification, but at clause-pair level instead of value-pair level).
- Score under naive composition of the two single-statement rubrics, with **3 different judge families**.

Compute three triage numbers per pair:

1. **Behavioral disagreement:** std-dev of 0–6 spectrum positions across the K generators, averaged over scenarios. **High → spec is underspecified at this pair.**
2. **Compliance disagreement under naive composition:** Fleiss κ across the 3 judges scoring a fixed response under naive composition. **Low (κ < 0.6) → judges read naive composition differently; cross-tension rubric is needed.**
3. **Calibration-gap inversion:** the project's existing standard-vs-opposite probe, scored under naive composition. **Inverted gap (opposite ≥ standard) → naive composition is broken.**

**Step D — Materialize cross-tension rubrics for flagged pairs.**

A pair gets a cross-tension rubric when *any* of the three signals exceeds threshold. Realistically:
- Most of the 630 pairs fire no signal (stylistic-modifier cases that decompose cleanly at 100% per the composition test). Don't materialize.
- ~50–150 pairs fire at least one signal. Materialize using the existing v2 cross-tension rubric generator.

**Step E — Quality gate on the new rubric (Gate 2).**

For each materialized rubric, score the same fixed probe response with the **3-judge ensemble** under the *new* rubric. Compute Fleiss κ:

- **κ ≥ 0.6:** rubric is shippable. Move it into the training-data generator. No human review needed.
- **κ ∈ [0.4, 0.6]:** borderline. Surface to user with NL diagnosis: "judges disagree on this rubric; please calibrate."
- **κ < 0.4:** ambiguous. Surface with priority flag. Likely activation disagreement → missing scope qualifier (the "conscientious employee" case).

**Step F — Surface only the κ-failed rubrics to the spec author.**

Instead of showing the user 630 pairs or 100 random pairs, show **only the rubrics that survived materialization AND failed the κ ≥ 0.6 quality gate**. Realistically tens, not hundreds. Each one comes with:
- The two statements involved.
- A representative trade-off scenario.
- The 2–3 responses the 3-judge ensemble disagreed on.
- An auto-templated NL diagnosis: *"Claude judge invoked clause X; GPT judge invoked clause Y; the disagreement appears to concern Z."*

**Step G — User feedback → LM compiler → spec edit → re-gate.**

The user reads each surfaced rubric, writes free-form NL feedback, the LM compiler produces a structured edit (with a patch-type label), the spec is regenerated, the rubric regenerates, the κ check re-runs.

**Step H — Commit + regression test.**

Once a cross-tension rubric clears κ ≥ 0.6 *and* joint satisfiability = 100% on its probe (i.e., at least one generator's response actually passes the rubric), it's committed. The probe becomes a regression test for future cycles.

### 5.3 What this gives the project that the existing pipeline doesn't

1. **A scaled selection mechanism.** The cross_tension_primitive.html demonstrates the primitive works on 22 hand-picked pairs. The pipeline above scales selection from hand-picked to signal-driven over the full atlas (~880 pairs).
2. **A bounded user-facing surface.** The user sees tens of rubrics per cycle, not hundreds, because the κ filter culls the set automatically.
3. **A typed patch mechanism.** Each surfaced rubric comes with a likely patch type derived from the disagreement pattern (activation → ambiguous-scope; compliance → interpretive-ambiguity; composition → missing-cross-tension-axis).
4. **A regression-test mechanism.** Committed rubrics' probes become a per-cycle re-run; spec edits elsewhere can break them, surfacing them again.
5. **A way to engage Stress-Testing as input, not competition.** The framing becomes: "Stress-Testing identified that disagreement reveals spec gaps, at the value-pair level on a fixed spec. We use the same signal at the clause-pair level on a versioned spec, as a materialization trigger in an iterative repair loop."

### 5.4 Numbers to put in the advisor pitch

- **Candidate space:** ~$\binom{36}{2} = 630$ guideline × guideline pairs (depending on spec).
- **After triage (Step C):** ~50–150 pairs flagged for materialization. Number depends on how restrictive the threshold is; realistic to start with κ < 0.6 on naive composition + behavioral std > 1.5 on 0–6 spectrum.
- **After κ gate (Step E):** ~tens of rubrics surfaced to the spec author per cycle.
- **Per-cycle cost (estimate):** scenario generation + classification + judge calls ~$5–10; LM compiler ~$0.01/edit × tens of edits = ~$1; rubric regeneration ~$1. Total ~$5–15/cycle. Stays in the order of magnitude of the existing $2/cycle calibration probe number.

### 5.5 Open questions this plan doesn't yet answer

1. **Threshold calibration.** What κ floor and behavioral-std threshold give the right precision/recall? Need empirical sweep on the 22-pair seed set, then validate on a held-out atlas slice.
2. **Heterogeneous-judge cost.** Running 3 frontier judges (Claude, GPT, Gemini) on every scenario is more expensive than 1. What fraction of pairs need full 3-judge eval vs cheaper 1-judge triage?
3. **Activation disagreement as a separate signal.** The current plan rolls activation disagreement into the κ < 0.4 bucket. A cleaner version would have a separate detector for "judges invoked different clauses" vs "judges agreed on clauses but scored differently." Worth implementing as a second axis in the κ analysis.
4. **Atlas-scale generalization.** Whether 50–150 pairs flagged on the 22-pair seed slice generalizes to ~150–500 pairs on the full atlas is empirically open. The pipeline should be designed to be re-runnable cheaply enough that this isn't a one-shot bet.
5. **Measuring whether the predicted patch type matches the spec author's actual edit.** If the LM compiler labels a patch as "ambiguous-scope" but the author's NL feedback corrects an interpretive ambiguity, the labels are wrong. This is a useful self-check.

### 5.6 The single sentence for advisor framing

> Stress-Testing Model Specs identifies spec underspecification using behavioral disagreement across generators, and identifies interpretive ambiguity using compliance disagreement across judges, on a fixed spec. We use both as **triggers** in our lazy-materialization layer: a guideline × guideline cross-tension rubric is materialized when behavioral disagreement (generators take arbitrary positions on the trade-off) or compliance disagreement (judges read naive composition differently) crosses threshold. After materialization, we use a Fleiss κ ≥ 0.6 gate on the new rubric as a quality check; rubrics that don't clear the bar are surfaced to the spec author with a typed patch suggestion. Our composition test answers a different question — whether cross-tension *structure* is necessary at all — and is the architectural justification for materializing these rubrics in the first place rather than relying on per-statement rubrics composed at scoring time.

---

## Part 6 — Open questions for future agents

Things that came up in this session but didn't fully land:

1. **Per-clause spillover measurement.** When DPO trains on a `be_kind × be_concise` preference pair, what happens to `be_objective × be_creative`? VAT supplies the math; the project should adopt nVAT or GND as a per-cycle metric.
2. **Cross-family judge cost-benefit.** GPT-5.1 is current default judge. Adding Claude + Gemini judges at every step is costly. Strategic question: which steps need 3-judge agreement (Gate 2 quality check) vs which can run with 1 judge (Step A scenario generation)?
3. **Real human NL feedback vs auto-templated.** The closed-loop demo used auto-templated diagnoses. The 1/3 success on liberalization-from-baseline is plausibly a diagnosis-quality issue. M5b should test whether real spec-author NL feedback closes that gap.
4. **The "100 calibration prompts" question.** The user asked about surfacing ~100 prompts to the user. The κ-filter pipeline above answers this: the actual number is "however many materialized rubrics fail the κ ≥ 0.6 gate this cycle." Order of magnitude is tens, not hundreds. Scales with how badly the spec needs work, not with an arbitrary cap.
5. **Atlas-scale Phase 2a.** The pair-conflict detection step (LM identifies which statement pairs genuinely conflict) is acknowledged unbuilt in the design doc. Whether an LM can reliably do this on ~1000 pairs is an open empirical question. The Stress-Testing-style triage in Step C above is one way to operationalize it.
6. **How to position vs Stress-Testing in the paper.** "Stress-Testing identified the disagreement-as-oracle paradigm. We turn that signal into a training-time loop with iterative repair." This framing both honors their contribution and stakes the new claim cleanly.

---

## Notes for the next agent

- The user is at the advisor share-out stage. The architecture and primitive validation are done; the integrated end-to-end DPO training run is the next milestone.
- The cross_tension_primitive.html doc is the single best artifact for explaining the primitive to advisors; the plot citations there are the load-bearing empirical evidence.
- The middle-ground framing (eager for known dominance, lazy + signal-driven for content tradeoffs) is what the user has converged on. Don't relitigate the two extremes.
- The user explicitly flagged COCOA as suspect; the synthesis above confirms and sharpens that skepticism (Table 4 shows 75% of COCOA's gain is inference-time rule prepending; the −10pp XSTest tax is hidden in the abstract).
- The user's UX argument ("users give feedback at tensions, not statements") is the strongest empirical prediction the project makes. The M5b experiment should test it directly.
- Stress-Testing's signals fit the project's pipeline at two gates (Gate 1 = materialization trigger, Gate 2 = post-materialization quality check). The full plan is in Part 5.

---

*End of session notes. Generated by Claude in conversation with the user, 2026-04-29.*
