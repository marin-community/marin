# Mechanistic surrogate approach registry

This registry was written after freezing the acceptance gate and before inspecting individual 3e18 outliers. Every exposure \(e_i^{(t)}\) is measured in simulated epochs, mixture weights and phase fractions are dimensionless, latent states are dimensionless, and \(Y\) is BPB. Unless stated otherwise, \(p_i\) denotes the proportional reference weight and \(f(i)\) a predeclared semantic family.

Registry-wide parameter-symmetry conventions:

- Fixed normalized states remove a continuous state/amplitude scale symmetry: response amplitudes and intercepts have BPB units, while rate, shape, coverage, replay, and concentration parameters are dimensionless. Empirical collinearity is still reported as weak identification rather than treated as an exact symmetry.
- Bucket and family labels are fixed by the declared partition and are not exchangeable. Capability factors without fixed labels are permutation-invariant. Low-rank factorizations are additionally invariant to sign flips and, when both factors are unconstrained, orthogonal rotations; stability is evaluated on the implied interaction matrix rather than raw factors.
- Training phases are ordered and have fixed durations, so swapping phase labels is not a symmetry. A one-phase restriction ties the policies and refits the restricted form; it never averages phase-specific fitted coefficients.
- Constant design columns are absorbed into the intercept. Reference-normalized deficit, coverage, and replay features are zero at the proportional reference whenever their mechanism permits, preventing an intercept/feature-shift symmetry.

State-representation convention: every family below specifies either a dynamic transition or a static state/invariant. For a dynamic entry, the variables on the left-hand side of the governing transition are its latent state. For a static entry, the displayed policy-to-invariant map is the complete state transition; there is no history-dependent latent state beyond that sufficient statistic. The machine-readable registry preserves both fields when separately named and treats these two cases explicitly rather than inventing an artificial recurrence for a static model.

## A. Retained-state learning ODE

- **Premise:** each bucket builds a useful final-state capability while it is sampled and forgets while other data are sampled.
- **State transition:** for phase \(t\), \(s_i^{(t+1)}=s_i^{(t)}e^{-d_i(1-w_i^{(t)})D_t}+(1-e^{-q_iw_i^{(t)}D_t})\), with \(s_i^{(0)}=0\).
- **Response:** \(Y=b-\sum_i a_i s_i^{(2)}+\sum_i h_i r_i\), where \(r_i=e_i-u_i\) is literal repeated exposure beyond expected unique coverage.
- **Additional degrees of freedom:** family-pooled \(q_f,d_f\); nonnegative bucket amplitudes \(a_i,h_i\).
- **Single-phase restriction:** tie \(w^{(0)}=w^{(1)}\) and fit the same transition; no parameter is removed by hand.
- **Expected StarCoder signature:** WSD should have greater late-data leverage than cosine when \(d_{\text{StarCoder}}>d_{\text{broad}}\); the swoosh rotates rather than merely rescales.
- **Expected optimism fix:** policies starving a needed bucket lose retained state exponentially even when another bucket is heavily repeated.
- **Cheapest falsification:** fitted forgetting rates collapse to zero or are fold-unstable.
- **Status:** rejected. On frozen 3e18 heldouts it produced Uncheatable RMSE (0.02527), 12 optimism errors above (0.05), and worst optimism (0.18138); Table-9 RMSE was (0.04321) with 42 optimism errors. StarCoder leave-region-out RMSE was (0.12875) (cosine) and (0.15730) (WSD), so the transition law failed both the deployment and two-domain shape audits.

## B. Unique-coverage plus literal replay

- **Premise:** useful information is proportional to distinct examples seen; repeats beyond unique coverage add memorization or distribution-narrowing harm.
- **State/invariant:** \(u_i(e_i)=1-e^{-e_i}\) is the expected fraction of a uniformly sampled finite bucket observed at least once; \(r_i(e_i)=e_i-u_i(e_i)\) is expected duplicate exposure.
- **Response:** \(Y=b-\sum_i a_i u_i+\sum_i h_i r_i\), optionally evaluated on retained phase-weighted exposure \(e_i=\lambda_f e_i^{(0)}+e_i^{(1)}\).
- **Additional degrees of freedom:** bucket amplitudes \(a_i,h_i\), and at most one family retention \(\lambda_f\in[0,1]\); no learned replay onset.
- **Single-phase restriction:** phase tying changes only the computed exposure; the response is unchanged.
- **Expected StarCoder signature:** a smooth Nike swoosh whose upturn is fixed by finite-data duplication rather than a free threshold.
- **Expected optimism fix:** concentrated policies pay literal duplicate cost even if their maximum epoch lies outside the fit-panel range.
- **Cheapest falsification:** replay amplitudes collapse to zero on two independent panels or the fixed curvature misses both StarCoder upturns.
- **Status:** rejected. Literal unique coverage plus replay gave frozen 3e18 RMSE (0.02517/0.03922) on Uncheatable/Table-9 and StarCoder leave-region-out RMSE (0.24501/0.17757). The fixed occupancy curvature does not match either StarCoder surface and does not price concentrated Delphi policies adequately.

## C. Family bottleneck production

- **Premise:** broad evaluation requires several capabilities; a severely undercovered family can bottleneck aggregate performance even when average bucket utility is high.
- **State:** family coverage \(c_f=\sum_{i\in f}p_i u_i/\sum_{i\in f}p_i\).
- **Response:** \(Y=b+\left(\sum_f a_f(c_f+\delta)^{-\nu}\right)^{1/\nu}+\sum_i h_i r_i\), with \(a_f,h_i\ge0\), fixed small \(\delta\), and pooled \(\nu>0\).
- **Additional degrees of freedom:** one family amplitude, bucket replay amplitudes, and one bottleneck sharpness \(\nu\).
- **Single-phase restriction:** compute the same \(c_f,r_i\) from the tied schedule.
- **Expected StarCoder signature:** with two singleton families it becomes a smooth weakest-link surface; the optimum balances capability coverage before replay cost.
- **Expected optimism fix:** simultaneous starvation of many broad buckets raises BPB super-additively.
- **Cheapest falsification:** \(\nu\to0\), zero family amplitudes, or worse leave-region-out StarCoder error than the additive ablation.
- **Status:** rejected. The soft bottleneck form gave frozen 3e18 RMSE (0.03135/0.06409) and StarCoder leave-region-out RMSE (0.62692/0.14320). It improves no two independent panels and can select high-regret points on WSD.

## D. Error-mass survival / hazard

- **Premise:** pretraining removes independent latent error masses; exposure contributes a hazard rate that makes unresolved error decay multiplicatively.
- **Latent state:** \(z_k^{(t+1)}=z_k^{(t)}\exp[-\sum_i q_{ki}e_i^{(t)}]\exp[g_k\Delta_t]\), where \(z_k\) is unresolved error mass and \(\Delta_t\) is a forgetting hazard when its supporting family is absent.
- **Response:** \(Y=Y_\infty+\sum_{k=1}^K z_k^{(2)}+\sum_i h_i r_i\), with \(K\le3\) and nonnegative hazards.
- **Additional degrees of freedom:** a low-rank nonnegative bucket-to-capability loading \(q_{ki}\), capability masses, and pooled forgetting hazards.
- **Single-phase restriction:** tie the phase policy and use the same sequential hazard composition.
- **Expected StarCoder signature:** sums of exponentials represent a steep early descent and a separate replay-driven ascent without an output calibrator.
- **Expected optimism fix:** missing support leaves positive unresolved error mass instead of a finite additive benefit offset.
- **Cheapest falsification:** capability factors permute or collapse across bootstrap fits, or \(K>1\) gives no two-panel gain.
- **Status:** rejected. A one-rate family survival realization gave frozen 3e18 RMSE (0.02859/0.05109), with worst optimism (0.20420/0.27596), and StarCoder leave-region-out RMSE (0.37353/0.16701). More latent factors are not justified without a new identifying intervention.

## E. Weibull shortage and overload

- **Premise:** useful learning follows a Weibull time-to-acquisition law while repetition harm follows a separately justified duplicate-exposure law.
- **State:** \(s_i=1-\exp[-(e_i/\kappa_f)^{k_f}]\), \(r_i=e_i-(1-e^{-e_i})\).
- **Response:** \(Y=b-\sum_i a_i s_i+\sum_i h_i r_i\), with family-pooled \(\kappa_f,k_f\) and nonnegative amplitudes.
- **Additional degrees of freedom:** two nonlinear shape parameters per family plus bucket amplitudes.
- **Single-phase restriction:** identical state evaluated on a tied policy.
- **Expected StarCoder signature:** \(k_f>1\) creates a delayed learning knee; \(k_f<1\) creates rapid early gains.
- **Expected optimism fix:** a delayed acquisition threshold can price broad undercoverage without an arbitrary BPB correction.
- **Cheapest falsification:** shape parameters hit search boundaries or change ordering across folds/swarms.
- **Status:** rejected. The screened Weibull family gave frozen 3e18 RMSE (0.02488/0.03627) and failed StarCoder leave-region-out at (0.21230/0.16595). Shape parameters did not resolve the optimum-region error on two panels.

## F. Low-rank family competition

- **Premise:** simultaneous exposure to families with conflicting gradients reduces useful learning, while compatible families share representation.
- **Transition:** \(\dot s_f=q_fu_f(1-s_f)/(1+v_f^\top\sum_g v_gu_g)\), integrated within each constant-mixture phase.
- **Response:** \(Y=b-\sum_f a_fs_f^{(2)}+\sum_i h_ir_i\), with nonnegative \(q_f,a_f,h_i\) and rank-one or rank-two signed family embeddings \(v_f\).
- **Additional degrees of freedom:** one or two embedding coordinates per family beyond the additive state model.
- **Single-phase restriction:** tie phase inputs; competition remains a property of the mixture, not of the policy label.
- **Expected StarCoder signature:** asymmetric broad/code competition changes the valley orientation without phase-specific output heads.
- **Expected optimism fix:** mixtures concentrating on a few apparently strong buckets can lose shared broad capability through conflict.
- **Cheapest falsification:** embeddings are non-identifiable beyond sign/rotation, collapse to zero, or fail to improve both StarCoder leave-region-out and one multi-bucket panel.
- **Status:** rejected. The concentration-based low-rank proxy was nearly indistinguishable from the additive unique-coverage model and gave frozen 3e18 RMSE (0.02515/0.03920); StarCoder leave-region-out RMSE was (0.22249/0.17782). No transferable competition signal was identified.

## G. Soft weakest-capability survival

- **Premise:** aggregate BPB is dominated by the largest remaining capability deficit, but the dominant capability changes smoothly across policies.
- **State:** additive nonnegative deficits \(d_f=(c_f+\delta)^{-\alpha}-(1+\delta)^{-\alpha}\).
- **Response:** \(Y=b+T\log\sum_f\exp[(a_fd_f)/T]+\sum_i h_ir_i\), with temperature \(T>0\).
- **Additional degrees of freedom:** one family amplitude and a single temperature beyond the additive deficit model.
- **Single-phase restriction:** unchanged response on tied-policy coverage.
- **Expected StarCoder signature:** a smooth maximum of broad and code shortage surfaces with a valley near equalized deficits.
- **Expected optimism fix:** severe shortage cannot be canceled by surplus elsewhere.
- **Cheapest falsification:** \(T\to\infty\), unstable dominant families, or degraded ordinary-panel ranking.
- **Status:** rejected. The weakest-capability realization had the worst or near-worst frozen heldout behavior: RMSE (0.03135/0.06409), with Table-9 worst optimism (0.28907). Its additional bottleneck sharpness did not earn its complexity.

## H. Phase-specific capability heads with shared physics

- **Premise:** early exposure builds transferable representation and late exposure directly controls the final state, but both obey the same acquisition and replay laws.
- **State:** \(u_i^{(t)}=1-e^{-e_i^{(t)}}\), \(r_i^{(t)}=e_i^{(t)}-u_i^{(t)}\).
- **Response:** \(Y=b-\sum_i(a_{i,0}u_i^{(0)}+a_{i,1}u_i^{(1)})+\sum_i h_i(r_i^{(0)}+r_i^{(1)})\), with hierarchical shrinkage of \(a_{i,0}-a_{i,1}\) toward zero.
- **Additional degrees of freedom:** one phase contrast per bucket, regularized as a random effect; replay remains shared.
- **Single-phase restriction:** set \(a_{i,0}=a_{i,1}\) and fit that restriction, rather than averaging a two-head fit.
- **Expected StarCoder signature:** a phase-order-dependent valley with the same replay upturn in both phases.
- **Expected optimism fix:** only if current optimism is caused by conflating early representation with late task utility.
- **Cheapest falsification:** phase contrasts are unstable, or a shared-head ablation has equal heldout performance.
- **Status:** rejected. It ranked the cosine surface better than most first-round models but gave frozen 3e18 RMSE (0.03495/0.04315), including 20/64 optimism errors. Its WSD leave-region-out RMSE was (0.15097), and it did not preserve ordinary-panel calibration.

## I. Cumulative recency kernel

- **Premise:** the final model integrates exposure through a normalized memory kernel over training time instead of assigning an arbitrary scalar multiplier to phase 1.
- **State:** \(x_i=\int_0^1 k_\lambda(t)w_i(t)D\,dt\), \(k_\lambda(t)=\lambda e^{-\lambda(1-t)}/(1-e^{-\lambda})\).
- **Response:** unique-coverage plus literal replay evaluated on \(x_i\), while duplicate harm uses physical total epochs \(e_i^{(0)}+e_i^{(1)}\).
- **Additional degrees of freedom:** one global or family-pooled memory time constant \(\lambda\).
- **Single-phase restriction:** for a phase-tied policy, normalized recency exposure equals total policy exposure up to the fixed budget scale.
- **Expected StarCoder signature:** WSD and cosine differ through their phase boundaries; inferred memory should move in the expected direction.
- **Expected optimism fix:** prevents a free effective-exposure multiplier from erasing physical underexposure or repetition.
- **Cheapest falsification:** \(\lambda=0\), instability across the two StarCoder schedules, or no gain over total exposure on two panels.
- **Status:** rejected. This was the strongest first-round Delphi candidate but still gave frozen heldout RMSE (0.02070/0.03224), 9/8 optimism errors, and calibration slopes (1.796/1.761). WSD leave-region-out RMSE was (0.30606), so a scalar memory kernel does not transfer across schedules.

## J. Compatible-family forgetting

- **Premise:** exposure outside a semantic family overwrites its retained state, while buckets within the family are mutually compatible.
- **Transition:** group state follows a two-phase acquisition/decay ODE with decay hazard (d_f(1-w_f^{(t)})).
- **Response:** nonnegative retained-state benefits plus literal group replay harm.
- **Additional degrees of freedom:** global acquisition and forgetting shapes; one benefit and replay amplitude per structural group.
- **Single-phase restriction:** tie the phase inputs and integrate the same acquisition/decay law over the uninterrupted schedule; no forgetting or response coefficient is removed.
- **Expected StarCoder signature:** singleton families reduce to explicit broad/code cross-forgetting.
- **Expected optimism fix:** a concentrated policy loses retained capability in every family absent from the late mixture, even when total exposure is large.
- **Cheapest falsification:** failure to transfer from either StarCoder surface to frozen 3e18 heldouts.
- **Status:** rejected. It gave frozen heldout RMSE \(0.02613/0.02980\), 13/12 optimism errors, and StarCoder leave-region-out RMSE \(0.12875/0.15730\). The family-compatible transition does not improve two independent panels.

## K. Hierarchical coverage, overload hazard, and series reliability

- **Premise:** a semantic family requires coverage of its constituent structural groups; replay may either independently harm or reduce the retained useful state.
- **State:** normalized group coverage, optionally pooled by a negative-order mean or multiplied as serial reliability; the overload variant multiplies acquisition state by \(\exp(-\zeta r)\).
- **Response:** nonnegative group/family shortage and replay amplitudes.
- **Additional degrees of freedom:** one to four global shape parameters plus group amplitudes.
- **Single-phase restriction:** tie phase inputs, then evaluate the same pooled coverage, overload, and reliability operators on the tied schedule.
- **Expected StarCoder signature:** a noncompensatory valley that penalizes starving either broad or code.
- **Expected optimism fix:** serial reliability prevents surplus coverage in one family from canceling severe shortage or overload in another.
- **Cheapest falsification:** fewer optimism threshold crossings obtained only through large bias, poor ranking, or worse leave-region-out error.
- **Status:** rejected. On Table-9, hierarchical coverage reduced the optimism count to 3 only by reaching heldout RMSE \(0.06640\), bias \(+0.05737\), and calibration slope \(2.308\); Uncheatable RMSE was \(0.02905\). The overload form reached worst optimism \(0.32799\). Every route worsened WSD leave-region-out RMSE to at least \(0.16988\), except compatible-family forgetting, which merely reproduced the rejected retained-state result.

## L. Reference-support divergence

- **Premise:** local bucket gains can compensate additively, but a training distribution with missing target support incurs an importance-weight/sample-efficiency cost that cannot be canceled by oversampling another bucket.
- **State/invariant:** retained exposure ratios \(r_i=x_i/x_i^{\mathrm{prop}}\) induce
  \[
  q_i(r)=\frac{p_i(r_i+\epsilon)}{\sum_j p_j(r_j+\epsilon)}.
  \]
- **Response:** the frozen retained-state/inverse-deficit model receives one additional nonnegative term \(a_D D_{\mathrm{KL}}(p\Vert q)\). Under the Table-9 reducible-loss link, this is a multiplicative unresolved-error hazard; under the identity link it is an additive coverage cost.
- **Additional degrees of freedom:** one nonnegative support amplitude \(a_D\); the smoothing constant \(\epsilon\) is selected from a prespecified fit-panel grid.
- **Units and symmetries:** \(p,q,r,\epsilon,D_{\mathrm{KL}}\) are dimensionless. The normalization removes global exposure scale, leaving the base deficit terms responsible for scale. The only new fitted parameter is \(a_D\ge0\); \(\epsilon\) is selected on fit-panel CV.
- **Single-phase restriction:** tie the policy phases and recompute the same retained exposure and support distribution; no coefficient is removed.
- **Expected StarCoder signature:** little change along globally scaled diagonal schedules, but an increase when either domain loses relative support.
- **Expected optimism fix:** sparse or highly concentrated raw optima pay a convex support cost even when their overexposed buckets look locally beneficial.
- **Cheapest falsification:** the support amplitude collapses, the StarCoder leave-region-out geometry worsens, or calibration improves only on one target/swarm.
- **Status:** rejected after provisional promotion. The nested Delphi screen preserves OOF RMSE and Regret@1, keeps optimism counts at 4/4, and improves heldout calibration slopes from \(1.132\to1.116\) (Uncheatable) and \(1.235\to1.174\) (Table-9). The Table-9 slope-error reduction is 25.7%, which cleared the frozen material-improvement clause. The cross-swarm audit then falsified transfer: the coefficient is zero on 300M Uncheatable, 300M Table-9, production Uncheatable, and StarCoder WSD. It is active only on cosine StarCoder, where leave-region-out optimism counts and regret worsen. A support penalty fitted from residuals is not a transferable training mechanism.

## M. Sequential unresolved-error mass

- **Premise:** each bucket supplies evidence that removes a positive latent error mass; evidence is retained imperfectly when its family is absent late.
- **State transition:** with family mass \(W_f^{(t)}\), normalized phase duration \(\gamma_t\), exposure \(e_i^{(t)}\), and \(h_f^{(t)}=d(1-W_f^{(t)})\gamma_t\),
  \[
  I_i^{(t+1)}=e^{-h_f^{(t)}}I_i^{(t)}+a e_i^{(t)}\frac{1-e^{-h_f^{(t)}}}{h_f^{(t)}},
  \qquad z_i^{(t+1)}=e^{-I_i^{(t+1)}}.
  \]
  The ratio is defined as one at zero hazard. This is the exact constant-input transition of \(\dot I_i=a w_i-d(1-W_f)I_i\).
- **Response:** \(Y=b+\sum_i A_i z_i^{(2)}+\sum_g H_gR_g\), where \(R_g\) is literal duplicate exposure in structural group \(g\) and all amplitudes are nonnegative.
- **Additional degrees of freedom:** two pooled transition rates \(a,d\), one unresolved-error amplitude per bucket, and one nonnegative replay amplitude per structural group.
- **Units and symmetries:** epochs, \(I,z,a,d,h\) are dimensionless; \(A_i,H_g,b\) have BPB units. The nonlinear map prevents an exact \(a\)-\(A_i\) scale symmetry, but they can be weakly identified over a narrow exposure range and therefore require bootstrap checks.
- **Single-phase restriction:** tying the phase policy yields the same state as one uninterrupted phase whenever exposure is proportional to duration; the semigroup error is below \(6\times10^{-17}\) in the algebraic audit.
- **Expected StarCoder signature:** forgetting rotates the valley toward late code exposure while the positive error mass makes code starvation costly; replay creates the rising arm.
- **Expected optimism fix:** omitted buckets retain positive unresolved error rather than losing only a bounded additive benefit.
- **Cheapest falsification:** forgetting is zero/boundary-unstable, or the model fails either StarCoder leave-region-out or frozen Delphi calibration.
- **Status:** rejected. The algebraic audit passed, but frozen 3e18 heldout RMSE was \(0.03116/0.03685\) on Uncheatable/Table-9, with 19/7 optimism errors and worst optimism \(0.20174/0.19322\). StarCoder leave-region-out RMSE was \(0.30418/0.16799\). Acquisition and forgetting repeatedly selected the edge of the prespecified grid, so the positive error-mass response does not identify transferable dynamics.

## N. Compatible-family competition in evidence acquisition

- **Premise:** gradients from data outside a semantic family reduce the rate at which exposure supplies evidence for that family, while within-family data are compatible.
- **State transition:** use model M after replacing \(e_i^{(t)}\) with
  \[
  \widetilde e_i^{(t)}=\frac{e_i^{(t)}}{1+c(1-W_{f(i)}^{(t)})},\qquad c\ge0.
  \]
- **Response and units:** unchanged from model M; \(c\) is dimensionless and \(c=0\) is the exact nested ablation.
- **Additional degrees of freedom:** one nonnegative dimensionless competition coefficient \(c\) beyond model M.
- **Single-phase restriction:** phase tying gives a constant family mass and the same exact semigroup transition as a single uninterrupted phase.
- **Expected StarCoder signature:** positive \(c\) makes simultaneous broad/code exposure less efficient and can shift the optimum away from the diagonal; WSD should identify the direction separately from cosine.
- **Expected optimism fix:** concentrated policies are not automatically penalized. The mechanism instead tests whether ordinary mixed policies are less acquisition-efficient; it resolves optimism only if the current surrogate overcredits cross-family co-training.
- **Cheapest falsification:** \(c\) is selected at zero, changes boundary across panels, or does not improve at least one StarCoder and one multi-bucket panel.
- **Status:** rejected. The competition extension improved neither transfer nor identification: frozen 3e18 heldout RMSE was \(0.02351/0.03055\), with 11/12 optimism errors and worst optimism \(0.18220/0.17384\). StarCoder leave-region-out RMSE was \(0.28453/0.18107\). The selected competition value moved between the lower and upper grid boundaries across panels, and production fit worsened relative to model M.

## O. Monotone reducible-loss output links

- **Premise:** an exposure model may predict a latent reducible error whose nonlinear mapping to BPB is shared across policies.
- **State:** unchanged from the frozen inverse-deficit retained-state baseline.
- **Response:** \(Y=Y_\infty+(Y_0-Y_\infty)\phi(z)\), where \(z\) is the baseline latent deficit and \(\phi\) is a low-parameter positive monotone link such as a Box-Cox power.
- **Additional degrees of freedom:** one global link exponent and a BPB floor; no policy-specific terms.
- **Single-phase restriction:** unchanged because the response link does not inspect the policy class.
- **Expected StarCoder signature:** only vertical curvature changes; the argmin and level-set ordering remain unchanged for a strictly monotone link.
- **Expected optimism fix:** calibration could improve if the latent error scale is correct but compresses poor policies too strongly.
- **Cheapest falsification:** unchanged bad optimum ordering or worse heldout optimism despite improved fit-panel RMSE.
- **Status:** rejected. The prior Box-Cox deficit-link benchmark reached Uncheatable/Table-9 OOF RMSE \(0.00878/0.02091\), but frozen heldout RMSE worsened to \(0.01835/0.02419\), with 8/8 optimism errors and worst optimism \(0.16606/0.19035\). Because a monotone output link is argmin-invariant, it cannot repair an incorrect optimum surface and instead obscures the structural error.

## P. Self-coverage gates and explicit undercoverage harm

- **Premise:** a family or bucket cannot realize its fitted benefit until it reaches a minimum fraction of proportional coverage.
- **State:** retained exposure from the existing GRP transition, summarized as bucket and family exposure relative to the proportional policy.
- **Response:** either add a nonnegative soft undercoverage harm or multiply within-family benefit by \(g_f=X_f/(X_f+\rho X_f^{\mathrm{prop}})\), normalized to equal one at proportional.
- **Additional degrees of freedom:** one global undercoverage fraction or gate ratio; amplitudes remain nonnegative.
- **Single-phase restriction:** unchanged gate evaluated on the tied schedule.
- **Expected StarCoder signature:** stronger curvature when either domain is starved, without changing phase order by itself.
- **Expected optimism fix:** prevent a surplus bucket from compensating for a severely undercovered bucket or family.
- **Cheapest falsification:** undercoverage amplitude collapses or the gate worsens policy-matched heldout regret and optimism.
- **Status:** rejected. Bucket undercoverage exactly reproduced the base fit, indicating a zero or redundant channel. Hierarchical undercoverage made no material improvement. The multiplicative gate worsened two-phase 3e18 Uncheatable RMSE to \(0.02589\) and Table-9 RMSE to \(0.05384\), with selected optimism \(0.03387/0.09586\) and Table-9 Regret@1 \(0.05411\). Generic self-coverage does not explain the extreme failures.

## Q. Exact benchmark-component survival decomposition

- **Premise:** aggregate BPB is an exact average of positive component errors, so one aggregate head may hide a failed component behind gains on unrelated components.
- **State:** the same retained exposure or inverse-deficit state, with one target-specific nonnegative response head per Table-9 component.
- **Response:** \(Y=51^{-1}\sum_{c=1}^{51}Y_c\), matching the benchmark definition exactly.
- **Additional degrees of freedom:** one response head per component; nonlinear dynamics are shared in the nested version and regularization alone may vary by component.
- **Single-phase restriction:** every component head is refit on phase-tied policies under the same state law.
- **Expected StarCoder signature:** not applicable because StarCoder supplies one scalar target rather than a component decomposition.
- **Expected optimism fix:** a candidate must improve all relevant component errors rather than exploit cancellation in their average.
- **Cheapest falsification:** worse aggregate heldout calibration or regret despite exact component accounting.
- **Status:** rejected as a headline form. The tuned inverse-deficit component decomposition improved its older aggregate ablation but still reached Table-9 frozen-heldout RMSE \(0.02355\), Regret@1 \(0.02718\), five optimism errors, and worst optimism \(0.18054\). This is worse than the frozen strongest aggregate baseline on every primary heldout diagnostic except relative to weaker historical component ablations. Exact aggregation is useful bookkeeping, not the missing dynamics.

## R. Directional foundation-gated acquisition

- **Premise:** broad data builds reusable representation that increases the sample efficiency of later specialist data; the transfer direction is broad to specialist rather than a symmetric family self-gate.
- **State transition:** normalized training time is \(t\in[0,1]\), broad-family mass is \(W_F(t)\), and
  \[
  s_F(t)=1-\exp\left[-\kappa\int_0^t W_F(u)\,du\right].
  \]
  Specialist bucket \(i\) accumulates effective exposure
  \[
  x_i=\int_0^1 e_i(t)\left[1+\beta s_F(t)\right]dt,
  \]
  while foundation buckets use physical exposure. The integral is analytic for each constant-mixture phase.
- **Response:** group-level inverse-power shortage relative to the proportional reference plus literal physical replay harm; all response amplitudes are nonnegative.
- **Additional degrees of freedom:** foundation acquisition rate \(\kappa\) and transfer boost \(\beta\), plus the shared scaling exponent and transfer floor already present in the physical-exposure ablation.
- **Units and symmetries:** time, weights, \(s_F\), \(\kappa\), and \(\beta\) are dimensionless; exposure is in simulated epochs. Specialist amplitudes partly absorb a global efficiency scale, but schedule-dependent variation identifies \(\kappa\beta\). Bootstrap stability is therefore mandatory.
- **Single-phase restriction:** phase tying evaluates the exact continuous-time integral under a constant policy. At \(\beta=0\), the model reduces exactly to physical exposure.
- **Expected StarCoder signature:** broad-first/code-late schedules gain relative to tied schedules; WSD should show more directional rotation than cosine if late specialist acquisition matters.
- **Expected optimism fix:** specialist surplus cannot earn full credit when broad foundation exposure is delayed or absent; physical replay still prices extreme repetitions.
- **Cheapest falsification:** no improvement over \(\beta=0\) on both a StarCoder surface and one multi-bucket panel, or unstable boundary selection of \(\kappa,\beta\).
- **Status:** rejected. The algebraic audit passed, but the transfer parameters were not stable: the selected boost hit its upper grid boundary on both Delphi targets and the selected acquisition/boost changed across the StarCoder schedules. Frozen 3e18 heldout RMSE was (0.02135/0.02790) on Uncheatable/Table-9, with Regret@1 (0.06837/0.05378) and 9/9 optimism errors. StarCoder leave-region-out RMSE was (0.26489/0.16914), substantially worse than the strongest existing surface models. The directional transition is therefore falsified rather than merely under-regularized.

## S. Two-level equivalent prior exposure

- **Premise:** pretraining on broad foundation data supplies transferable prior evidence to every capability, while specialist buckets begin with less equivalent prior exposure and therefore become sharply costly when omitted. This is a scaling-law prior, not an output calibration.
- **Latent state:** each structural group has dimensionless evidence ratio (r_g=x_g/x_g^{\mathrm{prop}}). Optionally, (x_g) is the exact normalized exponential-memory integral of exposure with global recency rate \(\lambda\); at \(\lambda=0\), it is physical total exposure.
- **Response:** with foundation family \(F\),
  \[
  d_g=(r_g+\delta_{z(g)})^{-\alpha}-(1+\delta_{z(g)})^{-\alpha},\qquad
  z(g)\in\{F,S\},
  \]
  and \(Y=b+\sum_g A_gd_g+\sum_gH_gR_g\), where \(R_g\) is literal physical replay and all amplitudes are nonnegative.
- **Additional degrees of freedom:** two positive dimensionless prior scales \(\delta_F,\delta_S\), shared exponent \(\alpha\), and optionally one recency rate \(\lambda\). Setting \(\delta_F=\delta_S\) recovers the common-prior scaling deficit; setting \(\lambda=0\) removes phase order.
- **Units and symmetries:** exposures and priors are in simulated epochs normalized by proportional exposure, hence dimensionless. Amplitudes have BPB units. Over a narrow ratio range, \(\alpha\), \(\delta\), and amplitudes can trade scale, so fold/related-swarm stability is mandatory.
- **Single-phase restriction:** tie phase weights and evaluate the same evidence state. The normalized recency integral then equals total exposure; no response parameter is removed or averaged.
- **Expected StarCoder signature:** the common-prior ablation cannot rotate a phase-order surface; a stable nonzero recency rate can. A smaller specialist prior should steepen the code-starvation arm without changing the high-repetition arm.
- **Expected optimism fix:** policies omitting specialist support retain a large positive unresolved deficit even if another bucket is heavily overexposed.
- **Cheapest falsification:** \(\delta_F\) and \(\delta_S\) collapse together or swap ordering across swarms, recency is schedule-unstable, or the model cannot reduce extreme optimism without worsening fit-panel RMSE/regret.
- **Status:** rejected. In the generic screen, the intended smaller specialist prior did not transfer: selected floors collapsed equal on most panels and reversed on Delphi Table-9 and production. The strongest-baseline nested relaxation selected \((\delta_F,\delta_S)=(3,1)\) on Delphi Uncheatable but worsened heldout RMSE from \(0.01227\) to \(0.01703\), calibration slope from \(1.132\) to \(1.433\), optimism count from 4 to 9, and Regret@1 from \(0.00215\) to \(0.00544\); Table-9 selected the exact common-prior baseline. The premise is not identified consistently enough to support extrapolation.

## T. Family-specific physical replay hazard

- **Premise:** finite buckets differ in how repeated examples interfere with their target capabilities, so collision load should be conserved physically but may have a family-specific BPB response. A shared coefficient can underprice a 99-epoch specialist bucket because it is fitted mostly on moderate broad-data replay.
- **Latent state:** physical simulated epoch (e_i) induces either collision load (c_i=\max(e_i-1,0)^2) or duplicate mass (r_i=e_i-(1-e^{-e_i})). These depend on materialized repetition, not retained/effective exposure.
- **Response:** replace the frozen model's shared (H\sum_i c_i) with (\sum_f H_f\sum_{i\in f}c_i), (H_f\ge0). Equal (H_f) recover the shared model exactly. Duplicate-mass response is a separate shape ablation.
- **Additional degrees of freedom:** (F-1) response contrasts for (F) predeclared families; there is no new state-transition parameter.
- **Units and symmetries:** (e,c,r) are dimensionless; (H_f) has BPB units. Family coefficients are identifiable only where the fit panel independently varies repetition by family.
- **Single-phase restriction:** physical replay depends only on aggregate realized epochs, so tying phases leaves the same replay state and refits the same response form.
- **Expected StarCoder signature:** with two singleton families, the two arms of the Nike swoosh may have different high-repetition curvature, but the valley orientation is unchanged.
- **Expected optimism fix:** extreme repeated-family policies receive a family-appropriate collision cost instead of the average shared cost.
- **Cheapest falsification:** family coefficients are zero or fold-unstable, or the exact nested relaxation fails to improve both 3e18 targets without violating OOF/Regret@1 gates.
- **Status:** rejected. Family collision coefficients were active, but the quadratic tail was structurally wrong: the Table-9 model predicted BPB \(23.04\) for a heldout raw optimum with observed BPB \(1.47\), giving heldout RMSE \(1.34\). Duplicate-mass replay avoided explosion but did not clear the gate (Uncheatable/Table-9 heldout RMSE \(0.01291/0.02402\), optimism count 5/4, worst optimism \(0.09893/0.1398\)) and worsened OOF selection regret. This is an identified response-shape failure, not a ridge problem.

## U. Bounded retained unique-coverage state

- **Premise:** each bucket has a literal seen-data fraction \(u_i\in[0,1]\); new exposure can only cover unseen mass, while absence of a compatible family restores unseen mass through forgetting.
- **State transition:** in phase \(t\), first retain
  \[
  \widetilde u_i^{(t)}=u_i^{(t)}\exp[-d\gamma_t(1-W_{f(i)}^{(t)})],
  \]
  then acquire unique coverage exactly under a Poisson exposure model,
  \[
  u_i^{(t+1)}=\widetilde u_i^{(t)}+
  (1-\widetilde u_i^{(t)})(1-e^{-e_i^{(t)}}).
  \]
- **Response:** weighted group coverage is normalized by its proportional-policy value, mapped through a nonnegative inverse-power shortage head, and combined with literal physical duplicate-mass replay.
- **Additional degrees of freedom:** one global dimensionless forgetting rate \(d\), shortage floor, and exponent; response amplitudes are nonnegative.
- **Units and symmetries:** \(u,d,\gamma,W\) are dimensionless; \(e\) is in simulated epochs; amplitudes have BPB units. At \(d=0\), \(u_i=1-e^{-\sum_t e_i^{(t)}}\) exactly, independent of phase subdivision.
- **Single-phase restriction:** tying phases gives the same uninterrupted Poisson coverage state when \(d=0\); for \(d>0\), the fixed phase boundary is part of the retention mechanism and the tied model is refit under that transition.
- **Expected StarCoder signature:** saturation supplies the swoosh's high-exposure flattening while family absence rotates the valley toward late specialist coverage.
- **Expected optimism fix:** a heavily repeated bucket cannot compensate through unbounded benefit for another bucket whose coverage remains near zero.
- **Cheapest falsification:** forgetting selects a boundary across panels, or boundedness fails to improve 3e18 optimism and StarCoder leave-region-out jointly.
- **Status:** rejected. All seven panels selected the maximum screened forgetting rate \(d=4\), so the transition is not identified in the interior. Frozen 3e18 Uncheatable/Table-9 heldout RMSE was \(0.02493/0.03399\), with 11/9 optimism errors and worst optimism \(0.1952/0.2497\). StarCoder cosine/WSD leave-region-out RMSE was \(0.0615/0.1590\). The bounded state prevents numerical explosion but does not explain the out-of-support tail.

## V. Complementary CES capability production

- **Premise:** a benchmark family is produced jointly by several capabilities; surplus in one group cannot freely replace a missing essential group. This replaces the additive deficit response rather than adding a bottleneck feature to it.
- **Latent state:** normalized group evidence \(r_g=x_g/x_g^{\mathrm{prop}}\), using either physical exposure or the independently defined bounded retained-coverage transition from model U.
- **Response:** fixed group shares \(q_g\) are their proportional-policy masses. For family \(f\),
  \[
  D_f=\left[\sum_{g\in f}q_g(r_g+\delta)^{-\rho}\right]^{\alpha/\rho}
  -(1+\delta)^{-\alpha},
  \qquad Y=b+\sum_f A_fD_f+\sum_fH_fR_f,
  \]
  with \(A_f,H_f\ge0\) and literal physical duplicate mass \(R_f\).
- **Additional degrees of freedom:** global complementarity order \(\rho>0\), equivalent-prior floor \(\delta>0\), scaling exponent \(\alpha>0\), and optional global forgetting rate. The response has two nonnegative coefficients per declared family plus an intercept.
- **Units and symmetries:** \(r,q,D,\rho,\delta,\alpha\) are dimensionless; \(R_f\) is in simulated epochs and its coefficient has BPB per epoch units; \(A_f\) has BPB units. Over a narrow exposure range \((\rho,\delta,\alpha,A_f)\) may trade curvature, so cross-panel and fold stability are required.
- **Single-phase restriction:** tying phase weights gives the same physical evidence ratios; the retained variant uses the same bounded state transition under the tied policy and is refit without phase-specific response parameters.
- **Expected StarCoder signature:** singleton families reduce to ordinary scaling deficits, while multi-group families show a steeper joint-starvation arm. The physical variant cannot rotate phase-order level sets; retained evidence must supply any rotation.
- **Expected optimism fix:** policies that simultaneously starve several groups in one family receive super-additive deficit rather than allowing groupwise compensation.
- **Cheapest falsification:** no gain over additive deficits on both 3e18 targets, weakly identified complementarity order, or failure to transfer from multi-bucket panels to StarCoder/production.
- **Status:** rejected. The physical and retained variants both failed the frozen screen. Fit-only selection chose the least complementary screened order \(\rho=0.25\) on five of seven panels, while the retained variant again selected maximum forgetting \(d=4\). Frozen 3e18 Uncheatable/Table-9 heldout RMSE was \(0.02652/0.05212\) for retained CES, with 8/4 optimism errors and worst optimism \(0.1594/0.1638\); physical CES was no better overall. StarCoder cosine/WSD leave-region-out RMSE was \(0.0615/0.1590\) for the retained form and \(0.2144/0.1686\) physically. The response is driven toward weak complementarity and does not transfer.

## W. Novel-sample acquisition with replay-induced forgetting

- **Premise:** repeated samples do not merely stop adding information; optimization on duplicate tokens can erode competence acquired from novel tokens elsewhere. This directly couples the two observed extreme-policy signatures: severe starvation and heavy repetition.
- **State transition:** bucket exposure \(E_i(t)\) gives unseen probability \(e^{-E_i(t)}\). For current policy \(w(t)\), the duplicate-token fraction is
  \[
  q(t)=\sum_i w_i(t)(1-e^{-E_i(t)}),
  \]
  and competence follows
  \[
  \dot s_i(t)=\dot E_i(t)e^{-E_i(t)}-d\,q(t)s_i(t).
  \]
  The piecewise-constant policy ODE is integrated deterministically across both phases.
- **Response:** retained bucket competence is grouped and normalized by its proportional-policy state, then mapped through a nonnegative inverse-power deficit. Replay is not added again as a response feature because it already enters the transition hazard.
- **Additional degrees of freedom:** one global dimensionless replay-hazard rate \(d\), shortage floor, and exponent; one nonnegative BPB amplitude per structural group.
- **Units and symmetries:** normalized time, \(E,s,q,d\) are dimensionless; response amplitudes have BPB units. At \(d=0\), \(s_i=1-e^{-E_i}\) exactly. Scaling \(d\) cannot be absorbed into a response amplitude because it changes phase order and cross-bucket coupling.
- **Single-phase restriction:** tie the policy across phases and integrate the same ODE; changing an artificial boundary while keeping the exposure rate fixed changes the state by less than numerical tolerance.
- **Expected StarCoder signature:** high exposure creates a rising arm through duplicate-token hazard, and late replay is more harmful because there is less subsequent novel acquisition to recover competence. WSD should therefore rotate more than cosine.
- **Expected optimism fix:** an over-repeated bucket actively raises the predicted cost of simultaneously starved groups, rather than contributing an independent additive replay term.
- **Cheapest falsification:** \(d\) collapses or hits a boundary, the Nike swoosh/phase rotation is absent, or the interaction fails to reduce severe 3e18 optimism without losing ordinary fit.
- **Status:** rejected. The algebraic limits passed and the hazard was active, but it was not transferable: fit-only selection ranged from \(d=1\) to the upper boundary \(d=16\) across panels. Frozen 3e18 Uncheatable/Table-9 heldout RMSE was \(0.04279/0.08676\), Regret@1 \(0.08987/0.11804\), with 6/3 optimism errors and worst optimism \(0.1980/0.1939\). StarCoder cosine/WSD leave-region-out RMSE was \(0.1348/0.1481\). A single global duplicate-token hazard does not explain the starvation-plus-replay tail.

## X. Target-weighted distinct-data scaling

- **Premise:** a benchmark values a bucket according to the amount of distinct target-relevant information observed, with sampling either Poisson or approximately without replacement and optional forgetting of early evidence.
- **Latent state:** each bucket accumulates target-weighted unique coverage under a finite-population sampling law; a global forgetting rate can discount phase-0 coverage before the final response.
- **Response:** nonnegative inverse-power shortage in normalized distinct coverage plus literal replay harm.
- **Additional degrees of freedom:** sampling law, one acquisition exponent, one prior floor, and one forgetting rate; all are selected on fit-panel OOF.
- **Single-phase restriction:** tie the phase policy and evaluate the same distinct-coverage law; forgetting applies to the same artificial boundary and must therefore vanish or preserve the semigroup limit.
- **Expected StarCoder signature:** finite-data saturation should reproduce the descending arm and replay should reproduce the ascending arm without a free optimum calibrator.
- **Expected optimism fix:** concentrated policies cannot obtain unlimited benefit from repeated target-relevant samples.
- **Cheapest falsification:** ordinary-panel rank collapses, phase forgetting is schedule-unstable, or neither StarCoder surface is predicted out of region.
- **Status:** rejected. Frozen 3e18 heldout RMSE was \(0.03968/0.08124\) for Uncheatable/Table-9, with heldout Spearman \(0.237/-0.302\). The 300M OOF Spearman values were \(-0.012/-0.105\), and StarCoder cosine/WSD leave-region-out RMSE was \(0.18035/0.15284\). Forgetting collapsed to zero except on already-poor surfaces. Target-weighted distinct coverage does not preserve the ordinary fit, so it cannot explain optimum-region failures.

## Y. Finite-corpus collision load

- **Premise:** repeated draws reduce the effective sample size even before exact duplicate saturation; the physical collision probability is an invariant of the sampling distribution and bucket sizes.
- **Latent state:** within- or across-phase Kish effective sample size and its implied collision load, computed from realized token allocations and fixed bucket sizes.
- **Response:** the frozen deficit model receives one nonnegative collision-load term. The zero coefficient is the exact nested ablation.
- **Additional degrees of freedom:** one BPB-valued response amplitude and a prespecified choice of within-phase versus aggregate collision invariant.
- **Single-phase restriction:** the aggregate invariant is unchanged by phase tying; the within-phase invariant reduces to the same formula on repeated tied policies.
- **Expected StarCoder signature:** high concentration raises the swoosh arm according to finite-corpus reuse, while the valley location remains controlled by the deficit state.
- **Expected optimism fix:** concentrated policies pay a physical sample-efficiency cost even when their fitted bucket benefit remains favorable.
- **Cheapest falsification:** Regret@1 worsens, or calibration improves on only one target without a transferable coefficient.
- **Status:** rejected as a headline model. The within-phase invariant improved Uncheatable heldout RMSE \(0.01227\to0.01183\), calibration slope \(1.132\to1.026\), and optimism count \(4\to3\), but worsened Regret@1 from \(0.00215\) to \(0.01081\). On Table-9 it improved the slope \(1.235\to1.059\) and worst optimism \(0.15871\to0.13624\), but heldout RMSE worsened to \(0.02101\) and the optimism count stayed at four. The invariant detects replay load but does not preserve decision quality on both targets.

## Z. Collision-limited acquisition

- **Premise:** collisions should reduce acquisition itself rather than enter as an independent output penalty: the effective evidence dose is physical exposure divided by a collision-dependent inefficiency factor.
- **State transition:** replace the frozen model's exposure by \(e_i/[1+c\,C_i]\), where \(C_i\) is the fixed finite-corpus collision invariant and \(c\ge0\).
- **Response:** unchanged frozen inverse-deficit response, so \(c=0\) is an exact nested ablation.
- **Additional degrees of freedom:** one dimensionless collision sensitivity.
- **Single-phase restriction:** tie phases and recompute the same collision-limited dose.
- **Expected StarCoder signature:** the high-repetition arm rises because duplicate-heavy exposure stops producing evidence, not because of post-hoc harm.
- **Expected optimism fix:** raw optima cannot convert arbitrarily concentrated exposure into equal useful evidence.
- **Cheapest falsification:** fit-panel CV selects \(c=0\) on either primary target.
- **Status:** rejected. Fit-panel OOF selected \(c=0\) on both 3e18 targets, exactly recovering the frozen baseline. The acquisition coupling is unsupported by the training panel and cannot be retained based on heldout behavior.

## AA. Importance-weight effective-sample-size scaling

- **Premise:** a mixture far from the proportional data distribution has fewer effective target-distribution samples because importance weights have high variance.
- **Invariant:** smoothed policy-to-proportional density ratios induce the standard importance-sampling effective sample-size fraction, optionally aggregated across phases.
- **Response:** the frozen model's evidence state is scaled by a fixed power of this fraction before the same deficit response.
- **Additional degrees of freedom:** one smoothing floor and one ESS exponent, selected only by fit-panel OOF.
- **Single-phase restriction:** tie phase distributions and evaluate the same importance ESS.
- **Expected StarCoder signature:** off-diagonal concentration loses evidence efficiency while nearby diagonal schedules are nearly unchanged.
- **Expected optimism fix:** support-deficient raw optima receive less effective evidence rather than an arbitrary BPB surcharge.
- **Cheapest falsification:** the exact identity ablation is selected, or heldout calibration changes without RMSE/regret improvement.
- **Status:** rejected. Uncheatable selected the exact identity ablation. Table-9 selected floor \(0.1\), exponent \(0.25\), but heldout RMSE worsened \(0.020171\to0.020245\), optimism count remained four, and the calibration-slope error improved only 18%, below the frozen 20% materiality gate. The effect is too weak and target-specific to support a transferable mechanism.

## AB. Phase-boundary adaptation debt

- **Premise:** changing the data distribution at a phase boundary perturbs optimizer statistics. Late novelty, abandonment of early support, or phase-label mutual information may create an adaptation cost not represented by exposure alone.
- **Invariant:** with proportional-prior-smoothed phase distributions \(q_0,q_1\), test \(\gamma_1D_{\mathrm{KL}}(q_1\Vert q_0)\), \(\gamma_1D_{\mathrm{KL}}(q_0\Vert q_1)\), and the phase-label Jensen-Shannon information. Every term is dimensionless and zero for a tied policy.
- **Response:** one nonnegative BPB-valued amplitude is added to the frozen deficit model; zero is the exact nested ablation.
- **Additional degrees of freedom:** one amplitude; the proportional-prior pseudocount is selected on fit-panel OOF.
- **Single-phase restriction:** all three invariants are exactly zero when phase policies are tied.
- **Expected StarCoder signature:** WSD 80/20 should exhibit a different phase-change cost than cosine 50/50, while the diagonal remains unchanged.
- **Expected optimism fix:** highly asymmetric predicted optima pay the optimizer's distribution-shift adaptation cost.
- **Cheapest falsification:** no term clears the frozen materiality gate on both targets, or an apparent result depends on bucket-count-dependent smoothing.
- **Status:** rejected after a provisional gate pass. In the original joint refit, phase-label information reduced heldout calibration-slope error by 23.1% on Uncheatable and 21.6% on Table-9 while preserving the four optimism errors, so it cleared the frozen materiality rule. A stricter transfer audit then froze each panel's strongest pre-search surrogate and fit only the nested transition-debt amplitude. The coefficient was exactly zero on Delphi Table-9, 300M Uncheatable, production Uncheatable, cosine StarCoder, and WSD StarCoder; it was positive in only 3/5 folds on 300M Table-9. Only Delphi Uncheatable retained a stable but tiny amplitude, with essentially unchanged heldout diagnostics. The initial gain came from jointly moving the other 57 coefficients in a weaker base form, not from transferable evidence for adaptation debt.

## AC. Learning-rate plasticity exposure

- **Premise:** an example's durable effect is proportional to its exposure weighted by optimizer plasticity, so early and late tokens should contribute according to the integrated learning-rate schedule rather than a freely learned late multiplier.
- **State:** \(x_i=\sum_t e_i^{(t)}\bar\eta_t^q\), where \(\bar\eta_t\) is the phase-average normalized learning rate and \(q\ge0\) is a global plasticity exponent.
- **Response:** the same bounded shortage and replay response is evaluated on \(x_i\); physical repetition remains measured by realized epochs.
- **Additional degrees of freedom:** one dimensionless exponent \(q\), with \(q=0\) the exact total-exposure ablation.
- **Single-phase restriction:** tied policies use the same schedule-weighted exposure; only the fixed learning-rate schedule distinguishes phases.
- **Expected StarCoder signature:** cosine 50/50 and WSD 80/20 should rotate differently because their phase-average learning rates differ.
- **Expected optimism fix:** low-learning-rate late overexposure cannot masquerade as equal effective evidence.
- **Cheapest falsification:** both StarCoder surfaces select \(q=0\), or schedule-specific estimates disagree.
- **Status:** rejected. Both StarCoder surfaces selected \(q=0\), exactly recovering the physical-exposure ablation. Cosine/WSD leave-region-out RMSE remained \(0.2144/0.1686\), with WSD Regret@1 \(0.652\) BPB. Learning-rate weighting alone does not generate the observed phase-order surface.

## AD. Gradient-noise-limited acquisition

- **Premise:** concentrated mixtures provide fewer independent gradient directions, reducing useful acquisition according to a concentration-dependent signal-to-noise factor.
- **State:** phase exposure is multiplied by \((1+k\sum_i w_i^2)^{-1}\), with \(k\ge0\), before the same bounded acquisition transition.
- **Response:** unchanged bounded shortage and physical replay response; \(k=0\) is the exact nested ablation.
- **Additional degrees of freedom:** one dimensionless noise sensitivity \(k\).
- **Single-phase restriction:** tie policies and apply the same concentration factor in both phase intervals.
- **Expected StarCoder signature:** corners lose acquisition efficiency and the high-concentration arms rise without an output penalty.
- **Expected optimism fix:** highly concentrated raw optima cannot turn every token into independent evidence.
- **Cheapest falsification:** selected \(k\) differs substantially by schedule or leave-region-out geometry does not improve.
- **Status:** rejected. Cosine selected \(k=0.3\) while WSD selected \(k=0.03\), a tenfold schedule dependence, yet leave-region-out RMSE remained \(0.2143/0.1655\) and WSD Regret@1 remained \(0.652\) BPB. The concentration factor is neither stable nor sufficient.

## Structural identification audit

## AE. Parallel reliability network

- **Premise:** groups inside a declared family are parallel supports; the family fails only when every support path fails.
- **Latent state:** normalized retained group exposure (r_g) gives path-failure probability (p_g=\delta/(\delta+r_g)); a proportional-mass-weighted geometric product gives family failure.
- **State transition:** phase exposure is accumulated through the normalized exponential recency kernel before converting each group state to (p_g).
- **Response:** nonnegative family amplitudes map log reliability debt to BPB, alongside literal physical replay.
- **Additional degrees of freedom:** one equivalent-evidence prior and one recency rate, plus family response amplitudes.
- **Single-phase restriction:** tied policies use the same retained exposure in both phase intervals and the same reliability network.
- **Expected StarCoder signature:** either broad or code exposure can rescue its family reliability until the remaining path also fails; the surface should have rounded shortage arms rather than an additive plane.
- **Expected optimism fix:** simultaneous starvation of every support path in a family creates positive reliability debt that surplus in another family cannot cancel.
- **Cheapest falsification:** numerical pathologies, nontransferable recency/prior parameters, or worse leave-region prediction than the additive reliability ablation.
- **Status:** rejected. Frozen Delphi heldout RMSE was (0.02334/0.03799) on Uncheatable/Table-9, with 10/5 optimism errors and Regret@1 (0.02533/0.03197). StarCoder cosine/WSD leave-region-out RMSE was (2.286/0.1667). Parallel substitutability does not transfer and can become numerically pathological.

## AF. Bayesian precision with exponential process loss

- **Premise:** exposure contributes independent information precision while incompatible updates erase existing precision at a first-order rate.
- **Latent state:** (P_g>0) is equivalent evidence precision for group (g).
- **State transition:** (dot P_g=r_g-h_gP_g), solved exactly within each phase.
- **Response:** BPB is a nonnegative log-precision debt (sum_g A_g\log(P_g^{\rm prop}/P_g)) plus literal replay.
- **Additional degrees of freedom:** one forgetting rate and one equivalent-prior precision.
- **Single-phase restriction:** at zero forgetting, precision is total physical exposure and is phase-boundary invariant; tied schedules otherwise compose the same exact transition.
- **Expected StarCoder signature:** late code exposure restores code precision after broad-data interference while physical replay creates the high-code arm.
- **Expected optimism fix:** an omitted bucket retains finite positive uncertainty instead of losing only a bounded additive benefit.
- **Cheapest falsification:** forgetting/prior parameters hit boundaries or fail to transfer across both StarCoder schedules and a multi-bucket panel.
- **Status:** rejected. The shared-law audit chose the grid boundary (h=8), prior (0.01), but Delphi heldout RMSE was (0.03194/0.03984), with 16/19 optimism errors and Regret@1 (0.06837/0.11804). A target-specific screen was also unstable across panels. Linear precision loss is not the missing state law.

## AG. Finite representation capacity gating

- **Premise:** evidence is useful only when the model allocates finite representation capacity to the corresponding family.
- **Latent state:** (z_g\in[0,1]) is the fraction of finite representation capacity allocated to family (g).
- **State transition:** capacity relaxes toward current family mass, (dot z_g=a(q_g-z_g)), exactly within each phase.
- **Response:** capability is total evidence precision times terminal capacity; nonnegative amplitudes map its log debt relative to proportional to BPB, plus physical replay.
- **Additional degrees of freedom:** one adaptation rate and two equivalent-prior floors.
- **Single-phase restriction:** tied policies drive one autonomous relaxation toward a fixed target and refit the same response.
- **Expected StarCoder signature:** a slow capacity state should create order-dependent hysteresis and rotate WSD more strongly than cosine.
- **Expected optimism fix:** a concentrated policy cannot retain full broad capability after reallocating capacity to one family.
- **Cheapest falsification:** adaptation runs to the instantaneous boundary, family capacity is non-identifiable, or heldout optimism worsens on two targets.
- **Status:** rejected. Although fit-panel and StarCoder metrics were competitive, Delphi heldout RMSE was (0.03289/0.05059), with 23/94 optimism errors; the adaptation rate selected the maximum (16) nearly everywhere. Sharing one law across panels still yielded (0.02260/0.02867) heldout RMSE and 12/13 optimism errors. The apparent local gain is a fast-capacity boundary fit that does not transfer.

## AH. Exact finite-subset traversal with retained competence

- **Premise:** Marin simulated epoching materializes a fixed subset and recycles it, so unique coverage is exactly (u_i=\min(E_i,1)), not a Poisson occupancy curve.
- **Latent state:** (u_i\in[0,1]) is traversed subset mass and (s_i\in[0,1]) is retained competence.
- **State transition:** newly traversed subset mass is added once; existing competence decays only under out-of-family updates. Replay is exactly ((E_i-1)_+).
- **Response:** nonnegative family amplitudes map retained-competence deficit relative to proportional to BPB, with a separate literal replay term.
- **Additional degrees of freedom:** one compatible-family forgetting rate and one prior floor.
- **Single-phase restriction:** at zero forgetting the terminal state is exactly (\min(E_i^{(0)}+E_i^{(1)},1)), independent of phase subdivision.
- **Expected StarCoder signature:** a sharp learning knee at one materialized epoch and a linear replay arm after that knee.
- **Expected optimism fix:** repeated tokens beyond one epoch cannot add unique evidence and omitted buckets retain maximal deficit.
- **Cheapest falsification:** the observed swoosh lacks the fixed one-epoch knee or a smooth occupancy law predicts held-out regions better.
- **Status:** rejected at the StarCoder shape gate. Cosine/WSD leave-region-out behavior was poor (fit/leave-region geometry did not recover the swoosh; WSD leave-region regret was extreme). Exact data uniqueness is a sampler invariant, but one traversal is not equivalent to learned competence.

## AI. Reduced gradient-flow capability bowl

- **Premise:** a family representation follows gradient-flow relaxation toward the current mixture, while evaluation loss is locally quadratic around a target-specific optimal representation.
- **Latent state:** (z_f) is the learned representation coordinate for family (f).
- **State transition:** (dot z_f=a(q_f-z_f)), solved exactly within each phase.
- **Response:** (Y=b+\sum_f c_f(z_f-\mu_f)^2) plus replay, with constrained reachable centers and (c_f\ge0).
- **Additional degrees of freedom:** one adaptation rate, one center and curvature per family, and replay amplitudes.
- **Single-phase restriction:** tying phases leaves one exact relaxation toward a constant mixture and refits the same convex bowl.
- **Expected StarCoder signature:** a curved valley whose orientation changes with adaptation rate, with replay lifting high-exposure arms.
- **Expected optimism fix:** moving too far past a capability-specific optimum becomes harmful even before finite-data replay dominates.
- **Cheapest falsification:** fitted centers leave the reachable state region, adaptation hits a boundary, or leave-region minima are misplaced.
- **Status:** rejected. Cosine/WSD selected rates (8/16) at or near the fast boundary; leave-region RMSE was about (0.151/0.104), and fitted centers moved toward degenerate near-zero representations. A convex local bowl does not extrapolate the global surface.

## AJ. Power-law retention survival

- **Premise:** memory hazard decreases with age, so evidence acquired at time (t) survives future out-of-family update mass (A) as ((1+A/\tau)^{-p}), rather than memoryless exponential decay.
- **Latent state:** retained family evidence is the convolution of acquisition with survival (S(A)=(1+A/\tau)^{-p}).
- **State transition:** each exposure increment is added to retained evidence and discounted by the future out-of-family update mass through (S).
- **Response:** normalized family deficit and exact finite-subset replay predict BPB with nonnegative amplitudes.
- **Additional degrees of freedom:** timescale (\tau), survival exponent (p), shortage floor, and shortage exponent.
- **Single-phase restriction:** a tied piecewise-constant policy is invariant to where the phase boundary is drawn because the convolution depends only on future interference mass.
- **Expected StarCoder signature:** long-tail memory should preserve early code exposure more than exponential forgetting while retaining an order-dependent valley.
- **Expected optimism fix:** extreme late concentration cannot erase all early evidence instantly, and omitted families retain explicit deficit.
- **Cheapest falsification:** the kernel selects its memoryless boundary or fails both StarCoder leave-region tests.
- **Status:** rejected. Both StarCoder surfaces selected the most memoryless screened boundary ((\tau,p)=(1,4)). Cosine/WSD leave-region RMSE was (0.1534/0.1629), with 23/18 optimism errors above (0.05). A long-tail kernel does not recover the Nike-swoosh geometry.

## AK. Kalman--Bucy uncertainty under interference

- **Premise:** in-family evidence reduces posterior uncertainty while incompatible updates inject process variance.
- **Latent state:** (V_g>0) is posterior variance for capability group (g).
- **State transition:** (dot V_g=q_g-r_gV_g^2), solved exactly per phase.
- **Response:** nonnegative amplitudes map log-variance debt relative to proportional to BPB, plus physical replay.
- **Additional degrees of freedom:** one prior variance and one process-variance rate.
- **Units and limits:** (V) is inverse equivalent passes; (q) has those units per normalized training time. At (q=0), (V(t)=V_0/(1+rV_0t)). Tied schedules compose exactly.
- **Single-phase restriction:** tie the phase inputs and apply the same autonomous Riccati transition over the uninterrupted schedule; the semigroup law makes an artificial phase boundary irrelevant.
- **Expected StarCoder signature:** uncertainty falls under in-domain exposure and rises under incompatible updates, producing recency-sensitive shortage arms.
- **Expected optimism fix:** poorly covered buckets carry irreducible uncertainty that overtraining another bucket cannot cancel.
- **Cheapest falsification:** variance parameters hit search boundaries, calibration collapses, or schedule transfer fails.
- **Status:** rejected. WSD selected both variance parameters at the maximum (10). Cosine/WSD leave-region RMSE was (0.2368/0.2111), WSD Regret@1 was (0.1236), and the calibration slope fell to (0.228). Posterior uncertainty is not the missing out-of-support cost.

## AL. Fast memory with slow consolidation

- **Premise:** a vulnerable fast competence pool acquires current evidence, while a slow pool consolidates it and is protected from direct interference.
- **Latent state:** (f_g\in[0,1]) is fast competence and (s_g\in[0,1]) is consolidated slow competence.
- **State transition:** (dot f_g=a r_g(1-f_g)-h o_gf_g), (dot s_g=k(f_g-s_g)).
- **Response:** terminal capability is the prespecified convex mixture ((1-\omega)f_g+\omega s_g), followed by family capability debt and replay.
- **Additional degrees of freedom:** acquisition, forgetting, consolidation, and slow-pool share; response amplitudes remain nonnegative.
- **Single-phase restriction:** the autonomous ODE semigroup makes tied schedules invariant to artificial phase subdivision.
- **Expected StarCoder signature:** early code exposure can survive through the slow state while late code controls the fast state, yielding two recency scales.
- **Expected optimism fix:** a narrow late mixture cannot instantly destroy consolidated broad competence, while absent families still lose fast competence.
- **Cheapest falsification:** the slow state collapses to zero/instantaneous consolidation or fails to improve both schedule surfaces.
- **Status:** rejected. Both StarCoder surfaces selected the same interior/edge combination ((a,h,k,\omega)=(2,2,8,0.25)), so the law was reproducible, but cosine/WSD leave-region RMSE remained (0.1314/0.1649), with 64/13 optimism errors. A second memory timescale is identifiable but insufficient.

## Shared transition-law audit

## AO. Learned-state-gated family competition

- **Premise:** gradient conflict requires both a competing data stream and an already learned competing representation; raw mixture concentration alone is not sufficient.
- **Transition:** bounded group competence follows \(\dot z_g=a r_g(1-z_g)-c z_g\sum_{f\ne f(g)}q_fz_f\), where \(r_g\) is physical group-exposure rate, \(q_f\) is current family mass, and \(z_f\) is proportional-mass-weighted family competence.
- **Response:** nonnegative target-specific amplitudes map competence debt relative to the proportional policy and physical replay to BPB.
- **Additional degrees of freedom:** one acquisition rate and one competition rate; \(c=0\) is the exact independent-learning ablation.
- **Single-phase restriction:** a tied policy drives one autonomous ODE, invariant to an artificial phase boundary.
- **Expected StarCoder signature:** the rare-data arm should become harmful only after its representation has been learned, producing an interior Nike-swoosh without a direct output penalty.
- **Expected optimism fix:** extreme policies lose useful competence through learned competition before the response link, rather than receiving a post-hoc extrapolation correction.
- **Cheapest falsification:** incompatible parameters across the two schedules, poor leave-region prediction, or selection of \(c=0\).
- **Status:** rejected at the two-domain shape gate. Cosine/WSD selected acquisition at the screened maximum \(a=8\) while competition changed from \(c=2\) to \(c=4\). Leave-region RMSE was \(0.0998/0.1207\), with selected-point optimism \(0.1212/0.1104\) BPB and worst optimism \(0.2500/0.4697\). Learned competitors are a coherent state variable, but this transition still does not identify the Nike-swoosh geometry or transfer across schedules.

## AM. Concentration-driven cross-family displacement

- **Premise:** concentrated updates align gradients with a narrow family and displace representations needed by other families.
- **Transition:** each group acquires competence from its own exposure while excess family Herfindahl concentration creates a rank-one, out-of-family displacement hazard.
- **Response:** nonnegative terminal capability deficits and physical replay amplitudes map the displaced competence state to BPB.
- **Additional degrees of freedom:** one acquisition rate and one displacement rate; zero displacement is the exact nested ablation.
- **Single-phase restriction:** a tied policy applies one autonomous acquisition-displacement law throughout training.
- **Expected StarCoder signature:** extreme specialization raises the high-target arm while preserving an interior valley; schedule changes alter the accumulated displacement.
- **Expected optimism fix:** concentrated optima lose unsupported cross-family competence rather than receiving an output-level surcharge.
- **Cheapest falsification:** leave-region prediction remains poor or the selected rates differ materially between the two schedules.
- **Status:** rejected. Cosine/WSD selected acquisition rates \(2/8\) and both drove displacement to the screened maximum \(10\). Leave-region RMSE remained \(0.1537/0.1296\); WSD Regret@1 was \(0.1647\), and its selected optimum was \(0.1585\) BPB too optimistic. A concentration hazard does not recover the two-domain geometry and is schedule-unstable.

## AN. Diversity-gated global acquisition

- **Premise:** broad representation learning requires concurrent support from every foundation family; family starvation reduces the efficiency of all useful acquisition, while surplus cannot make acquisition more than fully efficient.
- **Transition:** in phase \(t\), let \(r_f^{(t)}=w_f^{(t)}/p_f\). Acquisition efficiency is \(h_t=\min\{1,\exp[\sum_f p_f\log((r_f^{(t)}+\delta)/(1+\delta))]\}^{\kappa}\), and effective bucket exposure is \(\sum_t h_t e_i^{(t)}\).
- **Response:** the existing nonnegative group shortage response and physical replay are evaluated on gated exposure. \(\kappa=0\) is exactly the physical-exposure ablation.
- **Additional degrees of freedom:** one equivalent-support floor \(\delta\) and one dimensionless sensitivity \(\kappa\); shortage curvature is selected on fit-panel OOF.
- **Single-phase restriction:** tied phases have identical \(h_t\), so the model is invariant to an artificial phase boundary.
- **Expected StarCoder signature:** broad-data starvation raises the surface sharply, while moderate rare-data enrichment remains efficient.
- **Expected optimism fix:** mixtures that jointly starve many proportional families cannot retain full global learning efficiency.
- **Cheapest falsification:** \(\kappa=0\) is selected, StarCoder leave-region geometry remains poor, or a positive gate worsens 3e18 regret/optimism.
- **Status:** rejected at the two-domain shape gate. Cosine/WSD selected incompatible gate settings: \((\delta,\kappa)=(0.3,2.0)\) versus \((0.03,0.25)\). Leave-region RMSE was \(0.2137/0.1662\); WSD Regret@1 was \(0.6520\), and its selected point was \(0.6371\) BPB too optimistic. Global acquisition efficiency prices starvation but does not identify the phase-order surface.

Nonlinear transition parameters were also selected jointly across all seven fit panels by a prespecified minimax panel-relative OOF criterion, while response amplitudes and ridge remained target specific. The shared retained-state law cost only 4.7% mean OOF RMSE, yet produced Delphi heldout RMSE (0.02926/0.03109), 14/10 optimism errors, and worst optimism (0.180/0.216). Shared posterior-precision and capacity laws were worse. Independent hyperparameter selection is therefore not the primary cause of the optimum-region failure.

## Structural identification audit

The fit panel itself does not uniquely identify Table-9 extrapolation. Freezing the model set by the 5% OOF-RMSE tolerance before looking at heldouts leaves four statistically competitive Table-9 forms. Across the 259 policy-matched heldouts their median prediction range is \(0.0225\) BPB, the 90th percentile is \(0.0675\), and the maximum is \(0.2000\); 51 heldouts have a model envelope wider than \(0.05\). Their selected observed Regret@1 ranges from \(0.00476\) to \(0.11224\). For Uncheatable, the two OOF-equivalent forms agree much more closely (median range \(0.00116\), maximum \(0.0332\)) but share the same extreme-policy optimism. This separates two problems: Table-9 is underidentified by the current interventions, while Uncheatable exposes a more coherent missing mechanism.

## Reopening rule

A blocked route is reopened only with a new state variable, invariant, transition law, or response mechanism. A different ridge value, output affine transform, interaction dictionary, or deployment trust region is not a new mechanism.
