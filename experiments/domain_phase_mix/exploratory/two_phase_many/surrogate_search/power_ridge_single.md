# Power-Ridge: A Surrogate Model for Many-Domain Data Mixture Prediction

## Summary

We introduce **Power-Ridge**, a surrogate model for predicting held-out perplexity
from pretraining data mixture weights in a 39-domain, 2-phase setting. The model is
a single equation:

$$\hat{y} = c + \boldsymbol{\beta}^{\top} \begin{bmatrix} \mathbf{w} \\ \sqrt{\mathbf{w}} \end{bmatrix}, \qquad \mathbf{w} = \alpha \, \mathbf{w}_0 + (1 - \alpha) \, \mathbf{w}_1$$

with $\alpha = 0.37$, $\boldsymbol{\beta} \in \mathbb{R}^{2M}$ estimated via Ridge
regression, and $\sqrt{\cdot}$ applied element-wise. On `eval/uncheatable_eval/bpb`
with 241 proxy runs:

| | $R^2$ | $\rho_s$ | Regret@1 | $P$ |
|---|---:|---:|---:|---:|
| **Power-Ridge** | $\mathbf{0.789 \pm 0.011}$ | $\mathbf{0.887 \pm 0.006}$ | $\mathbf{0.002}$ | $78$ |
| Olmix loglinear | $0.440$ | $0.733$ | $0.009$ | $79$ |
| DS-RE-CEQ | $-2887$ | $0.444$ | $0.012$ | $162$ |

The core idea is a two-term power-law feature expansion of mixture weights, motivated
by overfitting scaling laws from the Finetuner's Fallacy (DatologyAI, 2026).


## Motivation

### Why DS-RE-CEQ fails at 39 domains

DS-RE-CEQ was designed for the 2-domain, 3-phase StarCoder setting where it achieves
$R^2 = 0.97$ with 19 parameters. Its forward pass is:

$$\hat{y} = c_0 - A \cdot \text{CES}_\rho(\mathbf{S}; \mathbf{a}) + B \cdot P$$

where the per-domain retained state $S_d = \sum_k \pi_d^{(k)} r_{k,d} \, z_{k,d}$
depends on signal extraction $z_{k,d} = \log(1 + \varphi_d \text{Cum}_{k,d} + E_{k,d}) - \log(1 + \varphi_d \text{Cum}_{k,d})$,
retention $r_{k,d} = \exp\bigl(-\sum_{j>k} g_j \lambda_{j,d} (1 - w_{j,d})\bigr)$,
and the penalty $P = [\text{softplus}(\sum_{d \in \mathcal{S}} X_d - \tau)]^2$.

At $M = 39$ domains and $N = 2$ phases this requires 162 parameters:

| Component | Count | Role |
|-----------|------:|------|
| $a_d$ (CES weights) | 38 | Domain importance |
| $\pi_d^{(k)}$ (phase importance) | 39 | Per-domain phase weighting |
| $\lambda_{k,d}$ (interference) | 39 | Per-domain forgetting rate |
| $\varphi_d$ (satiety) | 39 | Per-domain diminishing returns |
| $g_k, \rho, \tau, c_0, A, B$ | 7 | Global |

With only 241 data points, these per-domain parameters are unidentifiable. The model
catastrophically overfits ($R^2 = -2887$). Ablation confirms the diagnosis:

| Ablation | $R^2$ | What was removed |
|----------|------:|------------------|
| Full DS-RE-CEQ | $-2887$ | — |
| Drop per-domain $\pi_d$ | $+0.21$ | 39 params removed |
| Drop interference | $-4346$ | 40 params removed, but Sp improves to 0.70 |
| DS-RE-CEQ-ST(lite) | $+0.51$ | Drop $\pi_d$, $\varphi_d$, $g_k$; 78 params removed |

Removing per-domain phase importance $\pi_d$ alone recovers positive $R^2$.
The lite variant with 84 parameters is the only DS-RE-CEQ configuration that
generalizes, but it still underperforms Ridge-Raw ($R^2 = 0.53$).

### What DS-RE-CEQ gets right

The signal extraction $z_d = \log(1 + E_d)$ models diminishing returns: the first
tokens from a domain help more than additional tokens. This is the right inductive
bias. The failure is not in this idea but in implementing it through CES aggregation
over 39 log-signals with per-domain satiety and interference — too many parameters
for the data.

### The scaling law connection

The Finetuner's Fallacy (arXiv:2603.16177) derives an overfitting scaling law that
decomposes held-out domain loss into a training-loss term and a train–test gap:

$$\mathcal{L}_{\text{test}}(T, \delta) = A_{\text{train}} \cdot T^{b_{\text{train}}(\delta)} + A_{\text{gap}}(\delta) \cdot T^{b_{\text{gap}}(\delta)}$$

where $T$ is total tokens, $\delta$ is the domain mixture fraction,
$b_{\text{train}} < 0$ (training loss decreases as a power law of data seen), and
$0 < b_{\text{gap}} < 1$ (the overfitting gap increases sublinearly with repeated
exposure). Key properties:

- The training-loss exponent $b_{\text{train}}(\delta) = \delta \, b_s + (1-\delta) \, b_g$
  interpolates between specialized and general exponents.
- The gap amplitude $A_{\text{gap}}(\delta) = \alpha_1 \, \delta^{\alpha_2} \exp(\alpha_3 \delta)$
  grows with mixture fraction — more weight on a domain means more repetitions and
  more overfitting.
- The gap exponent satisfies $0 < b_{\text{gap}} < 1$: overfitting is **sublinear**,
  not quadratic as DS-RE-CEQ's $[\text{softplus}(\cdot)]^2$ assumes.

In our setting $T$ is fixed (1.2B tokens for all runs). Per-domain effective training
tokens are $T_d = \delta_d \cdot T \propto w_d$. Substituting into the scaling law,
each domain's contribution to held-out loss is:

$$f_d(w_d) \;\sim\; a_d \, w_d^{b_{\text{train}}} + c_d \, w_d^{b_{\text{gap}}}$$

Both exponents are fractional. A two-term basis $\{w_d,\; w_d^{1/2}\}$ can
approximate this for a range of exponents near $0.5$. Ridge regression finds the
optimal linear combination over all domains simultaneously.


## Model

### Definition

Given $N = 2$ training phases with mixture weight vectors
$\mathbf{w}_0, \mathbf{w}_1 \in \Delta^{M-1}$, define the phase-weighted mixture:

$$\mathbf{w} = \alpha \, \mathbf{w}_0 + (1 - \alpha) \, \mathbf{w}_1, \qquad \alpha = 0.37$$

The prediction is:

$$\hat{y} = c + \sum_{d=1}^{M} \Bigl[\beta_d^{(\text{lin})} \, w_d + \beta_d^{(\text{pow})} \sqrt{w_d}\,\Bigr]$$

or equivalently $\hat{y} = c + \boldsymbol{\beta}^\top \mathbf{x}$ where
$\mathbf{x} = [\mathbf{w};\, \sqrt{\mathbf{w}}\,] \in \mathbb{R}^{2M}$.

The coefficients $c \in \mathbb{R}$ and $\boldsymbol{\beta} \in \mathbb{R}^{2M}$
are estimated by Ridge regression:

$$\min_{c, \boldsymbol{\beta}} \sum_{r=1}^{R} \bigl(\hat{y}_r - y_r\bigr)^2 + \lambda \|\boldsymbol{\beta}\|_2^2$$

with $\lambda$ chosen by leave-one-out cross-validation (closed-form via the
RidgeCV identity). The model has $2M + 1 = 79$ nominal parameters, with effective
degrees of freedom controlled by $\lambda$.

### Per-domain response curve

For domain $d$, the model predicts:

$$f_d(w) = \beta_d^{(\text{lin})} \, w + \beta_d^{(\text{pow})} \, \sqrt{w}$$

This is a two-term power series in $w^{1/2}$:

- At small $w$: the $\sqrt{w}$ term dominates — steep initial gain from the first
  tokens allocated to this domain.
- At large $w$: the linear term dominates — slower marginal gain, or a penalty if
  over-concentrating on one domain crowds out others (since $\sum_d w_d = 1$).

The curve is concave when $\beta_d^{(\text{pow})} < 0$ and $\beta_d^{(\text{lin})} > 0$
(the common case for helpful small domains), matching the diminishing-returns shape
that DS-RE-CEQ modeled with $\log(1 + E_d)$.

### Why $\sqrt{w}$ and not $\log w$

| Features | $R^2$ | $\Delta$ vs raw |
|----------|------:|------:|
| $w$ only | $0.532$ | — |
| $w + \sqrt{w}$ | $\mathbf{0.789}$ | $+0.257$ |
| $w + \log w$ | $0.788$ | $+0.256$ |
| $w + w^{1/3}$ | $0.653$ | $+0.121$ |
| $w + w^{2/3}$ | $0.680$ | $+0.148$ |

$\sqrt{w}$ and $\log w$ are equally effective — both capture the concave response.
We prefer $\sqrt{w}$ because:

1. It is non-singular at $w = 0$ (important since many domains have near-zero weight
   in some runs).
2. It has a direct interpretation as a power-law basis function with exponent
   $\gamma = 0.5$, connecting to the Finetuner's Fallacy exponents.
3. The two-term expansion $\{w, \sqrt{w}\}$ approximates $w^b$ for any $b$ via
   Taylor expansion around $b = 0.5$.

### Phase weighting

The optimal $\alpha = 0.37$ means
$\mathbf{w} = 0.37 \, \mathbf{w}_0 + 0.63 \, \mathbf{w}_1$. Both phases contribute
roughly equally, with slight emphasis on phase 1 (the 20% annealing phase with
decaying learning rate). This makes sense for perplexity: the model's loss on
held-out text integrates over what it learned during the full training trajectory,
with modest recency bias toward the annealing phase.

This differs from downstream task prediction (e.g., MMLU), where phase 1 dominates
($\alpha \approx 0.10$). Knowledge-intensive benchmarks depend more heavily on the
final model state.


## Fitted coefficients

### Feature importance decomposition

The $\sqrt{w}$ features carry 72% of total feature importance (measured as
$\sum_d |\beta_d \cdot \bar{x}_d|$) and explain $R^2 = 0.56$ alone. The raw $w$
features explain $R^2 = 0.14$ alone. Combined: $R^2 = 0.91$ (full-data fit).

The $\sqrt{w}$ transform is doing most of the work. The linear term provides a
correction that captures domain-specific deviations from the $\sqrt{w}$ scaling.

### Domain group behavior

| Group | $\overline{\beta}^{(\text{lin})}$ | $\overline{\beta}^{(\text{pow})}$ | Interpretation |
|-------|---:|---:|---|
| Dolmino-synth | $+0.320$ | $-0.117$ | Helpful but saturates quickly |
| Dolma3-other | $+0.045$ | $-0.070$ | Moderately helpful, moderate saturation |
| Dolmino-curated | $-0.040$ | $-0.043$ | Slight dilution, slight saturation |
| CC-high quality | $-0.074$ | $-0.006$ | Mild dilution, $\approx$ linear |
| CC-low quality | $-0.066$ | $+0.010$ | Dilution, $\approx$ linear |

**Synthetic data** (synth\_code, synth\_instruction, synth\_math, synth\_qa,
synth\_thinking) has strongly positive $\beta^{(\text{lin})}$ and strongly negative
$\beta^{(\text{pow})}$. These are small domains (18B–40B tokens) repeated
100–280$\times$ at the target budget. The diminishing returns from epoching are the
dominant nonlinearity, and $\sqrt{w}$ captures them.

**CC domains** have $\beta^{(\text{pow})} \approx 0$: they are large enough
(40B–540B tokens each) that the sweep never epochs them significantly. Their response
is approximately linear in $w$.

This pattern connects directly to the Finetuner's Fallacy: the overfitting gap
$A_{\text{gap}}(\delta) \cdot T^{b_{\text{gap}}}$ is largest for small,
highly-repeated domains — exactly where $\beta^{(\text{pow})}$ is most negative.

### Top domains by marginal effect

Net effect of increasing domain weight by $\Delta w = +0.01$ (evaluated at the mean
weight $\bar{w} = 0.026$):

| Domain | $\Delta \hat{y}$ | Character |
|--------|---:|---|
| dolma3\_stack\_edu | $-0.0058$ | Strong help, strong saturation |
| dolmino\_stack\_edu\_fim | $-0.0052$ | Strong help, strong saturation |
| dolmino\_olmocr\_pdfs\_hq | $-0.0021$ | Moderate help |
| dolma3\_cc/literature\_high | $-0.0020$ | Moderate help, linear |
| dolma3\_cc/sci\_math\_tech\_high | $-0.0019$ | Moderate help, linear |
| dolmino\_synth\_qa | $-0.0015$ | Moderate help, saturating |
| dolma3\_arxiv | $-0.0012$ | Moderate help, saturating |
| dolmino\_stem\_heavy\_crawl | $+0.0017$ | Hurts (dilutes helpful domains) |


## Evaluation

### Many-domain `uncheatable_eval/bpb` (R=241, M=39, N=2)

10 random fold seeds, 5-fold each (single-seed for CES/DS-RE-CEQ variants due to
cost):

| Model | $R^2$ | $\rho_s$ | Regret@1 | $P$ |
|-------|---:|---:|---:|---:|
| **Power-Ridge** ($\alpha = 0.37$) | $\mathbf{0.789 \pm 0.011}$ | $\mathbf{0.887 \pm 0.006}$ | $\mathbf{0.002}$ | $79$ |
| Lean-DSRE-V2 | $0.558 \pm 0.015$ | $0.732 \pm 0.013$ | $0.007$ | $32$ |
| Ridge-Raw (no $\sqrt{w}$) | $0.532 \pm 0.007$ | $0.775 \pm 0.005$ | $0.009$ | $39$ |
| DS-RE-CEQ-ST(lite) | $0.509$ | $0.704$ | $0.007$ | $84$ |
| CES | $0.470$ | $0.697$ | $0.006$ | $81$ |
| Olmix loglinear | $0.440 \pm 0.010$ | $0.733 \pm 0.006$ | $0.009$ | $79$ |
| CES-Overfit | $-12.5$ | $0.029$ | $0.020$ | $159$ |
| DS-RE-CEQ | $-2887$ | $0.444$ | $0.012$ | $162$ |

At $M = 39$, Power-Ridge dominates all nonlinear models. Plain CES ($R^2 = 0.47$,
$P = 81$) has the same parameter budget as Power-Ridge ($P = 79$) but much worse
fit — the CES aggregation function is harder to optimize than Ridge, and its
$\log(1 + E)$ signal extraction is less effective than $\sqrt{w}$ for this data.
Adding the overfit penalty (CES-Overfit, $P = 159$) causes the same catastrophic
overfitting as DS-RE-CEQ.

### Two-phase StarCoder (R=116, M=2, N=2)

| Model | $R^2$ | $\rho_s$ | Regret@1 | $P$ |
|-------|---:|---:|---:|---:|
| CES-Overfit | $\mathbf{0.983}$ | $0.941$ | $0.002$ | $9$ |
| CES | $0.917$ | $\mathbf{0.940}$ | $\mathbf{0.001}$ | $7$ |
| Power-Ridge ($\alpha = 0.2$) | $0.905$ | $0.892$ | $0.011$ | $5$ |

At $M = 2$ the CES family is well-identified: 7–9 parameters for 116 points. The
overfit penalty helps because the StarCoder domain (217B tokens vs 5.7T Nemotron)
has a strong U-shaped epoch–loss response. Power-Ridge still captures most of the
signal ($R^2 = 0.91$) with fewer parameters by approximating the U-shape through
$\sqrt{w}$ and $\sqrt{1-w}$.

### Three-phase StarCoder (R=160, M=2, N=3)

| Model | $R^2$ | $\rho_s$ | Regret@1 | $P$ |
|-------|---:|---:|---:|---:|
| DS-RE-CEQ | $\mathbf{0.972}$ | $\mathbf{0.965}$ | $\mathbf{0.003}$ | $19$ |
| Power-Ridge ($\alpha = 0.0$) | $0.873$ | $0.837$ | $0.053$ | $5$ |
| CES-Overfit | $0.822$ | $0.910$ | $0.014$ | $12$ |
| CES | $0.700$ | $0.867$ | $0.005$ | $9$ |

DS-RE-CEQ's phase-specific interference model (satiety, retention, conflict gates)
earns its 19 parameters here: three sequential phases with cosine LR create genuine
inter-phase dynamics that neither CES nor Power-Ridge can capture. Power-Ridge's
phase-mixing reduces three phases to a single weighted average, losing the sequential
structure. CES ignores phases entirely.

### Cross-dataset summary

| Setting | Best model | $R^2$ | Why |
|---------|-----------|------:|-----|
| $M = 39$, $N = 2$ | Power-Ridge | $0.79$ | Per-domain params unidentifiable; $\sqrt{w}$ + Ridge is efficient |
| $M = 2$, $N = 2$ | CES-Overfit | $0.98$ | Few params, U-shaped overfit well-modeled |
| $M = 2$, $N = 3$ | DS-RE-CEQ | $0.97$ | Sequential phase dynamics identifiable at low $M$ |

Power-Ridge scales to many domains where mechanistic models cannot. Mechanistic
models (CES-Overfit, DS-RE-CEQ) excel when $M$ is small enough for per-domain
parameters to be identifiable and the phase structure is rich enough to reward
sequential modeling.

### Noise ceiling

- Full-data fit: $R^2 = 0.91$, residual std $= 0.0057$.
- OOF $R^2 = 0.79$ implies an overfitting gap of $0.12$.
- Theoretical max OOF $R^2 \approx 0.85$–$0.90$ (set by irreducible stochastic
  training noise).

Power-Ridge is within ${\sim}0.06$–$0.11$ of the ceiling.


## Relationship to DS-RE-CEQ

Power-Ridge is a linearized, regularized simplification of DS-RE-CEQ. The
correspondence:

| DS-RE-CEQ | Power-Ridge |
|-----------|-------------|
| $z_d = \log(1 + E_d)$ | $\beta_d^{(\text{pow})} \sqrt{w_d}$ |
| $\text{CES}_\rho(\mathbf{S};\, \mathbf{a})$ | $\sum_d \beta_d \, x_d$ (Ridge) |
| Per-domain $a_d$ | Per-domain $\beta_d$ (Ridge-regularized) |
| Per-domain $\varphi_d$ (satiety) | Implicit in $\sqrt{w}$ nonlinearity |
| Per-domain $\pi_d^{(k)}$ (phase importance) | Shared $\alpha$ |
| Per-domain $\lambda_d$ (interference) | Dropped (2-phase: unidentifiable) |
| $[\text{softplus}(\sum X_d - \tau)]^2$ | Dropped ($\sqrt{w}$ subsumes) |
| L-BFGS-B, 8 restarts, ${\sim}600$s | Closed-form, $< 0.01$s |

DS-RE-CEQ entangles two concerns: (1) concave per-domain response and (2)
cross-domain aggregation. Power-Ridge separates them:

1. **Concave response**: Fixed $\sqrt{w}$ feature — zero learnable nonlinear
   parameters.
2. **Aggregation**: Ridge linear combination with $\ell_2$ shrinkage, automatically
   zeroing irrelevant domains.

This is why 78 Ridge-regularized parameters outperform 162 freely-optimized ones.


## Limitations

1. **Additive model.** Power-Ridge cannot capture domain interactions (e.g., synergy
   between code and math). The CES aggregation in DS-RE-CEQ can in principle model
   complementarity ($\rho < 0$) or substitutability ($\rho > 0$), but cannot fit
   these with 241 points at $M = 39$.

2. **Fixed power exponent.** The $\gamma = 0.5$ exponent is a hyperparameter. It
   works well empirically but the true exponent may vary by domain or dataset. A
   learned per-group $\gamma_g$ could improve the model if more data were available.

3. **Phase mixing coefficient.** The optimal $\alpha = 0.37$ is tuned on this
   dataset. Different phase schedules (e.g., 3-phase, different boundary fractions)
   would require re-tuning.

4. **Metric-specific.** Power-Ridge predicts perplexity well ($R^2 = 0.79$) but not
   MMLU ($R^2 = 0.08$). MMLU at 60M scale has a noise-to-signal ratio $> 1$,
   requiring larger proxy models for mixture optimization on knowledge-intensive
   benchmarks.
