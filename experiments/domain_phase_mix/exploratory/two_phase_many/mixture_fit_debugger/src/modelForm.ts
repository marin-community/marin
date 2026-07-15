import type { FitDetail, ModelId, PolicyClass } from "./types";

interface FormulaLayer {
  label: string;
  title: string;
  tex: string;
  explanation: string;
}

interface FormulaChip {
  label: string;
  tex: string;
}

export interface ModelForm {
  topLevelTex: string;
  topLevelExplanation: string;
  layers: FormulaLayer[];
  chips: FormulaChip[];
}

function compactNumber(value: number): string {
  const magnitude = Math.abs(value);
  if (magnitude !== 0 && (magnitude < 1e-3 || magnitude >= 1e4)) return value.toExponential(2);
  return Number(value.toPrecision(4)).toString();
}

function parameterValue(detail: FitDetail, ...keys: string[]): number | null {
  for (const key of keys) {
    const parameter = detail.parameters.find((candidate) => candidate.key === key);
    if (parameter?.value !== null && parameter?.value !== undefined) return parameter.value;
  }
  return null;
}

function fittedChip(detail: FitDetail, label: string, symbol: string, ...keys: string[]): FormulaChip | null {
  const value = parameterValue(detail, ...keys);
  if (value === null) return null;
  return { label, tex: String.raw`${symbol}=${compactNumber(value)}` };
}

function commonExposureLayer(): FormulaLayer {
  return {
    label: "01 / Inputs",
    title: "Realized phase exposure",
    tex: String.raw`e_i^{(t)}=c_i^{(t)}w_i^{(t)},\qquad t\in\{0,1\}`,
    explanation: "A phase weight becomes simulated epochs through that bucket's phase-specific epoch factor.",
  };
}

function aggregateExposureLayer(): FormulaLayer {
  return {
    label: "01 / Inputs",
    title: "Total realized exposure",
    tex: String.raw`e_i=e_i^{(0)}+e_i^{(1)}=\left(c_i^{(0)}+c_i^{(1)}\right)w_i`,
    explanation: "The one-phase policy ties both phase weights, so only total simulated epochs remain as a degree of freedom.",
  };
}


function baselineCoordinateLayer(policyClass: PolicyClass): FormulaLayer {
  if (policyClass === "single_phase") {
    return {
      label: "01 / Policy coordinate",
      title: "Phase-collapsed mixture",
      tex: String.raw`x(w)=\bar w=\alpha_0w^{(0)}+\alpha_1w^{(1)}`,
      explanation: "The policy ablation predicts from aggregate token shares, so off-policy schedules are collapsed before scoring.",
    };
  }
  return {
    label: "01 / Policy coordinate",
    title: "Independent phase weights",
    tex: String.raw`x(w)=\operatorname{vec}\!\left(w^{(0)},w^{(1)}\right)`,
    explanation: "Early and late mixture weights enter as separate coordinates without exposure or grouping features.",
  };
}

function linearForm(policyClass: PolicyClass): ModelForm {
  return {
    topLevelTex: String.raw`\widehat Y_b(w)=b_0+\beta^\top x(w)`,
    topLevelExplanation: "A transparent affine baseline tests how much of the response is explained without curvature or mechanistic structure.",
    layers: [
      baselineCoordinateLayer(policyClass),
      {
        label: "02 / Response",
        title: "Affine policy response",
        tex: String.raw`\widehat Y_b=b_0+\sum_k\beta_kx_k`,
        explanation: "Centered ordinary least squares selects the minimum-norm coefficient representation on the simplex.",
      },
    ],
    chips: [],
  };
}

function olmixLoglinearForm(detail: FitDetail, policyClass: PolicyClass): ModelForm {
  const floor = fittedChip(detail, "additive floor", String.raw`c`, "c");
  return {
    topLevelTex: String.raw`\widehat Y_b(w)=c+\exp\!\left(\beta^\top x(w)\right),\qquad c>0`,
    topLevelExplanation: "The OLMix baseline fits one positive exponential ridge independently for the selected benchmark target.",
    layers: [
      baselineCoordinateLayer(policyClass),
      {
        label: "02 / Response",
        title: "Positive log-linear law",
        tex: String.raw`c=e^{s_c},\qquad \widehat Y_b=e^{s_c}+e^{\beta^\top x}`,
        explanation: "The additive floor and exponential response keep predictions positive while allowing nonlinear variation along one ridge.",
      },
      {
        label: "03 / Fit",
        title: "Robust target loss",
        tex: String.raw`\min_{s_c,\beta}\sum_j\operatorname{Huber}_{0.02}\!\left(\widehat Y_b(w_j)-Y_b(w_j)\right)`,
        explanation: "The repository's frozen OLMix procedure uses deterministic multistart L-BFGS-B; proposal-time KL regularization is separate.",
      },
    ],
    chips: floor ? [floor] : [],
  };
}

function dspForm(
  detail: FitDetail,
  effectiveExposure: boolean,
  geometry: boolean,
  policyClass: PolicyClass,
): ModelForm {
  if (policyClass === "single_phase") {
    const layers = [
      aggregateExposureLayer(),
      {
        label: "02 / Bucket response",
        title: "Saturation and overexposure",
        tex: String.raw`\begin{array}{c}S_i=1-e^{-\rho_i e_i}\\[2pt]P_i=\operatorname{softplus}\!\left(\log(1+e_i)-\tau_i\right)^2\end{array}`,
        explanation: "The phase multiplier disappears; nonnegative amplitudes reward useful exposure and penalize repetition.",
      },
    ];
    const chips: FormulaChip[] = [];
    if (geometry) {
      layers.push({
        label: "03 / Global geometry",
        title: "Aggregate concentration",
        tex: String.raw`G(w)=\theta_{\mathrm{agg}}\lVert w\rVert_2^2`,
        explanation: "Phase divergence is identically zero and late concentration duplicates aggregate concentration, leaving one global term.",
      });
      const chip = fittedChip(detail, "aggregate concentration", String.raw`\theta_{\mathrm{agg}}`, "geometry:aggregate_hhi");
      if (chip) chips.push(chip);
    }
    return {
      topLevelTex: geometry
        ? String.raw`\widehat Y_b(w)=b_0-\sum_i a_iS_i(w)+\sum_i p_iP_i(w)+G(w)`
        : String.raw`\widehat Y_b(w)=b_0-\sum_i a_iS_i(w)+\sum_i p_iP_i(w)`,
      topLevelExplanation: "This is the phase-ablated DSP: the policy and response depend only on aggregate exposure.",
      layers,
      chips,
    };
  }
  const gammaChip = fittedChip(
    detail,
    effectiveExposure ? "late-epoch value" : "late-benefit premium",
    String.raw`\gamma`,
    "gamma",
  );
  const chips = [gammaChip].filter((chip): chip is FormulaChip => chip !== null);
  const topLevelTex = geometry
    ? String.raw`\widehat Y_b(w)=b_0-\sum_i a_iS_i(w)+\sum_i p_iP_i(w)+G(w)`
    : String.raw`\widehat Y_b(w)=b_0-\sum_i a_iS_i(w)+\sum_i p_iP_i(w)`;
  const responseLayer: FormulaLayer = effectiveExposure
    ? {
        label: "02 / Exposure link",
        title: "Shared effective exposure",
        tex: String.raw`z_i=e_i^{(0)}+\gamma e_i^{(1)}`,
        explanation: "The same late-phase multiplier enters useful saturation and the repetition penalty.",
      }
    : {
        label: "02 / Phase link",
        title: "Raw exposure plus late benefit",
        tex: String.raw`z_i=e_i^{(0)}+e_i^{(1)},\qquad r_i=\frac{e_i^{(1)}}{z_i+\varepsilon}`,
        explanation: "Total exposure stays literal; phase 1 changes only the useful-data benefit through its exposure share.",
      };
  const primitives: FormulaLayer = {
    label: "03 / Bucket response",
    title: "Saturation and overexposure",
    tex: effectiveExposure
      ? String.raw`\begin{array}{c}S_i=1-e^{-\rho_i z_i}\\[2pt]P_i=\operatorname{softplus}\!\left(\log(1+z_i)-\tau_i\right)^2\end{array}`
      : String.raw`\begin{array}{c}S_i=(1+\gamma r_i)\left(1-e^{-\rho_i z_i}\right)\\[2pt]P_i=\operatorname{softplus}\!\left(\log(1+z_i)-\tau_i\right)^2\end{array}`,
    explanation: "Nonnegative amplitudes reward useful exposure and charge bucket-specific repetition beyond a learned onset.",
  };
  const layers = [commonExposureLayer(), responseLayer, primitives];
  if (geometry) {
    layers.push({
      label: "04 / Global geometry",
      title: "Schedule-level correction",
      tex: String.raw`\begin{array}{c}G(w)=\theta_{\mathrm{TV}}\,\frac12\left\lVert w^{(0)}-w^{(1)}\right\rVert_1+\theta_{\mathrm{agg}}\lVert\bar w\rVert_2^2+\theta_1\lVert w^{(1)}\rVert_2^2\\[2pt]\bar w=\alpha_0w^{(0)}+\alpha_1w^{(1)}\end{array}`,
      explanation: "Three nonnegative global terms charge phase divergence, aggregate concentration, and late-phase concentration.",
    });
    for (const [label, symbol, key] of [
      ["phase divergence", String.raw`\theta_{\mathrm{TV}}`, "geometry:phase_tv"],
      ["aggregate concentration", String.raw`\theta_{\mathrm{agg}}`, "geometry:aggregate_hhi"],
      ["late concentration", String.raw`\theta_1`, "geometry:phase1_hhi"],
    ] as const) {
      const chip = fittedChip(detail, label, symbol, key);
      if (chip) chips.push(chip);
    }
  }
  return {
    topLevelTex,
    topLevelExplanation: "Predicted benchmark loss is an additive sum of interpretable bucket responses.",
    layers,
    chips,
  };
}

function separateHeadsForm(detail: FitDetail, policyClass: PolicyClass): ModelForm {
  const l2 = fittedChip(detail, "ridge selected by CV", String.raw`\lambda_{L2}`, "l2");
  if (policyClass === "single_phase") {
    return {
      topLevelTex: String.raw`\widehat Y_b(w)=b_0+\sum_i B_i(e_i)`,
      topLevelExplanation: "The policy ablation fits one asymmetric bowl to each bucket's total exposure.",
      layers: [
        aggregateExposureLayer(),
        {
          label: "02 / Aggregate coordinate",
          title: "Distance from preferred exposure",
          tex: String.raw`u_i=\log(1+e_i)-\mu_i`,
          explanation: "The learned center corresponds to a preferred total exposure of exp(mu) - 1 epochs.",
        },
        {
          label: "03 / Aggregate head",
          title: "Asymmetric exposure bowl",
          tex: String.raw`B_i(e)=a_i^-\min(u_i,0)^2+a_i^+\max(u_i,0)^2`,
          explanation: "Nonnegative curvatures separately model underexposure and overexposure around the aggregate optimum.",
        },
      ],
      chips: l2 ? [l2] : [],
    };
  }
  return {
    topLevelTex: String.raw`\widehat Y_b(w)=b_0+\sum_i\sum_{t\in\{0,1\}}B_i^{(t)}\!\left(e_i^{(t)}\right)`,
    topLevelExplanation: "Each phase has its own preferred exposure and asymmetric cost for falling below or above it.",
    layers: [
      commonExposureLayer(),
      {
        label: "02 / Phase coordinate",
        title: "Distance from preferred exposure",
        tex: String.raw`u_i^{(t)}=\log\!\left(1+e_i^{(t)}\right)-\mu_i^{(t)}`,
        explanation: "The learned center corresponds to a preferred phase exposure of exp(mu) - 1 epochs.",
      },
      {
        label: "03 / Phase head",
        title: "Asymmetric exposure bowl",
        tex: String.raw`\begin{array}{rl}B_i^{(t)}(e)={}&a_{i,t}^{-}\min(u_i^{(t)},0)^2\\[2pt]&{}+a_{i,t}^{+}\max(u_i^{(t)},0)^2\end{array}`,
        explanation: "Separate nonnegative curvatures model underexposure and overexposure on each side of each phase optimum.",
      },
    ],
    chips: l2 ? [l2] : [],
  };
}

function retainedExposureLayer(policyClass: PolicyClass): FormulaLayer {
  if (policyClass === "single_phase") return aggregateExposureLayer();
  return {
    label: "01 / Memory",
    title: "Retained effective exposure",
    tex: String.raw`x_i=\exp\!\left[-\lambda(1-w_i^{(1)})\right]e_i^{(0)}+\eta e_i^{(1)},\qquad e_i^{(t)}=c_i^{(t)}w_i^{(t)}`,
    explanation: "Phase-0 exposure decays when a bucket is absent late; phase-1 exposure receives a learned relative value.",
  };
}

function compactRetainedForm(detail: FitDetail, policyClass: PolicyClass): ModelForm {
  const chips: FormulaChip[] = [];
  for (const [label, symbol, keys] of [
    ["Weibull rate", String.raw`\rho`, ["rho"]],
    ["Weibull shape", String.raw`p`, ["power"]],
    ["late-epoch value", String.raw`\eta`, ["eta"]],
    ["forgetting rate", String.raw`\lambda`, ["lambda"]],
    ["ridge", String.raw`\lambda_{L2}`, ["l2"]],
  ] as const) {
    if (policyClass === "single_phase" && (label === "late-epoch value" || label === "forgetting rate")) continue;
    const chip = fittedChip(detail, label, symbol, ...keys);
    if (chip) chips.push(chip);
  }
  return {
    topLevelTex: String.raw`\widehat Y_b(w)=b_0-\sum_i a_i\!\left[1-e^{-(\rho z_i)^p}\right]+c\sum_i[q_i-1]_+^2`,
    topLevelExplanation: "Useful retained learning saturates per bucket; literal replay beyond one epoch enters through one shared harm channel.",
    layers: [
      retainedExposureLayer(policyClass),
      {
        label: "02 / Literal replay",
        title: "Separate exposure ledger",
        tex: String.raw`q_i=e_i^{(0)}+e_i^{(1)},\qquad R(q)=\sum_i[q_i-1]_+^2`,
        explanation: "Repetition harm uses actual simulated epochs rather than the retained-state coordinate used for learning.",
      },
      {
        label: "03 / Learning response",
        title: "Shared Weibull saturation",
        tex: String.raw`S_i(z_i)=1-\exp\!\left[-(\rho z_i)^p\right],\qquad a_i\ge 0,\ c\ge 0`,
        explanation: "Each bucket has one benefit amplitude; all buckets share the response timescale, shape, and replay cost.",
      },
    ],
    chips,
  };
}

function bucketFamilyGrpForm(detail: FitDetail, policyClass: PolicyClass): ModelForm {
  const chips: FormulaChip[] = [];
  for (const [label, symbol, keys] of [
    ["power exponent", String.raw`a`, ["a"]],
    ["late-epoch value", String.raw`\eta`, ["eta"]],
    ["forgetting rate", String.raw`\lambda`, ["lambda"]],
    ["family penalty onset", String.raw`\tau`, ["tau"]],
    ["ridge", String.raw`\lambda_{L2}`, ["l2"]],
  ] as const) {
    if (policyClass === "single_phase" && (label === "late-epoch value" || label === "forgetting rate")) continue;
    const chip = fittedChip(detail, label, symbol, ...keys);
    if (chip) chips.push(chip);
  }
  return {
    topLevelTex: String.raw`\widehat Y_b(w)=b_0-\sum_i a_i x_i^a-\sum_{C:\,|C|>1}A_C\!\left(\sum_{i\in C}x_i\right)^a+\sum_C B_C P_C(x)`,
    topLevelExplanation: "Bucket-specific quality is learned freely, while low-dimensional family channels capture coverage complementarity and shared replay harm.",
    layers: [
      retainedExposureLayer(policyClass),
      {
        label: "02 / Bucket response",
        title: "Unordered quality tiers",
        tex: String.raw`S_i(x_i)=x_i^a,\qquad a_i\ge 0`,
        explanation: "Every bucket, including every production Q tier, receives its own amplitude; no monotonic quality ordering is imposed.",
      },
      {
        label: "03 / Family response",
        title: "Coverage and replay channels",
        tex: String.raw`\begin{array}{c}X_C=\sum_{i\in C}x_i\\[2pt]P_C(x)=\operatorname{softplus}\!\left(\log(1+X_C)-\tau\right)^2\end{array}`,
        explanation: "Non-singleton families add a nonlinear coverage benefit; every family has one nonnegative repetition-harm coefficient.",
      },
    ],
    chips,
  };
}

function powerSeparateHeadsGrpForm(
  detail: FitDetail,
  policyClass: PolicyClass,
  familyOnset = false,
): ModelForm {
  const chips: FormulaChip[] = [];
  for (const [label, symbol, keys] of [
    ["power exponent", String.raw`a`, ["a"]],
    ["forgetting rate", String.raw`\lambda`, ["lambda"]],
    ["family penalty onset", String.raw`\tau`, ["tau"]],
    ["ridge", String.raw`\lambda_{L2}`, ["l2"]],
  ] as const) {
    if (familyOnset && label === "family penalty onset") continue;
    if (policyClass === "single_phase" && label === "forgetting rate") continue;
    const chip = fittedChip(detail, label, symbol, ...keys);
    if (chip) chips.push(chip);
  }
  if (policyClass === "single_phase") {
    return {
      topLevelTex: String.raw`\widehat Y_b(w)=b_0-\sum_i a_i x_i^a-\sum_{C:\,|C|>1}A_C X_C^a+\sum_C B_C P_C(x)`,
      topLevelExplanation: "The policy-matched ablation fits one aggregate power-response head and retunes all shared shape terms.",
      layers: [
        aggregateExposureLayer(),
        {
          label: "02 / Aggregate response",
          title: "Bucket and family coverage",
          tex: String.raw`x_i=e_i,\qquad X_C=\sum_{i\in C}x_i,\qquad S(z)=z^a`,
          explanation: "Every bucket and non-singleton family has one nonnegative aggregate-response amplitude.",
        },
        {
          label: "03 / Replay harm",
          title: familyOnset ? "Family-specific onsets" : "Shared family onset",
          tex: familyOnset
            ? String.raw`P_C(x)=\operatorname{softplus}\!\left(\log\!\left(1+\sum_{i\in C}x_i\right)-\tau_C\right)^2`
            : String.raw`P_C(x)=\operatorname{softplus}\!\left(\log\!\left(1+\sum_{i\in C}x_i\right)-\tau\right)^2`,
          explanation: familyOnset
            ? "Each family learns its own replay onset, shrinkage-selected toward the shared-onset control."
            : "One nonnegative coefficient per family charges retained exposure beyond a shared onset.",
        },
      ],
      chips,
    };
  }
  return {
    topLevelTex: String.raw`\widehat Y_b(w)=b_0-\sum_{t=0}^1\!\left[\sum_i a_i^{(t)}(x_i^{(t)})^a+\sum_{C:\,|C|>1}A_C^{(t)}(X_C^{(t)})^a\right]+\sum_C B_C P_C(x)`,
    topLevelExplanation: "Useful bucket and family responses have independent early/late amplitudes; memory and replay harm remain shared.",
    layers: [
      {
        label: "01 / Phase state",
        title: "Retained early state and literal late exposure",
        tex: String.raw`x_i^{(0)}=e^{-\lambda(1-w_i^{(1)})}e_i^{(0)},\qquad x_i^{(1)}=e_i^{(1)},\qquad e_i^{(t)}=c_i^{(t)}w_i^{(t)}`,
        explanation: "Only phase-0 learning is discounted for non-revisit; phase 1 remains literal realized exposure.",
      },
      {
        label: "02 / Separate response heads",
        title: "Phase-specific bucket and family value",
        tex: String.raw`X_C^{(t)}=\sum_{i\in C}x_i^{(t)},\qquad S(z)=z^a,\qquad a_i^{(t)},A_C^{(t)}\ge 0`,
        explanation: "The response shape is shared, but the data can have different predictive value early and late.",
      },
      {
        label: "03 / Shared replay ledger",
        title: familyOnset ? "Family-specific overexposure onset" : "Family-level overexposure",
        tex: familyOnset
          ? String.raw`P_C(x)=\operatorname{softplus}\!\left(\log\!\left(1+\sum_{i\in C}(x_i^{(0)}+x_i^{(1)})\right)-\tau_C\right)^2`
          : String.raw`P_C(x)=\operatorname{softplus}\!\left(\log\!\left(1+\sum_{i\in C}(x_i^{(0)}+x_i^{(1)})\right)-\tau\right)^2`,
        explanation: familyOnset
          ? "Each family learns a replay onset with CV-selected shrinkage toward the shared-onset control."
          : "A single replay channel per family prevents the additional response heads from duplicating harm terms.",
      },
    ],
    chips,
  };
}

function retainedFamilyGrpForm(
  detail: FitDetail,
  policyClass: PolicyClass,
  literalFamilyReplay: boolean,
): ModelForm {
  const chips: FormulaChip[] = [];
  for (const [label, symbol, keys] of [
    ["Weibull rate", String.raw`\rho`, ["rho"]],
    ["Weibull shape", String.raw`p`, ["power"]],
    ["late-epoch value", String.raw`\eta`, ["eta"]],
    ["forgetting rate", String.raw`\lambda`, ["lambda"]],
    ["family penalty onset", String.raw`\tau`, ["tau"]],
    ["ridge", String.raw`\lambda_{L2}`, ["l2"]],
  ] as const) {
    if (policyClass === "single_phase" && (label === "late-epoch value" || label === "forgetting rate")) continue;
    if (literalFamilyReplay && label === "family penalty onset") continue;
    const chip = fittedChip(detail, label, symbol, ...keys);
    if (chip) chips.push(chip);
  }
  const replayTex = literalFamilyReplay
    ? String.raw`q_i=e_i^{(0)}+e_i^{(1)},\qquad R_C=\sum_{i\in C}[q_i-1]_+^2`
    : String.raw`X_C=\sum_{i\in C}x_i,\qquad R_C=\operatorname{softplus}\!\left(\log(1+X_C)-\tau\right)^2`;
  const replayExplanation = literalFamilyReplay
    ? "Replay harm uses actual simulated epochs beyond one pass and learns one nonnegative strength per family."
    : "Replay harm uses aggregate retained family exposure, one shared learned onset, and one nonnegative strength per family.";
  return {
    topLevelTex: String.raw`\widehat Y_b(w)=b_0-\sum_i a_iS(x_i)-\sum_{C:\,|C|>1}A_CS(\bar x_C)+\sum_CB_CR_C`,
    topLevelExplanation: "A shared learning curve pools statistical strength while bucket amplitudes, family coverage, and replay harm remain interpretable.",
    layers: [
      retainedExposureLayer(policyClass),
      {
        label: "02 / Learning response",
        title: "Shared Weibull saturation",
        tex: String.raw`S(x)=1-\exp[-(\rho x)^p],\qquad \bar x_C=\frac{1}{|C|}\sum_{i\in C}x_i`,
        explanation: "Every bucket and non-singleton family shares one saturation timescale and shape; amplitudes remain independently nonnegative.",
      },
      {
        label: "03 / Replay",
        title: literalFamilyReplay ? "Literal replay by family" : "Shared-onset family replay",
        tex: replayTex,
        explanation: replayExplanation,
      },
    ],
    chips,
  };
}

function grpForm(detail: FitDetail, swarmId: string, policyClass: PolicyClass): ModelForm {
  const shapeChips: FormulaChip[] = [];
  for (const [label, symbol, keys] of [
    ["late-epoch value", String.raw`\eta`, ["eta"]],
    ["retention rate", String.raw`\lambda`, ["lam", "lambda"]],
    ["quality discount", String.raw`\beta`, ["beta"]],
    ["signal scale", String.raw`\alpha`, ["alpha"]],
  ] as const) {
    if (policyClass === "single_phase" && (label === "late-epoch value" || label === "retention rate")) continue;
    const chip = fittedChip(detail, label, symbol, ...keys);
    if (chip) shapeChips.push(chip);
  }

  if (swarmId.startsWith("starcoder")) {
    return {
      topLevelTex: String.raw`\widehat Y_b(w)=b_0-\beta_B\log(1+\alpha x_B)-\beta_C\log(1+\alpha x_C)+\pi\sum_{i\in\{B,C\}}P(x_i)`,
      topLevelExplanation: "The two-corpus GRP uses retained exposure, log-satiating benefits, and one shared repetition penalty.",
      layers: [
        retainedExposureLayer(policyClass),
        {
          label: "02 / Benefit",
          title: "Corpus-level satiation",
          tex: String.raw`S_i(x_i)=\log(1+\alpha x_i)`,
          explanation: "Broad and code exposure receive separate nonnegative benefit amplitudes but share one response shape.",
        },
        {
          label: "03 / Penalty",
          title: "Aggregate repetition cost",
          tex: String.raw`P(x)=\operatorname{softplus}\!\left(\log(1+x)-\tau\right)^2`,
          explanation: "The two corpus penalties are summed and scaled by one learned nonnegative coefficient.",
        },
      ],
      chips: shapeChips,
    };
  }

  if (swarmId === "production") {
    return {
      topLevelTex: String.raw`\widehat Y_b(w)=b_0-\sum_i\beta_i x_i^a+\sum_i\pi_i\operatorname{softplus}\!\left(\log(1+x_i)-\tau\right)^2`,
      topLevelExplanation: "Production GRP is an ungrouped ablation: each bucket has a response amplitude and penalty, with shared shapes.",
      layers: [
        retainedExposureLayer(policyClass),
        {
          label: "02 / Benefit",
          title: "Shared diminishing returns",
          tex: String.raw`S_i(x_i)=x_i^a,\qquad 0<a<1`,
          explanation: "A shared power exponent gives every bucket diminishing returns while retaining bucket-specific amplitudes.",
        },
        {
          label: "03 / Penalty",
          title: "Bucket overexposure",
          tex: String.raw`P_i(x_i)=\operatorname{softplus}\!\left(\log(1+x_i)-\tau\right)^2`,
          explanation: "The production ablation omits semantic families and fits a nonnegative penalty amplitude per bucket.",
        },
      ],
      chips: shapeChips,
    };
  }

  return {
    topLevelTex: String.raw`\begin{array}{rl}\widehat Y_b(w)={}&b_0-\displaystyle\sum_{g\in G}\beta_g\,X_{g,\mathrm{sig}}^{\,a_{f(g)}}-\displaystyle\sum_f\beta_f X_f^{a_f}\\[3pt]&{}+\displaystyle\sum_f\pi_f\sum_{g\in G_f}P_f(X_{g,\mathrm{raw}})\end{array}`,
    topLevelExplanation: "GRP pools retained exposure through semantic bucket pairs and families, then adds family-specific repetition costs.",
    layers: [
      retainedExposureLayer(policyClass),
      {
        label: "02 / Grouping",
        title: "Singletons, quality pairs, and families",
        tex: String.raw`X_{g,\mathrm{sig}}=x_{g,\mathrm{high}}+\beta x_{g,\mathrm{low}},\qquad X_f=\sum_{i\in f}x_i`,
        explanation: "Singleton buckets remain individual; paired Common Crawl buckets share a quality-discounted signal; family totals add pooled evidence.",
      },
      {
        label: "03 / Response",
        title: "Family power law and repetition penalty",
        tex: String.raw`S_f(X)=X^{a_f},\qquad P_f(X)=\operatorname{softplus}\!\left(\log(1+X)-\tau_f\right)^2`,
        explanation: "Each semantic family learns a diminishing-returns exponent and a threshold for grouped overexposure.",
      },
    ],
    chips: shapeChips,
  };
}

export function modelForm(
  modelId: ModelId,
  detail: FitDetail,
  swarmId: string,
  policyClass: PolicyClass,
): ModelForm {
  if (modelId === "linear") return linearForm(policyClass);
  if (modelId === "olmix_loglinear") return olmixLoglinearForm(detail, policyClass);
  if (modelId === "canonical") return dspForm(detail, false, false, policyClass);
  if (modelId === "effective_exposure") return dspForm(detail, true, false, policyClass);
  if (modelId === "effective_exposure_geometry") return dspForm(detail, true, true, policyClass);
  if (modelId === "separate_heads") return separateHeadsForm(detail, policyClass);
  if (modelId === "compact_retained_state") return compactRetainedForm(detail, policyClass);
  if (modelId === "bucket_family_grp") return bucketFamilyGrpForm(detail, policyClass);
  if (modelId === "bucket_family_power_separate_heads") return powerSeparateHeadsGrpForm(detail, policyClass);
  if (modelId === "bucket_family_power_separate_heads_family_onset") {
    return powerSeparateHeadsGrpForm(detail, policyClass, true);
  }
  if (modelId === "bucket_family_weibull_shared_onset") {
    return retainedFamilyGrpForm(detail, policyClass, false);
  }
  if (modelId === "bucket_family_weibull_family_replay") {
    return retainedFamilyGrpForm(detail, policyClass, true);
  }
  return grpForm(detail, swarmId, policyClass);
}
