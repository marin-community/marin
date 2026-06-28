# Diversity follow-up: grouped pair-total surrogate

Selected model:
- `CCPairTotal-RetainedTotal`

Selected parameters:
```json
{
  "alpha": 5.693767311270728,
  "diversity_mode": "none",
  "eta": 6.323564464532408,
  "group_signal_kind": "log_after_sum",
  "lam": 0.004606280004722357,
  "pen_kind": "group_log_threshold",
  "premium_mode": "global",
  "reg": 2.1562923313580246e-06,
  "signal_kind": "retained_total",
  "tau": 1.3976070420563145
}
```

Train metrics:
```json
{
  "best_candidate": "run_00125",
  "best_value": 1.0571987628936768,
  "chosen_candidate": "run_00125",
  "chosen_value": 1.0571987628936768,
  "r2": 0.8156566204029498,
  "regret_at_1": 0.0,
  "rmse": 0.008225217452190273,
  "spearman": 0.8954562600733857
}
```

5-fold CV metrics (seed 0 split):
```json
{
  "best_candidate": "run_00125",
  "best_value": 1.0571987628936768,
  "chosen_candidate": "run_00125",
  "chosen_value": 1.0571987628936768,
  "fold_choices": "[\"run_00155\", \"run_00213\", \"run_00125\", \"baseline_unimax\", \"run_00200\"]",
  "foldmax_regret_at_1": 0.014046669006347878,
  "foldmean_regret_at_1": 0.00377960205078125,
  "foldmedian_regret_at_1": 0.0,
  "r2": 0.7617938423502401,
  "regret_at_1": 0.0,
  "rmse": 0.009349967636720422,
  "spearman": 0.8647637255238161
}
```

10x repeated 5-fold CV:
- mean CV RMSE: 0.009353
- std CV RMSE: 0.000140
- mean CV fold-mean Regret@1: 0.002944
- std CV fold-mean Regret@1: 0.002168

Predicted optimum:
- predicted bpb: 1.012505
- nearest observed run: baseline_unimax
- nearest observed TV distance: 0.454117
- phase 0 max weight: 0.246895
- phase 1 max weight: 0.176162
- phase 0 effective support: 14.853
- phase 1 effective support: 19.377
- phase 0 weights below 1e-6: 0
- phase 1 weights below 1e-6: 0
- phase 0 weights below 1e-4: 6
- phase 1 weights below 1e-4: 7

Observed best run:
```json
{
  "phase0_below_1e4": 0,
  "phase0_below_1e6": 0,
  "phase0_effn": 27.897806656881908,
  "phase0_entropy": 3.328548071288365,
  "phase0_max": 0.1237906636012571,
  "phase1_below_1e4": 0,
  "phase1_below_1e6": 0,
  "phase1_effn": 25.589006847567454,
  "phase1_entropy": 3.242162839241441,
  "phase1_max": 0.0954325883672167,
  "run_name": "run_00125",
  "value": 1.0571987628936768
}
```

Top phase-0 domains:
```json
[
  {
    "domain": "dolma3_stack_edu",
    "weight": 0.246894786893366,
    "epochs": 9.318371529249728
  },
  {
    "domain": "dolmino_stack_edu_fim",
    "weight": 0.20945888162443954,
    "epochs": 7.916555365789197
  },
  {
    "domain": "dolmino_synth_qa",
    "weight": 0.04172866367584748,
    "epochs": 0.4005535985345781
  },
  {
    "domain": "dolma3_cc/science_math_and_technology_low",
    "weight": 0.04163709835235941,
    "epochs": 1.89908091002295
  },
  {
    "domain": "dolmino_olmocr_pdfs_hq",
    "weight": 0.03556527036596937,
    "epochs": 0.8740330674122344
  },
  {
    "domain": "dolma3_cc/finance_and_business_high",
    "weight": 0.03507329939353761,
    "epochs": 0.32626439390616935
  },
  {
    "domain": "dolma3_cc/entertainment_high",
    "weight": 0.03462652634411929,
    "epochs": 0.41381692331417613
  },
  {
    "domain": "dolmino_common_crawl_hq",
    "weight": 0.03448449630772694,
    "epochs": 0.132436579516536
  },
  {
    "domain": "dolma3_cc/science_math_and_technology_high",
    "weight": 0.027569287999548796,
    "epochs": 0.5740049963291711
  },
  {
    "domain": "dolma3_cc/crime_and_law_high",
    "weight": 0.026946581656149906,
    "epochs": 0.8872877045959378
  },
  {
    "domain": "dolma3_cc/finance_and_business_low",
    "weight": 0.025161360330390535,
    "epochs": 0.4912152973556177
  },
  {
    "domain": "dolma3_cc/literature_high",
    "weight": 0.02453136391666872,
    "epochs": 0.48049846819687925
  }
]
```

Top phase-1 domains:
```json
[
  {
    "domain": "dolmino_stack_edu_fim",
    "weight": 0.17616211468793747,
    "epochs": 1.6645237521869332
  },
  {
    "domain": "dolma3_stack_edu",
    "weight": 0.1596473888340196,
    "epochs": 1.5063640078724176
  },
  {
    "domain": "dolma3_cc/science_math_and_technology_low",
    "weight": 0.05916770107200967,
    "epochs": 0.6746642780249764
  },
  {
    "domain": "dolmino_olmocr_pdfs_hq",
    "weight": 0.05556811708645008,
    "epochs": 0.34140308317369633
  },
  {
    "domain": "dolmino_synth_qa",
    "weight": 0.041413200053615394,
    "epochs": 0.09938136551152768
  },
  {
    "domain": "dolma3_cc/crime_and_law_high",
    "weight": 0.03784251123572711,
    "epochs": 0.3115162746703263
  },
  {
    "domain": "dolma3_cc/science_math_and_technology_high",
    "weight": 0.03599716130298597,
    "epochs": 0.18736927883229343
  },
  {
    "domain": "dolma3_cc/industrial_high",
    "weight": 0.034959801255292855,
    "epochs": 0.5414971897525406
  },
  {
    "domain": "dolma3_arxiv",
    "weight": 0.03195550756036066,
    "epochs": 1.4315995909851826
  },
  {
    "domain": "dolmino_common_crawl_hq",
    "weight": 0.03050902920336566,
    "epochs": 0.029292232051237016
  },
  {
    "domain": "dolma3_cc/literature_low",
    "weight": 0.029095700384190442,
    "epochs": 0.5343542191617718
  },
  {
    "domain": "dolmino_synth_code",
    "weight": 0.029057130563500937,
    "epochs": 1.9489268865807468
  }
]
```
