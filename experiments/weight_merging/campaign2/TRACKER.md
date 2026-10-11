# Campaign 2: all 48 completed merges

Measured work-time estimate for a fresh merge: calibration when required plus export and read-back audit; both serial stages included. Excludes queue, runtime setup, parent acquisition, and all benchmark evaluation. avg-01 through avg-04 and task-01 through task-03 exports use task duration and include process startup. Shared foundation cost is charged in full to every dependent row, not amortized. Times are approximate planning estimates measured under concurrent jobs on CoreWeave cw-rno2a; they are not isolated compute benchmarks. CPU exports used 16 CPUs/128 GB, ISO used one H100, and calibration used eight H100s/768 GB. See [reproduction instructions](README.md) and [unrounded evidence](results-evidence.json).

| Experiment | Method | Support | Variant | Changed Base (%) | NUPA200 (%) | BFCL (%) | Sum | Prep (min) | Export (min) | Total merge (min) |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| [avg-01](avg-01.json) | Weighted averaging | R12 | Base/NUPA/BFCL = 0.50/0.25/0.25 | 9.70 | 0.00 | 7.32 | 7.32 | 0.00 | 41.25 | 41.25 |
| [avg-02](avg-02.json) | Weighted averaging | R25 | Base/NUPA/BFCL = 0.50/0.25/0.25 | 19.40 | 0.50 | 20.49 | 20.99 | 0.00 | 50.77 | 50.77 |
| [avg-03](avg-03.json) | Weighted averaging | R25 | Base/NUPA/BFCL = 0.25/0.50/0.25 | 22.08 | 1.00 | 33.33 | 34.33 | 0.00 | 50.62 | 50.62 |
| [avg-04](avg-04.json) | Weighted averaging | R25 | Base/NUPA/BFCL = 0.25/0.25/0.50 | 19.41 | 0.50 | 19.51 | 20.01 | 0.00 | 40.32 | 40.32 |
| [task-01](task-01.json) | Task arithmetic | R12 | NUPA/BFCL delta scales = 0.75/0.75 | 11.75 | 0.50 | 8.94 | 9.44 | 0.00 | 40.72 | 40.72 |
| [task-02](task-02.json) | Task arithmetic | R25 | NUPA/BFCL delta scales = 0.75/0.75 | 23.50 | 1.50 | 33.33 | 34.83 | 0.00 | 50.35 | 50.35 |
| [task-03](task-03.json) | Task arithmetic | R25 | NUPA/BFCL delta scales = 1.00/0.50 | 23.50 | 0.50 | 34.15 | 34.65 | 0.00 | 41.66 | 41.66 |
| [task-04](task-04.json) | Task arithmetic | R25 | NUPA/BFCL delta scales = 0.50/1.00 | 22.09 | 0.00 | 30.08 | 30.08 | 0.00 | 42.67 | 42.67 |
| [ties-01](ties-01.json) | TIES | R12 | density 0.20; donor weights 0.50/0.50; scale 1.00 | 4.47 | 0.50 | 3.25 | 3.75 | 0.00 | 46.07 | 46.07 |
| [ties-02](ties-02.json) | TIES | R25 | density 0.20; donor weights 0.50/0.50; scale 1.00 | 8.95 | 0.00 | 38.21 | 38.21 | 0.00 | 50.47 | 50.47 |
| [ties-03](ties-03.json) | TIES | R25 | density 0.50; donor weights 0.50/0.50; scale 1.00 | 16.68 | 0.00 | 34.15 | 34.15 | 0.00 | 50.38 | 50.38 |
| [ties-04](ties-04.json) | TIES | R25 | density 1.00; donor weights 0.50/0.50; scale 1.00 | 23.50 | 0.00 | 31.71 | 31.71 | 0.00 | 43.01 | 43.01 |
| [dare-linear-01](dare-linear-01.json) | DARE-linear | R12 | keep 0.20; donor scales 0.50/0.50; seed 0 | 3.02 | 0.00 | 5.69 | 5.69 | 0.00 | 42.55 | 42.55 |
| [dare-linear-02](dare-linear-02.json) | DARE-linear | R25 | keep 0.20; donor scales 0.50/0.50; seed 0 | 6.04 | 0.50 | 35.77 | 36.27 | 0.00 | 48.11 | 48.11 |
| [dare-linear-03](dare-linear-03.json) | DARE-linear | R25 | keep 0.50; donor scales 0.50/0.50; seed 0 | 13.84 | 0.00 | 30.08 | 30.08 | 0.00 | 43.55 | 43.55 |
| [dare-linear-04](dare-linear-04.json) | DARE-linear | R25 | keep 0.80; donor scales 0.50/0.50; seed 0 | 20.14 | 0.00 | 28.46 | 28.46 | 0.00 | 41.94 | 41.94 |
| [dare-ties-01](dare-ties-01.json) | DARE-TIES | R12 | DARE keep 0.20; TIES density 0.20; equal donors; seed 0 | 3.02 | 0.00 | 10.57 | 10.57 | 0.00 | 45.31 | 45.31 |
| [dare-ties-02](dare-ties-02.json) | DARE-TIES | R25 | DARE keep 0.20; TIES density 0.20; equal donors; seed 0 | 6.04 | 0.00 | 45.53 | 45.53 | 0.00 | 54.03 | 54.03 |
| [dare-ties-03](dare-ties-03.json) | DARE-TIES | R25 | DARE keep 0.50; TIES density 0.20; equal donors; seed 0 | 8.34 | 0.00 | 41.46 | 41.46 | 0.00 | 51.04 | 51.04 |
| [dare-ties-04](dare-ties-04.json) | DARE-TIES | R25 | DARE keep 0.50; TIES density 0.50; equal donors; seed 0 | 13.84 | 0.00 | 40.65 | 40.65 | 0.00 | 52.35 | 52.35 |
| [structured-01](structured-01.json) | Layer/component merging | Early25 | Earliest complete layer MLP blocks; Base/NUPA/BFCL = 0.50/0.25/0.25 | 18.35 | 0.00 | 19.51 | 19.51 | 0.00 | 32.58 | 32.58 |
| [structured-02](structured-02.json) | Layer/component merging | Late25 | Latest complete layer MLP blocks; Base/NUPA/BFCL = 0.50/0.25/0.25 | 17.07 | 0.00 | 9.76 | 9.76 | 0.00 | 33.26 | 33.26 |
| [structured-03](structured-03.json) | Layer/component merging | SharedR25 | All shared experts, then routed experts; Base/NUPA/BFCL = 0.50/0.25/0.25 | 19.43 | 0.00 | 28.93 | 28.93 | 0.00 | 42.98 | 42.98 |
| [structured-04](structured-04.json) | Layer/component merging | AttentionR25 | All attention projections, then routed experts; Base/NUPA/BFCL = 0.50/0.25/0.25 | 19.43 | 0.00 | 30.89 | 30.89 | 0.00 | 43.17 | 43.17 |
| [activation-01](activation-01.json) | Activation-guided expert transfer | A12 | Domain-balanced expert allocation; exact donor replacement | 8.22 | 3.50 | 2.44 | 5.94 | 31.17 | 40.63 | 71.80 |
| [activation-02](activation-02.json) | Activation-guided expert transfer | A25 | Domain-balanced expert allocation; exact donor replacement | 16.30 | 10.00 | 5.69 | 15.69 | 31.17 | 42.17 | 73.34 |
| [activation-03](activation-03.json) | Activation-guided expert transfer | A25 | Domain-balanced expert allocation; 0.50 donor blend | 14.24 | 0.00 | 13.01 | 13.01 | 31.17 | 40.11 | 71.28 |
| [activation-04](activation-04.json) | Activation-guided expert transfer | RandomA25 | Matched-count random expert allocation; exact donor replacement; seed 0 | 15.93 | 0.00 | 26.02 | 26.02 | 31.17 | 41.52 | 72.68 |
| [serial-01](serial-01.json) | Serial merging | A12 | NUPA then BFCL; disjoint domain supports; exact replacement | 8.22 | 1.50 | 3.25 | 4.75 | 31.17 | 89.94 | 121.11 |
| [serial-02](serial-02.json) | Serial merging | A25 | NUPA then BFCL; disjoint domain supports; exact replacement | 16.30 | 7.00 | 5.69 | 12.69 | 31.17 | 80.23 | 111.39 |
| [serial-03](serial-03.json) | Serial merging | R25 | NUPA then BFCL; same support; each stage 0.50 donor blend | 20.51 | 0.50 | 23.58 | 24.08 | 0.00 | 89.30 | 89.30 |
| [serial-04](serial-04.json) | Serial merging | R25 | BFCL then NUPA; same support; each stage 0.50 donor blend | 22.08 | 0.00 | 21.14 | 21.14 | 0.00 | 84.14 | 84.14 |
| [ram-01](ram-01.json) | RAM / RAM+ | R12 | RAM; epsilon 1e-4; unit donor coefficients | 9.29 | 0.00 | 12.20 | 12.20 | 0.00 | 41.16 | 41.16 |
| [ram-02](ram-02.json) | RAM / RAM+ | R25 | RAM; epsilon 1e-4; unit donor coefficients | 18.58 | 0.50 | 38.21 | 38.71 | 0.00 | 40.53 | 40.53 |
| [ram-03](ram-03.json) | RAM / RAM+ | R25 | RAM+; epsilon 1e-4; unique rescale 0.10; cap 0.20 | 18.58 | 0.50 | 39.84 | 40.34 | 0.00 | 41.26 | 41.26 |
| [ram-04](ram-04.json) | RAM / RAM+ | R25 | RAM+; epsilon 1e-4; unique rescale 0.20; cap 0.40 | 18.58 | 1.00 | 40.65 | 41.65 | 0.00 | 42.59 | 42.59 |
| [expertpp-01](expertpp-01.json) | Expert Merging++ adaptation | R12 | Learn convex coefficients; logits 1.00, hidden 1.00; retention 1.00 | 8.98 | 0.00 | 11.38 | 11.38 | 120.33 | 39.40 | 159.73 |
| [expertpp-02](expertpp-02.json) | Expert Merging++ adaptation | R25 | Learn convex coefficients; logits 1.00, hidden 1.00; retention 1.00 | 17.16 | 0.50 | 21.95 | 22.45 | 185.69 | 40.32 | 226.01 |
| [expertpp-03](expertpp-03.json) | Expert Merging++ adaptation | R25 | Learn convex coefficients; logits 1.00, hidden 1.00; retention 4.00 | 16.97 | 0.00 | 24.39 | 24.39 | 187.34 | 40.15 | 227.50 |
| [expertpp-04](expertpp-04.json) | Expert Merging++ adaptation | R25 | Learn convex coefficients; logits 1.00, hidden 0.00; retention 1.00 | 17.13 | 0.00 | 20.33 | 20.33 | 195.42 | 42.22 | 237.64 |
| [ota-ffg-01](ota-ffg-01.json) | OTA + FFG adaptation | R12 | Calibrated second moments; graft density 0.25; equal donors | 3.27 | 0.50 | 8.13 | 8.63 | 31.17 | 56.01 | 87.17 |
| [ota-ffg-02](ota-ffg-02.json) | OTA + FFG adaptation | R25 | Calibrated second moments; graft density 0.25; equal donors | 6.49 | 0.50 | 22.76 | 23.26 | 31.17 | 54.38 | 85.54 |
| [ota-ffg-03](ota-ffg-03.json) | OTA + FFG adaptation | R25 | Calibrated second moments; graft density 0.50; equal donors | 11.54 | 0.50 | 20.33 | 20.83 | 31.17 | 54.27 | 85.43 |
| [ota-ffg-04](ota-ffg-04.json) | OTA + FFG adaptation | R25 | Calibrated second moments; graft density 1.00; equal donors | 19.53 | 0.00 | 25.20 | 25.20 | 31.17 | 50.99 | 82.15 |
| [iso-01](iso-01.json) | ISO-Merger adaptation | R12 | Base spectrum; mode keep 0.90; unit-retention solve | 11.51 | 0.00 | 10.57 | 10.57 | 0.00 | 79.21 | 79.21 |
| [iso-02](iso-02.json) | ISO-Merger adaptation | R25 | Base spectrum; mode keep 0.90; unit-retention solve | 23.03 | 0.00 | 39.84 | 39.84 | 0.00 | 119.24 | 119.24 |
| [iso-03](iso-03.json) | ISO-Merger adaptation | R25 | Base spectrum; mode keep 1.00; unit-retention solve | 23.14 | 1.00 | 43.09 | 44.09 | 0.00 | 113.55 | 113.55 |
| [iso-04](iso-04.json) | ISO-Merger adaptation | R25 | Base spectrum; mode keep 0.75; unit-retention solve | 22.85 | 0.50 | 32.52 | 33.02 | 0.00 | 122.61 | 122.61 |
