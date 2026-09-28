# Frontier factorial results (18 of 18 runs measured)

## table9_macro_bpb

Centre runs: centre_r0_cap16 1.0642, centre_r1_cap16 1.0615; 26-run centre mean 1.0639 (SD 0.0041). Centre pair difference 0.0027.

     term  effect_bpb     se        t        kind  corners_used
intercept     1.06755    NaN      NaN   intercept            16
        A    -0.00650 0.0019 -3.42295        main            16
        B    -0.00795 0.0019 -4.18495        main            16
        C    -0.00083 0.0019 -0.43631        main            16
        D    -0.00399 0.0019 -2.10055        main            16
        E    -0.00139 0.0019 -0.73345        main            16
       AB     0.00165 0.0019  0.86783 interaction            16
       AC    -0.00073 0.0019 -0.38634 interaction            16
       AD    -0.00174 0.0019 -0.91621 interaction            16
       AE    -0.00071 0.0019 -0.37495 interaction            16
       BC    -0.00165 0.0019 -0.86664 interaction            16
       BD     0.00073 0.0019  0.38412 interaction            16
       BE     0.00069 0.0019  0.36368 interaction            16
       CD    -0.00037 0.0019 -0.19631 interaction            16
       CE    -0.00052 0.0019 -0.27388 interaction            16
       DE    -0.00039 0.0019 -0.20617 interaction            16

Interactions beyond 2.5 SE: none

## Proposals (fitted model over the factor box; `proposal_ranking.csv`)

   candidate_id               kind  scale  predicted_bpb  feasible  tv_to_centre  max_materialized_epoch  kernel_forecast  kernel_mass
  prop_ppppp_x2 best corner scaled    2.0         1.0408      True        0.3160                 12.4553           1.0665       0.0000
prop_ppppp_x1p5 best corner scaled    1.5         1.0486      True        0.2374                 11.7844           1.0662       0.0006
  prop_ppzpz_x2   main-effect step    2.0         1.0504      True        0.1961                 12.4553           1.0654       0.0115
prop_ppzpz_x1p5   main-effect step    1.5         1.0544      True        0.1473                 11.7844           1.0658       0.1400
  prop_ppppp_x1    measured corner    1.0         1.0557      True        0.1590                 11.7844           1.0662       0.0852
  prop_ppppm_x1  unmeasured corner    1.0         1.0580      True        0.1384                 11.7844           1.0661       0.1804
  prop_ppzpz_x1   main-effect step    1.0         1.0586      True        0.0979                 11.7844           1.0661       0.6824
  prop_ppmpp_x1  unmeasured corner    1.0         1.0598      True        0.1188                 11.7844           1.0661       0.4689
  prop_ppmpm_x1    measured corner    1.0         1.0611      True        0.0986                 11.7844           1.0662       0.6951
  prop_pppmp_x1  unmeasured corner    1.0         1.0615      True        0.1406                 11.7844           1.0664       0.1122

## uncheatable_bpb

Centre runs: centre_r0_cap16 0.9951, centre_r1_cap16 0.9932. Centre pair difference 0.0019.

     term  effect_bpb      se        t        kind  corners_used
intercept     0.99653     NaN      NaN   intercept            16
        A     0.00185 0.00045  4.11234        main            16
        B     0.00061 0.00045  1.34735        main            16
        C    -0.00028 0.00045 -0.62146        main            16
        D     0.00244 0.00045  5.42957        main            16
        E    -0.00382 0.00045 -8.48417        main            16
       AB    -0.00002 0.00045 -0.03439 interaction            16
       AC    -0.00013 0.00045 -0.28963 interaction            16
       AD    -0.00015 0.00045 -0.33490 interaction            16
       AE     0.00031 0.00045  0.69100 interaction            16
       BC    -0.00054 0.00045 -1.20098 interaction            16
       BD     0.00002 0.00045  0.03955 interaction            16
       BE     0.00002 0.00045  0.03846 interaction            16
       CD    -0.00003 0.00045 -0.06250 interaction            16
       CE    -0.00025 0.00045 -0.55249 interaction            16
       DE    -0.00016 0.00045 -0.36218 interaction            16

Interactions beyond 2.5 SE: none

