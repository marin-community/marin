# Frontier factorial design (not launched)

Centre: `delphi_3e18_39bucket:a1a917b1981fc2cad2c2759dccf963b51fb941fd9816c49087f7b119545d4161` (26 runs, measured mean 1.0639).

Factors and +1-level weight deltas (mass balanced against the 26 Common Crawl cells in proportion
to their centre weights):

- A `code`: {"dolma3_stack_edu": 0.02, "dolmino_stack_edu_fim": 0.02}
- B `synth_qa`: {"dolmino_synth_qa": 0.04}
- C `cc_hq`: {"dolmino_common_crawl_hq": 0.04}
- D `synth_reasoning`: {"dolmino_synth_thinking": 0.006, "dolmino_synth_instruction": 0.006, "dolmino_synth_math": 0.006}
- E `pdf_arxiv`: {"dolmino_olmocr_pdfs_hq": 0.015, "dolma3_arxiv": 0.005}

Design: 2^(5-1) half fraction, V (E = ABCD): main effects and all ten two-factor interactions are estimable and unaliased with each other, plus 2 centre replicates;
18 runs at 3e18 (about 61e18 FLOPs). Effect standard error at the Table-9
repeat SD of 0.0038: about 0.0019 BPB for main effects and interactions.

Largest materialized epochs: 11.78 (dolma3_wikipedia); largest TV from
the centre: 0.159.

The candidate table is in the epoch-cap sweep launcher schema; a launcher would bind all rows to one data
seed (662009, the Table-9 validation seed) so that the centre replicates measure seed-free noise.
