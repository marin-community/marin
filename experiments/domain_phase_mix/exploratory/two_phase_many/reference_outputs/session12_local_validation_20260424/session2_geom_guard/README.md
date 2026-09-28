# Joint mixture/scale v31 geometry-guard continuation archive

Run:

```bash
python code/run_v31_geom_guard_archive.py --packet-root /path/to/chatgpt_pro_hybrid_data_mixing_packet_v31 --out /tmp/v31_geom_guard
```

Main files:

- `reports/REPORT.md`: concise findings and metric tables.
- `code/run_v31_geom_guard_archive.py`: continuation script.
- `code/run_v31_structural_archive.py`: prior rebuilt S2/AFS script used as an import.
- `csv/candidate_metrics.csv`: generated candidate metrics.
- `csv/combined_reference_metrics.csv`: generated metrics plus delayed power-beta references when present.
- `csv/optimum_diagnostics.csv`: raw-simplex and top-8-hull optimum diagnostics.
- `csv/structural_sanity_summary.csv`: monotonicity and structural checks.
- `models/*_model.json`: serialized S2 and geometry-guard artifacts.
- `plots/*.png`: generated comparison plots.
