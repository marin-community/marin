"""Standalone agentic evaluation package for Marin.

Extracted from OpenThoughts-Agent (#6958). Provides the utilities to run
agentic Harbor-based evaluations (SWE-bench, Terminal-Bench, etc.) on Iris
TPU/GPU clusters without depending on the full OT-Agent codebase.

Architecture:
  - ``presets/`` — benchmark preset catalog (dataset, concurrency, parser)
  - ``serve/`` — vLLM serve config construction + model-config registry
  - ``harness/`` — Harbor harness wiring (config load, command build, trial prune)
  - ``runtime/`` — worker runtime (Ray+vLLM lifecycle, harbor exec)
  - ``results/`` — pluggable result sinks (local, HF upload)
  - ``backends/`` — pluggable cluster backends (Iris TPU/GPU)
  - ``launch.py`` — launcher entry point (submit a job)
  - ``run_eval.py`` — worker entry point (run inside the pod)
"""
