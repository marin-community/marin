# Standalone Code

These scripts are intentionally small and self-contained. They do not import Marin, Levanter, Iris,
or any branch-local experiment modules. They only expect the CSV/NPZ files shipped in this packet.

Suggested environment:

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r standalone_code/requirements.txt
```

Smoke commands from the packet root:

```bash
python standalone_code/load_packet.py
python standalone_code/fit_mct_lrq.py --output-dir outputs/mct_lrq_demo
python standalone_code/grp_no_l2_exact.py --mode fit-best --output-dir outputs/grp_no_l2_exact
```

`fit_mct_lrq.py` is a compact, readable implementation of the MCT-LRQ family:

```text
L(w,N,D) = E_LRQ(w)
         + A ((N/N0)^(-alpha) - 1)
         + B_fam(w) ((D/D0)^(-beta) - 1)
         + C ((N/N0)^(-gamma) (D/D0)^(-delta) - 1)
```

`grp_no_l2_exact.py` is a standalone port of the repo's GRP power-family-penalty no-L2 path. It includes
the exact retained-exposure design, NNLS linear head, calibration objective, raw optimum optimizer, and
full nonlinear retuning procedure:

```bash
python standalone_code/grp_no_l2_exact.py --mode retune --method Powell --coarse-top-k 3
```
