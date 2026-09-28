# Existing RegMix path measurements

`historical_measurements.csv` has ten rows: three MARINER endpoint trainer seeds,
one RegMix endpoint seed, and one runtime-rounded midpoint seed for each of
Uncheatable and OlmoBaseEval Easy (`table9`). It retains each predictor's frozen
prediction. `weights_json` maps all 39 bucket names to actual runtime weights;
`historical_weights.csv` provides the same six distinct policies in long form.

`collect_historical_measurements.py` reproduces both files offline with `uv run`.
It checks endpoint measurements against their archived source CSVs, RegMix
endpoint weights against the original proposal CSV, and both midpoint
predictions against the pre-training freeze. Source and output hashes are in
`historical_measurements_receipt.json`. No models are fitted and no evaluation
aggregates are reconstructed from a current raw parent value.

Uncheatable endpoints retain their original legacy metric. The midpoint uses
the original seven frozen task weights applied to measured schema-2 pooled
component BPBs, without an empirical offset. This is the same bridge used for
the audited Figure 10 update: over 98 historical checkpoint evaluations, the
largest observed aggregate estimator discrepancy was 0.0000538 BPB. That check
does not establish exact legacy equivalence or a universal error bound. The
suite metric remains the native 51-component macro BPB. RegMix endpoints and
both midpoints have one trainer seed, so their measured differences remain
descriptive.

## Loading MARINER without fitting

The existing helper is `fit_predictors` in
`predict_delphi_path_midpoints_20260912.py`, but **do not call it** for this
purpose: it also fits other baselines. Reuse only its direct JSON-loading path:

```python
import json
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path("/Users/calvinxu/Projects/Work/Marin/mixture-selection")))
import mixture_selection as ms

payload = json.loads(frozen_json.read_text())
reference = ms.ObjectiveFit.from_json(payload)
indices = [input_buckets.index(bucket) for bucket in payload["buckets"]]
prediction = reference.predict(np.atleast_2d(input_weights)[:, indices])
```

Use the exact freeze at
`../delphi_path_midpoints_3e18_20260912/prediction_freeze/78b225d045b6d42ed175dd91d5b06cebecca7e7b04ad07102597fd447809337c/<target>/<target>_mariner.json`.
Targets are `uncheatable` and `table9`. The serialized `buckets` field is the
authoritative input order; `ObjectiveFit.predict` performs the saved exposure
transformation and frozen task aggregation. It does not fit or optimize.
