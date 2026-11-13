# RL Data Atlas style

Enum chips carry the entire state label. Never put text below a Quality, Result,
Contamination or Daytona chip. Put numeric issue rates and lexical overlap counts
in their own columns, using an em dash for missing measurements. Keep review
progress, reviewed/retained counts and lower bounds on the source Evidence page.

A Result from incomplete content review includes `provisional` inside the chip.
The source table legend and Getting started guide define that variant and all five
Contamination states. Keep the API, table, source details and guide consistent.
Use the shared chip colors, with light and dark theme rules; a remediation warning
also has a strong border. Phone layouts expose hidden columns through source
details and Evidence. Do not rely on color alone to convey a state.

Run the installed-app browser regressions with:

```bash
uv run pytest infra/marina/apps/rl_data_catalog/journeys --journeys -n 0
```

The journeys install only this app into disposable local Postgres. See the
[Atlas reference](../../../../docs/references/rl-data-atlas.md) for packaging,
loading and preview release procedures.
