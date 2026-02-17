# Plan DAG Viewer

A lightweight DAG viewer for `.agents/the_plan.yaml`.

## Run locally

From the repo root:

```bash
python3 -m http.server
```

Then open:

```
http://localhost:8000/scripts/pm/pm_dag/index.html
```

The page will try to load `/.agents/the_plan.yaml` automatically. You can also use the file picker to load any YAML.

## Notes

- Supports zoom/pan and simple filters (timeline-only, hide backburner, search).
- Clicking a node shows details and links to GitHub issues if `issue` is present.
