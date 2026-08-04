# PDF OCR sample browser

Ad-hoc local viewer for the 10k-document PDF OCR sample (issue 7616 exploration).
Left pane renders the source PDF with PDF.js; right pane shows the OCR text split
per page, with the educational-quality score chips and the model's reasoning.

## Launch

```bash
uv run experiments/build_pdf_source/_sample_browser/app.py
```

Then open http://127.0.0.1:8791. Flags: `--host`, `--port`, `--sample-dir`,
`--scores-dir`, `--scores-v2-dir`.

Startup reads all five sample parquets into memory (roughly 200 MB) and joins the
scoring checkpoints, so it takes a few seconds.

## Score versions

Three educational-quality score sets are browsable side by side:

| version | source | columns | reasoning |
| --- | --- | --- | --- |
| `v1` | oracle, grade-school capped | `edu_score_{begin,middle,end}` | `quality/scores-*.parquet` |
| `v2` | oracle, primary school through graduate | `edu_score_v2_{begin,middle,end}` | `quality_v2/scores-*.parquet` |
| `ft` | trained pooled fast-transformer | `ft_score_{begin,middle,end}` | none (a model score, not a rubric prompt) |

`ft` scores are continuous on the oracle's own 0–4 scale, produced by
`experiments/build_pdf_source/quality/train_pdf_scorer.py` and folded in with
`merge_ft_scores.py`.

## Which documents are shown

By default the browser is restricted to the ~3.3k documents carrying an `ft`
score. The scorer trains on a document-disjoint split, so only its holdout is
scored: showing a model score on a document the model trained on would display
memorised agreement as if it were prediction. Pass
`--restrict-to-version ''` to browse all 10k documents (the unscored ones then
show `…` for `ft`).

The selector in the top bar picks the active version; the choice persists in
`localStorage`. The active version drives the B/M/E chips, the reasoning
expansion, and all four min-score filters. Each chip also carries the other
version's score as a small subscript, and the reasoning panel shows both
versions' justifications one after the other, so disagreements are visible
without toggling.

Scores are read from the parquet columns where they exist. Because reasoning
checkpoints land well before the merge job writes the matching columns, any
score gap is backfilled from the reasoning rows at startup (failed rows, score
< 0, are skipped). That makes a version filterable as soon as its checkpoints
appear; the parquet column always wins where present. A version with neither
columns nor reasoning shows as "pending" in the selector and matches nothing.

## Reload for new columns

The sample parquets are rewritten as background jobs append columns
(`edu_score_*`, `edu_score_v2_*`, `pdf`, `pdf_fetch_status`). The schema and both
reasoning directories are read once at startup, so **restart the app** to pick up
columns or new scoring checkpoints that landed while it was running. Anything
still missing renders as "pending" rather than failing: score chips show `…`, and
the PDF pane shows "PDF not yet available" with the reason from the API.

## Data

- Sample: `/tmp/cc_focus_2026_22_pdf_ocr_all_sample10k/sample-*.parquet` (read-only).
- Model reasoning: `scores-*.parquet` in the `quality/` and `quality_v2/`
  scratchpad directories (read-only). Later
  checkpoints supersede earlier ones for the same `(id, segment)`; a failed row
  (negative score) never displaces a successful one.
- Served PDFs are cached under `~/.cache/sample_browser_pdfs/{id}.pdf`. Delete that
  directory if the `pdf` column is rewritten.

## API

- `GET /api/docs?offset=&limit=&min_begin=&max_begin=&min_middle=&max_middle=&min_end=&max_end=&min_overall=&max_overall=&status=&q=&q_field=title|text|both&score_version=v1|v2|ft`
  — `score_version` (default `v1`) governs the score filters; every row returns
  all versions' scores. Each segment takes an inclusive `[min, max]` range; either
  bound may be omitted. A segment with no score fails any active bound.
- `GET /api/doc/{id}` — metadata, per-page text, and both versions' scores plus reasoning.
- `GET /api/pdf/{id}` — PDF bytes, or 404 with a JSON `error` explaining why not.
- `GET /api/schema` — document count and which optional columns are present.

## UI

Keyboard: `←` / `→` move to the previous/next document in the result list.
Page buttons under the PDF pane scroll the OCR pane to the matching page, and
scrolling the OCR pane moves the PDF page in turn. Clicking a `B`/`M`/`E` chip
expands that segment's reasoning text for both score versions.
