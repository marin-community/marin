# Snapshot tests

We use snapshot tests to ensure that our HTML-to-text conversion is working as expected.

There is one split per source format: `web`, `wiki`, `ar5iv`, and `stackexchange`. Each split
has an `inputs/` directory of source documents and an `expected/` directory of markdown. The
`web` split nests its markdown one level deeper, under `expected/resiliparse/`, because the
extractor is part of the snapshot's identity.

## Running the tests

To run the snapshot tests, run `uv run pytest tests/test_snapshot.py`.

## Adding a test case

To add a test case, do the following:

* Add an html file to [`tests/snapshots/web/inputs/`](https://github.com/marin-community/marin/tree/main/tests/snapshots/web/inputs)
  (or the `inputs/` directory of whichever split you are extending; `stackexchange` takes
  `.json` rather than `.html`).
* Add the expected markdown output to
  [`tests/snapshots/web/expected/resiliparse/`](https://github.com/marin-community/marin/tree/main/tests/snapshots/web/expected/resiliparse),
  with the same name as the input file.
* Commit these files.

Pro-tip: rather than writing the expected file by hand, run
`uv run python tests/snapshots/generate_expected.py` to regenerate the expected outputs for
every split, then review and edit the result.

If it's reasonable, try to add a unit test as well. This will help ensure that
the conversion is correct. If you've made a change that you think is correct,
you can update the snapshots by copying `tests/snapshots/web/outputs/resiliparse/` over
`tests/snapshots/web/expected/resiliparse/`. This will overwrite the expected output with the
new output. You should review these changes before committing them.
