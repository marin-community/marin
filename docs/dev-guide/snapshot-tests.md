# Snapshot tests

We use snapshot tests to ensure that our HTML-to-text conversion is working as expected.

## Running the tests

To run the snapshot tests, run `pytest tests/test_snapshot.py`.

### Pandiff

We recommend installing pandiff to get a nice diff output for test failures.

To install, you'll need pandoc:

* MacOS: `brew install pandoc`
* Linux: `sudo apt-get install pandoc`

Then run `npm install -g pandiff`

## Adding a test case

To add a test case, do the following:

* Add an html file to [`tests/snapshots/web/inputs/`](https://github.com/marin-community/marin/tree/main/tests/snapshots/web/inputs)
* Add the expected markdown output to [`tests/snapshots/web/expected/`](https://github.com/marin-community/marin/tree/main/tests/snapshots/web/expected)
  with the same name as the input file.
* Commit these files.

Pro-tip: You can copy the markdown from `process_url.py`'s output to the
expected file and edit it as needed.

If it's reasonable, try to add a unit test as well. This will help ensure that
the conversion is correct. If you've made a change that you think is correct,
you can update the snapshots by copying `tests/snapshots/web/outputs/`
`tests/snapshots/web/expected/`. This will overwrite the expected output with the
new output. You should review these changes before committing them.
