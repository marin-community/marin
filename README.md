# Marin

> "*I am not afraid of storms, for I am learning how to sail my ship."*<br/>
> – Louisa May Alcott

## Getting Started

To get set up, create a new virtual environment (or `conda` environment) with the appropriate Python version (3.10),
then run the following:

```bash
git clone https://github.com/stanford-crfm/marin
cd marin
pip install -e ".[dev]"
```

This will install all the core dependencies and build `marin` as a Python package. Installing the `[dev]` requirements
will additionally install test, linting, and debugging dependencies (e.g., `pytest`).

---

## Organization

### Scripts

Scripts go in `scripts/$domain/`. Once there's a script for actually creating the domain, let's make a README.md in 
that directory that explains how to use the script.

### Source

Markdown conversion goes in `marin/markdown/`.

Library-y source code goes in `marin/$domain/`.


# Domains

## Web

### Working on Markdown conversion

My (@dlwh) workflow looks like this:

* `export PYTHONPATH=.:$PYTHONPATH`, or use an IDE that does this for you.
* find a web page that I'm concerned about
* run `python3 scripts/web/process_url.py <url>` 
* look at the outputs in `output/`. In particular compare `outputs/name.readability.html` to `outputs/name.md` to see what the conversion looks like.
* If you need to, make a gist of the md at https://gist.github.com/ and look at how GitHub renders it. This is the gold standard for what we're aiming for.

#### Adding a test case

We use snapshot testing in addition to unit tests. To add a test case, do the following:

* Add an html file to `tests/snapshots/inputs/` that you want to test.
* Add the expected markdown output to `tests/snapshots/expected/` with the same name as the input file.
* Commit these files.

Pro-tip: You can copy the markdown from `process_url.py`'s output to the expected file and edit it as needed.

If it's reasonable, try to add a unit test as well. This will help ensure that the conversion is correct.

#### Running tests

To run the tests, run `pytest` in the root directory. This will run the unit tests and snapshot tests.

#### Updating snapshots

If you've made a change that you think is correct, you can update the snapshots by copying `tests/snapshots/outputs/` to `tests/snapshots/expected/`. This will overwrite the expected output with the new output. You should review these changes before committing them.


## Wikipedia

TODO

## ArXiv

TODO

## Code

TODO

## Instruction Data

XXX

## Books

XXX

## StackExchange

XXX

## Reddit/Forums

XXX

## CC News?

XXX

## Semantic Scholar??

XXX

## PubMed

TODO

## Law

TODO