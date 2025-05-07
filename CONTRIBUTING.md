# Contributing to Marin


## Components

From a repo-organizational perspective, Marin consists of two main components:

* The Marin Python package (this repo)
* The Marin data browser (in the `data_browser` directory of this repo)



# Marin

Once the server is started, go to
[http://localhost:5000](http://localhost:5000) and navigate around to the
experiment JSON file to get a nicer view of the experiment (the URL is also
printed out when you run the experiment).

See the [data browser README](data_browser/README.md) for more details.

## Setup

1. Clone the repository
2. Create and activate a virtual environment
3. Install the package
4. Set up pre-commit hooks

```bash
git clone https://github.com/stanford-crfm/marin.git
cd marin
python -m venv venv
source venv/bin/activate
pip install -e .[dev]
pre-commit install
```

Alternatively, you can install all the core dependencies and build `marin` as a Python
package with `make init`.

## Linting

Pre-commit hooks will run automatically lint your code when you commit (assuming you have run `pre-commit install`).
You can also run them manually with `make lint`.

## Testing

### Python Code

You can run the tests with `make test`.

*Note* that to run the unit tests, you must not have set `RAY_ADDRESS`. You can unset it with `unset RAY_ADDRESS` or `export RAY_ADDRESS=""`.


# Data browser

Marin comes with a [data browser](data_browser/) that makes it easy to
view datasets (in various formats) and experiments produced by the executor.
After installing the necessary dependencies, run:

```bash
cd data_browser
python server.py --config conf/local.conf
```


For more information, see the [data browser README](data_browser/README.md).
