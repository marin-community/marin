# EvalStore

`marin-evalstore` owns Marin's import-light evaluation record contracts and the
`EvaluationStore` adapter that writes them to FineStore. Producers, migrations, and
evaldash import the contract from `evalstore.archive`.

The package depends on `marin-finestore`, PyArrow, and Pydantic. It deliberately does
not depend on `marin-core`, which lets evaldash install the shared schema without
vendoring the full evaluation runtime.
