# Run Script

This script is desinged to run marin experiments in side `experiments/*.py`. It sets up the correct environment
dependencies for you. And you can specify any `pip` dependencies or env variables you want to set. It also correct sets
up the working dir and includes any subpath under submodules in the PYTHONPATH.

## Usage

If you earlier used the command

```bash
ray job submit --working-dir . -- python experiments/check_pip_packages_env_variables.py
```

Now you can run the same command with

```bash
python marin/run/run.py -- python experiments/check_pip_packages_env_variables.py
```

You can also specify depependencies and env variables like this:

```bash
python marin/run/run.py --env_vars HF_TOKEN hf_abcd --pip_deps s3fs,fsspec
-- python experiments/check_pip_packages_env_variables.py
```

The script will also automatically append submodules directory to the PYTHONPATH, so you can import them
directly in your scripts.

## Addtioanl Notes

- These dependencies are cached and will only be available in the current session.
- The best way to think about the depependencies is like virtual env for this current job on top of base cluster env.
