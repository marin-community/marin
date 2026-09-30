# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# Poison: first on the path in the dt.py subprocess, so any `import silo` fails.
# The generated dt.py must work from its embedded client alone.
raise ImportError("generated dt.py must not import the silo package")
