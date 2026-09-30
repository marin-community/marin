# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Minimal stand-in for the daytona SDK's parameter classes.

dt.py imports these inside its functions and passes them to client(). The silo
client reads them by attribute, so plain attribute bags are enough to run the
real dt.py CLI in tests without the SDK installed.
"""


class Image:
    @staticmethod
    def from_dockerfile(path):
        image = Image()
        with open(path) as handle:
            image._dockerfile = handle.read()
        return image

    def dockerfile(self):
        return self._dockerfile


class _Params:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class Resources(_Params):
    pass


class CreateSnapshotParams(_Params):
    pass


class CreateSandboxFromSnapshotParams(_Params):
    pass


class SessionExecuteRequest(_Params):
    pass
