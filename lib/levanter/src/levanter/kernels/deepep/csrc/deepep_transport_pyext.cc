// Copyright The Levanter Authors
// SPDX-License-Identifier: Apache-2.0

#include <Python.h>

#define LEVANTER_STRINGIFY_IMPL(value) #value
#define LEVANTER_STRINGIFY(value) LEVANTER_STRINGIFY_IMPL(value)
#define LEVANTER_PYINIT_NAME_IMPL(name) PyInit_##name
#define LEVANTER_PYINIT_NAME(name) LEVANTER_PYINIT_NAME_IMPL(name)

#ifndef LEVANTER_DEEPEP_PYEXT_MODULE_NAME
#error "LEVANTER_DEEPEP_PYEXT_MODULE_NAME must be defined"
#endif

namespace {

PyModuleDef kModuleDef = {
    PyModuleDef_HEAD_INIT,
    LEVANTER_STRINGIFY(LEVANTER_DEEPEP_PYEXT_MODULE_NAME),
    nullptr,
    -1,
    nullptr,
};

}  // namespace

extern "C" PyMODINIT_FUNC LEVANTER_PYINIT_NAME(LEVANTER_DEEPEP_PYEXT_MODULE_NAME)(void) {
  return PyModule_Create(&kModuleDef);
}
