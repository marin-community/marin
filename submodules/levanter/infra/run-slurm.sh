#!/bin/bash
# This script is used for launching on a slurm cluster or other queueing systems
# This script assumes your virtual env or conda is already activated
set -e
LEV_ROOT=$(dirname "$(readlink -f $0)")/..

PYTHONPATH=${LEV_ROOT}:${LEV_ROOT}/src:${LEV_ROOT}/examples:$PYTHONPATH "$@" > >(tee -a stdout-$(hostname).log) 2> >(tee -a stderr-$(hostname).log >&2)
