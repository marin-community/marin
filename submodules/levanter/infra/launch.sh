#!/bin/bash
# This script is used for launching on TPU pods (or other direct run environments) via remote ssh with a virtual env
set -e
umask 000
LEV_ROOT=$(dirname "$(readlink -f $0)")/..

# figure out venv, first check if we wrote a path in infra/venv_path
if [ ! -d "$VENV" ] && [ -f "$LEV_ROOT/infra/venv_path.txt" ]; then
  VENV=$(cat "$LEV_ROOT"/infra/venv_path.txt)
fi

# if we still don't have a venv, we'll look in our default
if [ ! -d "$VENV" ]; then
  VENV=/files/venv32
fi

if [ ! -d "$VENV" ]; then
  VENV=~/files/venv310
fi

source $VENV/bin/activate

PYTHONPATH=${LEV_ROOT}:${LEV_ROOT}/src:${LEV_ROOT}/examples:$PYTHONPATH nohup "$@" >& "~/log-$(hostname).log" &
