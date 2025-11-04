umask 000
LEV_ROOT=$(dirname "$(readlink -f $0)")/..
ulimit -s 65536

# figure out venv, first check if we wrote a path in infra/venv_path
if [ ! -d "$VENV" ] && [ -f "$LEV_ROOT/infra/venv_path.txt" ]; then
  VENV=$(cat "$LEV_ROOT"/infra/venv_path.txt)
fi

# if we still don't have a venv, we'll look in our default
if [ ! -d "$VENV" ]; then
  VENV=$LEV_ROOT/.venv
fi

source $VENV/bin/activate


PYTHONPATH=${LEV_ROOT}:${LEV_ROOT}/src:${LEV_ROOT}/examples:$PYTHONPATH "$@"
