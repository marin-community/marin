#!/usr/bin/env bash
# Guarded `pulumi up`: confirm before updating a main-only stack from a non-`main` branch.
#
# Production clusters and shared-GCP stacks should be updated from `main` after a change merges.
# Stacks that follow that rule carry `marin-iac:main-only: "true"` in their `Pulumi.<stack>.yaml`.
#
# This wrapper runs in the operator's own shell, so unlike a check inside the Pulumi program it
# can prompt on the real terminal *before* `pulumi` starts its preview — `pulumi up` runs the
# program twice (preview then update), so an in-program guard only speaks up after the diff is
# already on screen and cannot reach the terminal. It is a soft guard, not a gate: declining
# aborts, and a bare `pulumi up` still bypasses it entirely.
set -euo pipefail

cd "$(dirname "$0")"

MAIN_BRANCH=main
branch=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || true)

# Resolve the target stack: an explicit --stack/-s on the command line wins, else the selected one.
stack=""
prev=""
for arg in "$@"; do
  case "$prev" in
    --stack | -s) stack=$arg ;;
  esac
  case "$arg" in
    --stack=*) stack=${arg#--stack=} ;;
  esac
  prev=$arg
done
[ -n "$stack" ] || stack=$(pulumi stack --show-name 2>/dev/null || true)

main_only=no
if [ -n "$stack" ] && grep -Eq '^[[:space:]]*marin-iac:main-only:[[:space:]]*"?true"?[[:space:]]*$' \
  "Pulumi.$stack.yaml" 2>/dev/null; then
  main_only=yes
fi

if [ "$main_only" = yes ] && [ -n "$branch" ] && [ "$branch" != "$MAIN_BRANCH" ]; then
  echo "⚠️  Stack '$stack' is main-only, but you are on branch '$branch', not '$MAIN_BRANCH'."
  echo "    Production stacks should be deployed from '$MAIN_BRANCH' after merging."
  if [ -t 0 ]; then
    printf "    Continue with \`pulumi up\` anyway? [y/N] "
    read -r answer || answer=""
    case "$answer" in
      y | Y | yes | YES) ;;
      *)
        echo "Aborted."
        exit 1
        ;;
    esac
  else
    # No terminal to ask on (e.g. piped input); warn and proceed so automation is not blocked.
    echo "    No terminal to confirm on; proceeding."
  fi
fi

exec pulumi up "$@"
