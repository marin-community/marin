"""Ray worker exec prelude; preserve its Python executable and import paths."""

import os
import sys

from marinskyrl.native_environment_control import prepare_worker_process


def main() -> None:
    # No interpreter, package or actor ownership substitution. Replace only
    # declared GPU/runtime env values before Ray imports the actor class.
    prepare_worker_process()
    os.execv(sys.executable, [sys.executable, *sys.argv[1:]])


if __name__ == "__main__":
    main()
