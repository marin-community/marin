import os
import platform

platform.machine = lambda: os.environ["FIXTURE_ARCHITECTURE"]
