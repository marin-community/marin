from pathlib import Path
from .safe_subprocess import run


def eval_script(path):
    # Matlab has the requirement that all functions must appear at the end
    # of the file. So we first have to write the call to the test-function at the
    # beginning of the file.
    with open(path, "r") as f:
        content = f.read()
    content = f"test();\n{content}"
    with open(path, "w") as f:
        f.write(content)
    filename = path.stem
    parent_dir = path.parent.absolute()

    # We use the matlab.engine to run the script; however, the way that the
    # matlab engine works requires that we call the script as if it were a
    # member of the matlab.engine object. So we have to write a python script
    # that calls the matlab script. This also ensures that the script is called
    # in a safe-subprocess. Who needs runtime reflection when you have IPC?
    program = f"""
import matlab.engine
import io 
import sys
out = io.StringIO()
err = io.StringIO()
eng = matlab.engine.start_matlab()
eng.addpath(r'{parent_dir}',nargout=0)
try: 
    r = eng.{filename}(nargout=0, stdout=out,stderr=err)
    print(out.getvalue())
except matlab.engine.MatlabExecutionError as e:
    print(err.getvalue(), file=sys.stderr)
"""
    r = run(["python3", "-c", program], timeout_seconds=30)

    # This is still somewhat brittle.
    if r.timeout:
        status = "Timeout"
        exit_code = -1
    elif r.stderr == "":
        status = "OK"
        exit_code = 0
    else:
        status = "Exception"
        exit_code = 1

    return {
        "status": status,
        "exit_code": exit_code,
        "stdout": r.stdout,
        "stderr": r.stderr,
    }
