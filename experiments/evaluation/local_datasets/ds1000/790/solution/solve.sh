#!/bin/sh
set -eu

# Write reference solution into the container solution file.
cat <<'PY' > /solution/solution.py
def dN1_dt (t, N1):
    return -100 * N1 + np.sin(t)
sol = scipy.integrate.solve_ivp(fun=dN1_dt, t_span=time_span, y0=[N0,])

PY
