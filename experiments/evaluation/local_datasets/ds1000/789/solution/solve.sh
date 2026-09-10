#!/bin/sh
set -eu

# Write reference solution into the container solution file.
cat <<'PY' > /solution/solution.py
def dN1_dt(t, N1):
    input = 1-np.cos(t) if 0<t<2*np.pi else 0
    return -100*N1 + input
sol = scipy.integrate.solve_ivp(fun=dN1_dt, t_span=time_span, y0=[N0,])
PY
