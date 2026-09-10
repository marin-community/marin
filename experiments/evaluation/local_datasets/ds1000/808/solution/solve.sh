#!/bin/sh
set -eu

# Write reference solution into the container solution file.
cat <<'PY' > /solution/solution.py
def bekkers_cdf(x,a,m,d,range_start,range_end):
    values = []
    for value in x:
        integral = integrate.quad(lambda k: bekkers(k,a,m,d),range_start,value)[0]
        normalized = integral/integrate.quad(lambda k: bekkers(k,a,m,d),range_start,range_end)[0]
        values.append(normalized)
    return np.array(values)
result = stats.kstest(sample_data, lambda x: bekkers_cdf(x,estimated_a, estimated_m, estimated_d,range_start,range_end))
PY
