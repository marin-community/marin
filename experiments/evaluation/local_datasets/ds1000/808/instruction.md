Problem:
I have an array of experimental values and a probability density function that supposedly describes their distribution:
def bekkers(x, a, m, d):
    p = a*np.exp((-1*(x**(1/3) - m)**2)/(2*d**2))*x**(-2/3)
    return(p)
I estimated the parameters of my function using scipy.optimize.curve_fit and now I need to somehow test the goodness of fit. I found a scipy.stats.kstest function which suposedly does exactly what I need, but it requires a continuous distribution function. 
How do I get the result (statistic, pvalue) of KStest? I have some sample_data from fitted function, and parameters of it.
A:
<code>
import numpy as np
import scipy as sp
from scipy import integrate,stats
def bekkers(x, a, m, d):
    p = a*np.exp((-1*(x**(1/3) - m)**2)/(2*d**2))*x**(-2/3)
    return(p)
range_start = 1
range_end = 10
estimated_a, estimated_m, estimated_d = 1,1,1
sample_data = [1.5,1.6,1.8,2.1,2.2,3.3,4,6,8,9]
</code>
result = ... # put solution in this variable
BEGIN SOLUTION
<code>

Execution context (contains [insert] placeholder):
import numpy as np
import copy
from scipy import integrate, stats


def generate_test_case(test_case_id):
    def define_test_input(test_case_id):
        if test_case_id == 1:
            range_start = 1
            range_end = 10
            estimated_a, estimated_m, estimated_d = 1, 1, 1
            sample_data = [1.5, 1.6, 1.8, 2.1, 2.2, 3.3, 4, 6, 8, 9]
        return (
            range_start,
            range_end,
            estimated_a,
            estimated_m,
            estimated_d,
            sample_data,
        )

    def generate_ans(data):
        _a = data

        def bekkers(x, a, m, d):
            p = a * np.exp((-1 * (x ** (1 / 3) - m) ** 2) / (2 * d**2)) * x ** (-2 / 3)
            return p

        range_start, range_end, estimated_a, estimated_m, estimated_d, sample_data = _a

        def bekkers_cdf(x, a, m, d, range_start, range_end):
            values = []
            for value in x:
                integral = integrate.quad(
                    lambda k: bekkers(k, a, m, d), range_start, value
                )[0]
                normalized = (
                    integral
                    / integrate.quad(
                        lambda k: bekkers(k, a, m, d), range_start, range_end
                    )[0]
                )
                values.append(normalized)
            return np.array(values)

        result = stats.kstest(
            sample_data,
            lambda x: bekkers_cdf(
                x, estimated_a, estimated_m, estimated_d, range_start, range_end
            ),
        )
        return result

    test_input = define_test_input(test_case_id)
    expected_result = generate_ans(copy.deepcopy(test_input))
    return test_input, expected_result


def exec_test(result, ans):
    assert np.allclose(result, ans)
    return 1


exec_context = r"""
import numpy as np
import scipy as sp
from scipy import integrate,stats
def bekkers(x, a, m, d):
    p = a*np.exp((-1*(x**(1/3) - m)**2)/(2*d**2))*x**(-2/3)
    return(p)
range_start, range_end, estimated_a, estimated_m, estimated_d, sample_data = test_input
[insert]
"""


def test_execution(solution: str):
    code = exec_context.replace("[insert]", solution)
    for i in range(1):
        test_input, expected_result = generate_test_case(i + 1)
        test_env = {"test_input": test_input}
        exec(code, test_env)
        assert exec_test(test_env["result"], expected_result)

Return only the Python code snippet that replaces [insert]:
- No explanations, Markdown, comments, prints, or extra text; no non-English output
- No imports, function/class definitions, or placeholders; only the missing logic
- Do not reassign/reset provided variables/inputs; use them to compute the result
- The snippet must be syntactically valid and run as-is in this context; deterministic (assume temperature=0)
- Do not read/write files or call external services
- Output only the code lines for [insert]; no Markdown fences, no explanations, no bullets.
