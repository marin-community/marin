# adapted from https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/minerva_math/utils.py


import re
import signal
from multiprocessing import Process, Queue
import multiprocessing.context

from eval.chat_benchmarks.LiveBench.livebench.process_results.util import last_boxed_only_string, remove_boxed

import warnings
import traceback

try:
    import sympy
    from sympy.parsing.latex import parse_latex
except ModuleNotFoundError:
    raise ModuleNotFoundError(
        "`sympy` is required for generating translation task prompt templates. \
please install sympy via pip install lm-eval[math] or pip install -e .[math]",
    )

try:
    import lark
except ModuleNotFoundError:
    raise ModuleNotFoundError(
        "`lark` is required for parsing latex. \
please install lark via pip install lark",
    )


def run_with_timeout(func, args=(), timeout=8):
    def wrapper(queue):
        try:
            result = func(*args)
            queue.put(result)
        except Exception as e:
            queue.put(e)

    queue = Queue()
    process = Process(target=wrapper, args=(queue,))
    process.start()
    process.join(timeout)

    if process.is_alive():
        process.terminate()
        process.join()
        raise TimeoutError("Operation timed out")

    result = queue.get()
    if isinstance(result, Exception):
        raise result
    return result


def amps_hard_process_results(ground_truth: str, llm_answer: str, debug=False) -> int:
    retval = 0
    parsed_answer = None

    if isinstance(ground_truth, list):
        ground_truth = ground_truth[-1]
    llm_answer = llm_answer.replace("+C", "")
    llm_answer = llm_answer.replace("+ C", "")
    llm_answer = llm_answer.replace("+ c", "")
    llm_answer = llm_answer.replace("+c", "")
    llm_answer = llm_answer.replace("\\\\fbox{", "\\\\boxed{")
    llm_answer = llm_answer.replace("\\dfrac", "\\frac")
    llm_answer = llm_answer.replace("\\tfrac", "\\frac")
    llm_answer = llm_answer.replace("\\left", "")
    llm_answer = llm_answer.replace("\\right", "")
    llm_answer = llm_answer.replace("\\bigl", "")
    llm_answer = llm_answer.replace("\\bigr", "")
    llm_answer = llm_answer.replace("\\Bigl", "")
    llm_answer = llm_answer.replace("\\Bigr", "")
    llm_answer = llm_answer.replace("\\,", "")
    llm_answer = llm_answer.replace("\\;", "")
    llm_answer = llm_answer.replace("\n", "")
    llm_answer = llm_answer.replace("\\cdot", "*")

    ground_truth = ground_truth.replace("\\left", "")
    ground_truth = ground_truth.replace("\\right", "")
    ground_truth = ground_truth.replace(" ^", "^")
    ground_truth = ground_truth.replace("\\ ", "*")

    last_boxed = last_boxed_only_string(llm_answer)
    if last_boxed:
        parsed_answer = normalize_final_answer(remove_boxed(last_boxed))

        # if is_equiv(ground_truth, parsed_answer):
        #     retval = 1
        try:
            res = run_with_timeout(is_equiv, args=(ground_truth, parsed_answer), timeout=8)
            if res:
                retval = 1
        except TimeoutError:
            warnings.warn("Timeout when comparing ground truth and parsed answer")
        except Exception as e:
            warnings.warn(f"Error when comparing ground truth and parsed answer: {e}")
    else:
        if llm_answer[-1] == ".":
            llm_answer = llm_answer[:-1]
        if ground_truth == llm_answer[-len(ground_truth) :]:
            parsed_answer = llm_answer[-len(ground_truth) :]
            retval = 1

    if debug and retval == 0:
        print("INCORRECT")
        print("GROUND TRUTH", ground_truth)
        if parsed_answer:
            print("SOLUTION", parsed_answer)
        print("END OF OUTPUT", llm_answer[-70:])
    return retval


# class timeout:
#     def __init__(self, seconds=1, error_message="Timeout"):
#         self.seconds = seconds
#         self.error_message = error_message

#     def handle_timeout(self, signum, frame):
#         print('Timeout')
#         raise TimeoutError(self.error_message)

#     def __enter__(self):
#         signal.signal(signal.SIGALRM, self.handle_timeout)
#         signal.alarm(self.seconds)

#     def __exit__(self, type, value, traceback):
#         signal.alarm(0)


def parse(x: str) -> list[sympy.Expr]:
    try:
        # first try to parse normally
        parsed_xs = parse_latex(x, backend="lark")
    except (sympy.parsing.latex.errors.LaTeXParsingError, sympy.SympifyError, TypeError, Exception):
        try:
            # this almost only happened for amazon.nova-pro-v1:0 where it outputs e.g. \\frac or \\sqrt all the time
            parsed_xs = parse_latex(x.replace("\\\\", "\\"), backend="lark")
        except:
            try:
                # if all else fails, try to parse using default backend
                parsed_xs = parse_latex(x)
            except:
                warnings.warn(f"couldn't parse {x}")
                return []

    if isinstance(parsed_xs, lark.Tree):
        # lark backend returns multiple options if there is ambiguity
        parsed_xs = parsed_xs.children
    else:
        parsed_xs = [parsed_xs]
    return parsed_xs


def is_equiv(x1: str, x2: str) -> bool:
    """
    x1 and x2 are normalized latex string
    """
    try:
        parsed_x1s = parse(x1)
        parsed_x2s = parse(x2)

        if len(parsed_x1s) == 0 or len(parsed_x2s) == 0:
            return False

        errors = []
        for parsed_x1 in parsed_x1s:
            for parsed_x2 in parsed_x2s:
                try:
                    diff = parsed_x1 - parsed_x2
                except Exception as e:
                    errors.append(f"couldn't subtract {x1} and {x2}: {e}")
                    continue

                try:
                    if sympy.simplify(diff) == 0:
                        return True
                except Exception as e:
                    errors.append(f"couldn't compare simplified {x1} - {x2} with 0: {e}")

                try:
                    if sympy.Abs(sympy.simplify(diff)) < 0.001:
                        return True
                except Exception as e:
                    errors.append(f"Had some trouble simplifying when comparing {x1} and {x2}: {e}")
        for error in errors:
            warnings.warn(error)
        return False
    except ImportError as e:
        warnings.warn(e)
        raise
    except Exception as e:
        warnings.warn(f"Failed comparing {x1} and {x2}: {e}")
        traceback.print_tb(e.__traceback__)
        return False


def normalize_final_answer(final_answer: str) -> str:
    """
    Normalize a final answer to a quantitative reasoning question.

    Copied character for character from appendix D of Lewkowycz et al. (2022)
    """
    final_answer = final_answer.split("=")[-1]

    # Extract answer that is in LaTeX math, is bold,
    # is surrounded by a box, etc.
    final_answer = re.sub(r"(.*?)(\$)(.*?)(\$)(.*)", "$\\3$", final_answer)
    final_answer = re.sub(r"(\\text\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\textbf\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\overline\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\boxed\{)(.*)(\})", "\\2", final_answer)

    # Normalize shorthand TeX:
    #  \fracab -> \frac{a}{b}
    #  \frac{abc}{bef} -> \frac{abc}{bef}
    #  \fracabc -> \frac{a}{b}c
    #  \sqrta -> \sqrt{a}
    #  \sqrtab -> sqrt{a}b
    final_answer = re.sub(r"(frac)([^{])(.)", "frac{\\2}{\\3}", final_answer)
    final_answer = re.sub(r"(sqrt)([^{\[])", "sqrt{\\2}", final_answer)
    final_answer = final_answer.replace("$", "")

    # Normalize 100,000 -> 100000
    if final_answer.replace(",", "").isdigit():
        final_answer = final_answer.replace(",", "")

    return final_answer
