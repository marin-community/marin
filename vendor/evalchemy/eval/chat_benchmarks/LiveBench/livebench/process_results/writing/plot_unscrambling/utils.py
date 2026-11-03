import difflib
import re


def levenshtein_distance(A, B):
    N, M = len(A), len(B)
    # Create an array of size NxM
    dp = [[0 for i in range(M + 1)] for j in range(N + 1)]

    # Base Case: When N = 0
    for j in range(M + 1):
        dp[0][j] = j
    # Base Case: When M = 0
    for i in range(N + 1):
        dp[i][0] = i
    # Transitions
    for i in range(1, N + 1):
        for j in range(1, M + 1):
            if A[i - 1] == B[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])  # Insertion  # Deletion  # Replacement

    return dp[N][M]


def extract_plot_summary(text: str) -> str:
    pattern = r"<PLOT_SUMMARY>(.*)"
    match = re.search(pattern, text, re.DOTALL)  # re.DOTALL allows . to match newline characters
    return match.group(1) if match else text


def plot_unscrambling_process_results(ground_truth: str, llm_answer: str, debug=False) -> float:
    # Split the ground truth and answer into sentences based on full stops
    llm_answer = extract_plot_summary(llm_answer)

    gt_sentences = [s.strip() for s in ground_truth.split(".")]
    ans_sentences = [
        s.strip()
        for s in llm_answer.split(".")
        if s.strip() != "</PLOT_SUMMARY>" and s.strip() != "**End of Plot Summary**"
    ]

    # Remove empty sentences resulting from trailing or double full stops.
    gt_sentences = [s for s in gt_sentences if s]
    ans_sentences = [s for s in ans_sentences if s]

    ans_ordering = []
    for x in gt_sentences:
        best_match = difflib.get_close_matches(x, ans_sentences, n=1, cutoff=0.0)
        if best_match:
            ans_ordering.append(ans_sentences.index(best_match[0]))

    n_sentences_gt = len(gt_sentences)
    raw_distance = levenshtein_distance(list(range(len(gt_sentences))), ans_ordering)
    score = 1 - (raw_distance / n_sentences_gt)

    if debug and score < 1:
        print("INCORRECT", score)
        print("GROUND TRUTH", gt_sentences)
        print("SOLUTION", ans_sentences)
        print("END OF OUTPUT", llm_answer[-50:])
    return score
